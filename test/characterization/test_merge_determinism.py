"""mask_and_merge_by_weights must be a function of its arguments alone.

Algorithm 1 breaks ties at random twice: when a task wins more elements than
its preference budget allows it has to give some back (line 6), and any
elements still unclaimed at the end are scattered across the tasks that are
short (line 12). Those draws used to come from the process-wide `random` and
`torch` RNGs, which made the merged model depend on everything that had drawn
from them earlier — so the model built for the fifth target environment
differed between a full run and a run resumed after four finished targets, and
between one `--merge_fn` and another.

The function now takes a seed. These tests pin what that buys: the same
arguments give the same model, whatever else the process has done.
"""

import random

import pytest
import torch

from src.merging.task_vector import TaskVector
from src.merging.task_vectors import mask_and_merge_by_weights
from src.utils import derive_seed

KEYS = {"layer1.weight": (6, 7), "layer1.bias": (6,), "layer2.weight": (5, 6)}
WEIGHTS = [0.5, 0.3, 0.2]
KEY = "layer1.weight"

# Task 2 is given almost the whole budget while the magnitudes stay random, so
# it cannot fill it from wins alone and the leftover-scattering branch runs.
LOPSIDED_WEIGHTS = [0.05, 0.9, 0.05]


def build_task_vectors(n_tasks=3, data_seed=0):
    generator = torch.Generator().manual_seed(data_seed)
    return [
        TaskVector(
            vector={k: torch.randn(shape, generator=generator) for k, shape in KEYS.items()}
        )
        for _ in range(n_tasks)
    ]


def _disturb_global_rngs():
    random.seed(999)
    torch.manual_seed(999)
    for _ in range(50):
        torch.randn(10)
        random.random()


def test_same_seed_gives_the_same_merge():
    first, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)
    second, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)

    assert torch.equal(first.vector[KEY], second.vector[KEY])


def test_merge_ignores_the_global_rngs():
    random.seed(1)
    torch.manual_seed(1)
    baseline, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)

    _disturb_global_rngs()
    disturbed, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)

    assert torch.equal(baseline.vector[KEY], disturbed.vector[KEY])


def test_merge_is_unaffected_by_preceding_merges():
    """The concrete failure this prevents. src/eval.py merges once per target
    environment in a loop, and skips targets whose results already exist — so
    while the tie-breaks came from the global RNGs, whether a target had been
    evaluated before changed the model built for the ones after it.
    """
    random.seed(1)
    torch.manual_seed(1)
    alone, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)

    random.seed(1)
    torch.manual_seed(1)
    for other_seed in (11, 22, 33):
        mask_and_merge_by_weights(build_task_vectors(), [0.2, 0.5, 0.3], seed=other_seed)
    after_others, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)

    assert torch.equal(alone.vector[KEY], after_others.vector[KEY])


def test_different_seeds_can_give_different_merges():
    """The seed has to actually reach the draws; if it were ignored, every test
    above would pass trivially."""
    first, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=7)
    second, _, _ = mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, seed=12345)

    assert not torch.equal(first.vector[KEY], second.vector[KEY])


def test_seed_is_keyword_only():
    """Positional would be easy to pass by accident where a caller means the
    target id or the run seed."""
    with pytest.raises(TypeError):
        mask_and_merge_by_weights(build_task_vectors(), WEIGHTS, 7)


# --- the leftover-scattering branch ----------------------------------------


def test_leftover_scattering_branch_is_reached():
    """Guards the tests below: if a change stopped this input from producing
    leftovers, they would silently stop covering the branch they exist for."""
    _, num_unaligned, _ = mask_and_merge_by_weights(
        build_task_vectors(), LOPSIDED_WEIGHTS, seed=0
    )

    assert any(v > 0 for v in num_unaligned.values()), num_unaligned


def test_leftover_scattering_is_deterministic():
    """This branch picks from a Python set. Set iteration order is a hash-table
    detail rather than something this code fixes, so the candidates are sorted
    before sampling."""
    first, first_unaligned, _ = mask_and_merge_by_weights(
        build_task_vectors(), LOPSIDED_WEIGHTS, seed=3
    )
    _disturb_global_rngs()
    second, second_unaligned, _ = mask_and_merge_by_weights(
        build_task_vectors(), LOPSIDED_WEIGHTS, seed=3
    )

    assert torch.equal(first.vector[KEY], second.vector[KEY])
    assert first_unaligned == second_unaligned


# --- how callers derive the seed -------------------------------------------


def test_derive_seed_is_stable_across_processes():
    """Built with crc32 rather than hash(): Python randomises string hashing
    per process, which would give a different merge on every run."""
    assert derive_seed(3, 12) == derive_seed(3, 12)


@pytest.mark.parametrize(
    "a,b", [((3, 12), (3, 13)), ((3, 12), (4, 12)), ((3, 12), (12, 3))]
)
def test_derive_seed_separates_runs_and_targets(a, b):
    """Both callers pass derive_seed(args.seed, target_id): different target
    environments in a run, and the same target across the paper's three seeds,
    must all get independent tie-breaking."""
    assert derive_seed(*a) != derive_seed(*b)


def test_merge_seeds_from_the_same_run_differ_per_target():
    run_seed = 3
    merges = {
        target_id: mask_and_merge_by_weights(
            build_task_vectors(), WEIGHTS, seed=derive_seed(run_seed, target_id)
        )[0].vector[KEY]
        for target_id in range(1, 5)
    }

    distinct = {tensor.numpy().tobytes() for tensor in merges.values()}
    assert len(distinct) > 1
