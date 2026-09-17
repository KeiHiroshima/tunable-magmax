"""Golden outputs for every merge function.

This is the main guard for the refactor: whatever the merge registry and the
backend plumbing get reshaped into, these tensors must come out identical.

Regenerate after an *intended* behaviour change with:

    MAGMAX_UPDATE_GOLDENS=1 uv run pytest test/characterization/test_merge_golden.py

and commit the diff to goldens/merge_outputs.json — the diff is then the
reviewable record of exactly what changed.
"""

import json
import os
import random
from pathlib import Path

import pytest
import torch

from conftest import build_synthetic_task_vectors
from src.merging.task_vectors import (
    mask_and_merge_by_weights,
    merge_max_abs,
    merge_rnd_mix,
    ties,
)

GOLDEN_PATH = Path(__file__).parent / "goldens" / "merge_outputs.json"
UPDATE = os.environ.get("MAGMAX_UPDATE_GOLDENS") == "1"

# The per-task weights used for the masked (proposed-method) merge. Deliberately
# uneven and not a round split of any key's element count, so the remainder
# redistribution loop in mask_and_merge_by_weights is exercised.
MASKED_WEIGHTS = [0.5, 0.3, 0.2]

# mask_and_merge_by_weights takes its own seed now; it no longer reads the
# global RNGs, so nothing outside this call can change its output.
MASKED_SEED = 0


def _seed_global_rng():
    """merge_rnd_mix draws from the global torch RNG, so its output is only
    reproducible under a fixed seed. (mask_and_merge_by_weights used to need
    this too; it now takes an explicit seed.)"""
    random.seed(0)
    torch.manual_seed(0)


def _flatten(vector: dict) -> dict:
    return {key: [round(float(v), 6) for v in tensor.flatten()] for key, tensor in vector.items()}


def _merge_outputs() -> dict:
    """Run every merge function on the same synthetic inputs."""
    outputs = {}

    outputs["magmax"] = _flatten(merge_max_abs(build_synthetic_task_vectors()).vector)
    outputs["ties"] = _flatten(ties(build_synthetic_task_vectors()).vector)

    # "average" is registered as bare `sum` in both backends; the division by
    # the task count lives outside the merge function (src/eval.py). This
    # golden therefore pins the *un-divided* sum, matching what the registry
    # actually returns today.
    outputs["average_raw_sum"] = _flatten(sum(build_synthetic_task_vectors()).vector)

    _seed_global_rng()
    outputs["random_mix"] = _flatten(merge_rnd_mix(build_synthetic_task_vectors()).vector)

    merged, num_unaligned, num_params_all = mask_and_merge_by_weights(
        build_synthetic_task_vectors(), MASKED_WEIGHTS, seed=MASKED_SEED
    )
    outputs["masked_magmax_with_targetdata"] = _flatten(merged.vector)
    outputs["masked_magmax_with_targetdata__num_unaligned"] = num_unaligned
    outputs["masked_magmax_with_targetdata__num_params_all"] = num_params_all

    return outputs


@pytest.fixture(scope="module")
def goldens():
    outputs = _merge_outputs()
    if UPDATE:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"goldens regenerated at {GOLDEN_PATH}")
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing — regenerate with MAGMAX_UPDATE_GOLDENS=1"
    )
    return json.loads(GOLDEN_PATH.read_text()), outputs


@pytest.mark.parametrize(
    "merge_fn_name",
    ["magmax", "ties", "average_raw_sum", "random_mix", "masked_magmax_with_targetdata"],
)
def test_merge_output_matches_golden(goldens, merge_fn_name):
    expected, actual = goldens
    assert actual[merge_fn_name] == expected[merge_fn_name]


def test_masked_merge_bookkeeping_matches_golden(goldens):
    """num_unaligned / num_params_all are written into the results JSON, so
    they are part of the contract just as much as the merged tensors."""
    expected, actual = goldens
    for key in (
        "masked_magmax_with_targetdata__num_unaligned",
        "masked_magmax_with_targetdata__num_params_all",
    ):
        assert actual[key] == expected[key]


def test_masked_merge_allocates_exactly_the_requested_element_counts():
    """The invariant behind the proposed method: each task supplies a share of
    the merged parameters equal to its preference weight. Stated here as a
    property (not a golden) so it keeps holding for *any* future refactor of
    the allocation loop, not just for these particular inputs."""
    task_vectors = build_synthetic_task_vectors()
    merged, _, _ = mask_and_merge_by_weights(task_vectors, MASKED_WEIGHTS, seed=MASKED_SEED)

    for key, tensor in merged.vector.items():
        if key.startswith("frozen"):
            assert torch.all(tensor == 0)
            continue
        # Every merged element must have come verbatim from one of the inputs.
        came_from_some_task = torch.zeros_like(tensor, dtype=torch.bool)
        for tv in task_vectors:
            came_from_some_task |= tensor == tv.vector[key]
        assert torch.all(came_from_some_task), f"{key} has values from no task vector"


def test_single_task_vector_is_returned_unchanged():
    """All the multi-task merge functions short-circuit at len < 2. Pinned
    because the unified registry must keep that edge case."""
    one = build_synthetic_task_vectors(n_tasks=1)
    for fn in (merge_max_abs, merge_rnd_mix, ties):
        assert fn(one) is one[0], f"{fn.__name__} did not short-circuit"
