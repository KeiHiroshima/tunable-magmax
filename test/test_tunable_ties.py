"""Tunable TIES (`--merge_fn weighted_ties_with_targetdata`).

TIES-Merging with each task's say in the sign election and in the disjoint
mean weighted by the target environment's preference vector:

    sign_p  = sign(sum_i w_i * trim(tau_i)[p])
    tau[p]  = sum_{i agreeing} w_i * trim(tau_i)[p] / sum_{i agreeing} w_i

The tests pin what that definition promises: uniform weights are plain TIES,
a task weighted 0 takes no part, only the weights' ratios matter, and the
weights really do decide the sign and the mean.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from src.merging import task_vectors as tv_module
from src.merging.registry import get_merge_spec
from src.merging.task_vector import TaskVector
from src.merging.task_vectors import (
    masked_magmax_by_weights,
    merge_with_targetdata,
    ties,
    weighted_ties_by_weights,
)
from src.merging.ties import state_dict_to_vector, topk_values_mask

SHAPES = {"layer1.weight": (4, 5), "layer1.bias": (4,), "layer2.weight": (3, 4)}


def _task_vectors(n=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        TaskVector(
            vector={k: torch.randn(s, generator=generator) for k, s in SHAPES.items()}
        )
        for _ in range(n)
    ]


def _constant(values):
    """One task vector per value, every entry equal to it: equal magnitudes,
    so TIES's top-20% pruning keeps every entry and only the sign election and
    the mean are exercised."""
    return [
        TaskVector(vector={k: torch.full(s, float(v)) for k, s in SHAPES.items()})
        for v in values
    ]


def _flat(task_vector):
    return state_dict_to_vector(task_vector.vector)


def test_uniform_weights_are_plain_ties():
    task_vectors = _task_vectors()
    plain = _flat(ties(task_vectors))
    # Weights of 1 are the very computation plain TIES does, bit for bit...
    assert torch.equal(_flat(ties(task_vectors, weights=[1.0] * 3)), plain)
    # ...any other uniform weights only up to rounding.
    assert torch.allclose(_flat(ties(task_vectors, weights=[1 / 3] * 3)), plain, atol=1e-6)


def test_only_the_ratio_of_the_weights_matters():
    task_vectors = _task_vectors()
    a = ties(task_vectors, weights=[0.5, 0.3, 0.2])
    b = ties(task_vectors, weights=[5.0, 3.0, 2.0])
    assert torch.allclose(_flat(a), _flat(b))


def test_a_task_weighted_zero_takes_no_part():
    task_vectors = _task_vectors()
    other = _task_vectors(seed=1)[1]
    swapped = [task_vectors[0], other, task_vectors[2]]

    weights = [0.6, 0.0, 0.4]
    assert torch.equal(
        _flat(ties(task_vectors, weights=weights)), _flat(ties(swapped, weights=weights))
    )
    # ...and is the same as leaving it out
    assert torch.equal(
        _flat(ties(task_vectors, weights=weights)),
        _flat(ties([task_vectors[0], task_vectors[2]], weights=[0.6, 0.4])),
    )


def test_one_task_weighted_alone_gives_its_pruned_task_vector():
    task_vectors = _task_vectors()
    merged = ties(task_vectors, weights=[0.0, 1.0, 0.0])
    pruned, _ = topk_values_mask(_flat(task_vectors[1])[None, :], K=20)
    assert torch.equal(_flat(merged), pruned[0])


def test_weights_decide_the_sign():
    # +1 weighted 0.9 against -3 weighted 0.1: 0.9 - 0.3 > 0, so "+" wins and
    # only the +1 entries are averaged. Plain TIES elects "-" (1 - 3 < 0).
    task_vectors = _constant([1.0, -3.0])
    assert torch.all(_flat(ties(task_vectors, weights=[0.9, 0.1])) == 1.0)
    assert torch.all(_flat(ties(task_vectors)) == -3.0)


def test_weights_decide_the_mean():
    task_vectors = _constant([1.0, 3.0])
    weighted = _flat(ties(task_vectors, weights=[0.75, 0.25]))
    assert torch.allclose(weighted, torch.full_like(weighted, 0.75 * 1 + 0.25 * 3))
    plain = _flat(ties(task_vectors))
    assert torch.allclose(plain, torch.full_like(plain, 2.0))


def test_weighted_fn_has_no_diagnostics_and_ignores_the_seed():
    task_vectors = _task_vectors()
    weights = [0.5, 0.3, 0.2]
    merged, diagnostics = weighted_ties_by_weights(task_vectors, weights, seed=1)
    again, _ = weighted_ties_by_weights(task_vectors, weights, seed=2)
    assert diagnostics == {}
    assert torch.equal(_flat(merged), _flat(again))
    assert torch.equal(_flat(merged), _flat(ties(task_vectors, weights=weights)))


def test_registered_as_a_preference_weighted_merge():
    spec = get_merge_spec("weighted_ties_with_targetdata")
    assert spec.needs_target_data
    assert spec.weighted_fn is weighted_ties_by_weights
    assert spec.label == "Tunable TIES"


# --- vision: merge_with_targetdata -----------------------------------------


def _vision_args(tmp_path, merge_fn):
    return SimpleNamespace(
        seed=3, target_id=4, results_db=str(tmp_path), merge_fn=merge_fn,
        merge_space="full", similarity_metric="labels", coeff=0.5,
    )


@pytest.mark.parametrize(
    "merge_fn,weighted_fn,diagnostics",
    [
        ("weighted_ties_with_targetdata", weighted_ties_by_weights, set()),
        ("masked_magmax_with_targetdata", masked_magmax_by_weights, {"num_unaligned", "num_params_all"}),
    ],
)
def test_vision_merges_by_the_inferred_preference_vector(
    tmp_path, monkeypatch, merge_fn, weighted_fn, diagnostics
):
    """Both merges share one flow: infer the preference vector, merge by it
    with the spec's weighted_fn, and record the merge's diagnostics (if any)
    in the results file eval.py completes afterwards."""
    weights = [0.7, 0.0, 0.3]
    monkeypatch.setattr(tv_module, "compute_preference_weights", lambda *a, **k: weights)
    task_vectors = _task_vectors()
    args = _vision_args(tmp_path, merge_fn)

    merged, returned_weights, metric = merge_with_targetdata(
        get_merge_spec(merge_fn).weighted_fn, task_vectors, None, None,
        similarity_metric="labels", args=args, pretrained_checkpoint="unused",
    )

    expected, _ = weighted_fn(task_vectors, weights, seed=tv_module.derive_seed(3, 4))
    assert torch.equal(_flat(merged), _flat(expected))
    assert returned_weights == weights and metric == "labels"

    log_path = tmp_path / merge_fn / "labels" / f"{merge_fn}_lambda0.5_labels_target4_seed3.json"
    if diagnostics:
        assert set(json.loads(log_path.read_text())) == diagnostics
    else:
        assert not log_path.exists()
