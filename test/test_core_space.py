"""src/merging/core_space.py against the reference implementation of Core
Space merging (apanariello4/core-space-merging, task_merger.py's
MatrixPerLayerMerger.get_core_matrices), on synthetic LoRA adapters in the
format src/lora.py saves."""

import pytest
import torch

from src.lora import adapter_task_vectors
from src.merging.core_space import CoreSpace
from src.merging.task_vector import TaskVector

R_LORA = 8
ALPHA = 32
# (out_features, in_features): one square weight like CLIP's q/v, one
# rectangular one so a transposed U/V mix-up cannot pass.
SHAPES = {"blocks.0.attn.q": (48, 48), "blocks.0.attn.v": (40, 56)}


def make_adapters(n_tasks, shapes=SHAPES, r=R_LORA, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "config": {"r": r, "alpha": ALPHA, "dropout": 0.1, "target_modules": ["q", "v"]},
            "modules": {
                name: {
                    "A": torch.randn(r, d_in, generator=generator, dtype=torch.float64),
                    "B": torch.randn(d_out, r, generator=generator, dtype=torch.float64),
                }
                for name, (d_out, d_in) in shapes.items()
            },
        }
        for _ in range(n_tasks)
    ]


def reference_core_matrices(A_list, B_list):
    """get_core_matrices from the reference implementation, verbatim apart
    from naming: per-task SVDs of A and B, aligned by R_B and Q_A."""
    Vh_A_ref = torch.linalg.svd(torch.cat(A_list, dim=0), full_matrices=False)[2]
    U_B_ref = torch.linalg.svd(torch.cat(B_list, dim=1), full_matrices=False)[0]
    M_list = []
    for A, B in zip(A_list, B_list):
        U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)
        U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)
        Q_A = Vh_A @ Vh_A_ref.T
        R_B = U_B_ref.T @ U_B
        M = torch.diag(S_B) @ (Vh_B @ U_A) @ torch.diag(S_A)
        M_list.append(R_B @ M @ Q_A)
    return M_list, U_B_ref, Vh_A_ref


def task_vectors_from(adapters, sequential):
    return [TaskVector(vector=v) for v in adapter_task_vectors(adapters, sequential)]


def test_core_matrices_from_dense_task_vectors_match_the_reference_formula():
    """Plan step 5 computes U_ref^T tau_i V_ref from the dense task vector; the
    reference builds the same matrix from per-task SVDs of A_i and B_i."""
    adapters = make_adapters(n_tasks=4)
    space = CoreSpace.from_adapters(adapters)
    cores = space.to_core(task_vectors_from(adapters, sequential=False))

    scaling = ALPHA / R_LORA
    for name in SHAPES:
        key = f"{name}.weight"
        A_list = [a["modules"][name]["A"] for a in adapters]
        B_list = [a["modules"][name]["B"] * scaling for a in adapters]
        ref_M, ref_U, ref_Vh = reference_core_matrices(A_list, B_list)

        U_ref, V_ref = space.bases[key]
        torch.testing.assert_close(U_ref, ref_U, rtol=0, atol=1e-12)
        torch.testing.assert_close(V_ref, ref_Vh.T, rtol=0, atol=1e-12)
        for core, M in zip(cores, ref_M):
            assert core.vector[key].shape == (4 * R_LORA, 4 * R_LORA)
            torch.testing.assert_close(core.vector[key], M, rtol=0, atol=1e-10)


@pytest.mark.parametrize("sequential", [False, True])
def test_round_trip_through_core_space_is_lossless(sequential):
    adapters = make_adapters(n_tasks=4)
    space = CoreSpace.from_adapters(adapters)
    task_vectors = task_vectors_from(adapters, sequential)

    for tv, core in zip(task_vectors, space.to_core(task_vectors)):
        back = space.from_core(core)
        for key, tau in tv.vector.items():
            torch.testing.assert_close(back.vector[key], tau, rtol=0, atol=1e-9)


def test_round_trip_is_lossless_when_stacked_rank_exceeds_the_weight():
    """T * r > d_out, d_in: each basis is capped at its side's dimension
    (min(d, R) columns), so the core matrix is d_out x d_in here, and the
    bases still span every task vector."""
    shapes = {"blocks.0.attn.q": (24, 20)}
    adapters = make_adapters(n_tasks=4, shapes=shapes)  # T * r = 32
    space = CoreSpace.from_adapters(adapters)
    U_ref, V_ref = space.bases["blocks.0.attn.q.weight"]
    assert U_ref.shape == (24, 24) and V_ref.shape == (20, 20)

    task_vectors = task_vectors_from(adapters, sequential=True)
    for tv, core in zip(task_vectors, space.to_core(task_vectors)):
        assert core.vector["blocks.0.attn.q.weight"].shape == (24, 20)
        back = space.from_core(core)
        for key, tau in tv.vector.items():
            torch.testing.assert_close(back.vector[key], tau, rtol=0, atol=1e-9)


def test_lora_scaling_does_not_change_the_spanned_bases():
    """alpha / r is folded into B; a common scalar must leave the subspace
    (hence every core matrix up to that scalar) unchanged."""
    adapters = make_adapters(n_tasks=3)
    unscaled = [
        {**a, "config": {**a["config"], "alpha": a["config"]["r"]}} for a in adapters
    ]
    scaled_space = CoreSpace.from_adapters(adapters)
    unscaled_space = CoreSpace.from_adapters(unscaled)
    for key, (U, V) in scaled_space.bases.items():
        U0, V0 = unscaled_space.bases[key]
        torch.testing.assert_close(U @ U.T, U0 @ U0.T, rtol=0, atol=1e-10)
        torch.testing.assert_close(V @ V.T, V0 @ V0.T, rtol=0, atol=1e-10)


def test_float32_task_vectors_stay_float32():
    adapters = make_adapters(n_tasks=2)
    space = CoreSpace.from_adapters(adapters)
    tv = TaskVector(
        vector={k: v.float() for k, v in adapter_task_vectors(adapters, False)[0].items()}
    )
    core = space.to_core([tv])[0]
    assert all(t.dtype == torch.float32 for t in core.vector.values())
    assert all(t.dtype == torch.float32 for t in space.from_core(core).vector.values())


def test_weights_without_a_basis_are_rejected():
    space = CoreSpace.from_adapters(make_adapters(n_tasks=2))
    with pytest.raises(KeyError, match="no Core Space basis"):
        space.to_core([TaskVector(vector={"head.weight": torch.zeros(3, 3)})])


# --- merging in Core Space ---------------------------------------------------


def test_average_in_core_space_equals_average_in_full_space():
    """A linear merge commutes with the rotation (the paper's eq. 11), so
    --merge_space core must not change "average" at all."""
    from src.merging.registry import apply_merge, get_merge_spec

    adapters = make_adapters(n_tasks=4)
    space = CoreSpace.from_adapters(adapters)
    task_vectors = task_vectors_from(adapters, sequential=True)
    spec = get_merge_spec("average")

    full = apply_merge(spec, task_vectors)
    core = apply_merge(spec, task_vectors, core_space=space)
    for key, value in full.vector.items():
        torch.testing.assert_close(core.vector[key], value, rtol=0, atol=1e-9)


@pytest.mark.parametrize("merge_fn", ["magmax", "ties", "random_mix"])
def test_nonlinear_merges_return_full_space_weights(merge_fn):
    """Element-wise merges run on the core matrices but hand back d_out x d_in
    deltas that still lie in the adapters' span."""
    from src.merging.registry import apply_merge, get_merge_spec

    adapters = make_adapters(n_tasks=3)
    space = CoreSpace.from_adapters(adapters)
    task_vectors = task_vectors_from(adapters, sequential=False)
    merged = apply_merge(get_merge_spec(merge_fn), task_vectors, core_space=space)

    for key, delta in merged.vector.items():
        assert delta.shape == task_vectors[0].vector[key].shape
        U_ref, V_ref = space.bases[key]
        torch.testing.assert_close(U_ref @ U_ref.T @ delta @ V_ref @ V_ref.T, delta)


def test_finetune_baseline_is_not_rotated():
    from src.merging.registry import apply_merge, get_merge_spec

    adapters = make_adapters(n_tasks=2)
    task_vectors = task_vectors_from(adapters, sequential=True)
    merged = apply_merge(
        get_merge_spec("finetune"), task_vectors, core_space=CoreSpace.from_adapters(adapters)
    )
    assert merged is task_vectors[-1]


def test_masked_magmax_budgets_count_core_matrix_entries():
    """mask_and_merge_by_weights runs unchanged on the core matrices, so each
    weight contributes R^2 = (T * r)^2 entries and task i wins int(R^2 * w_i)
    of them (its own assert checks the per-task counts)."""
    from src.merging.task_vectors import mask_and_merge_by_weights

    n_tasks = 4
    adapters = make_adapters(n_tasks=n_tasks)
    space = CoreSpace.from_adapters(adapters)
    task_vectors = task_vectors_from(adapters, sequential=True)

    merged_core, num_unaligned, num_params_all = mask_and_merge_by_weights(
        space.to_core(task_vectors), [0.1, 0.2, 0.3, 0.4], seed=0
    )

    R = n_tasks * R_LORA
    assert num_params_all == len(SHAPES) * R * R
    assert set(num_unaligned) == {f"task_{i + 1}" for i in range(n_tasks)}
    merged = space.from_core(merged_core)
    for key, delta in merged.vector.items():
        assert delta.shape == task_vectors[0].vector[key].shape
