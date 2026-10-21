"""Core Space merging (Panariello et al., NeurIPS 2025) for --merge_space core.

Every merge method in this repo picks or combines task-vector elements one by
one, so what an "element" is depends on the coordinate system the task vectors
are written in. Under --merge_space full (the default) that is the adapted
weight matrix itself. Under --merge_space core each LoRA task vector is first
rotated into a basis shared by all tasks, the unchanged merge function runs on
the rotated matrices, and the result is rotated back:

    tau_i  (d_out x d_in)  --to_core-->  M_i = U_ref^T tau_i V_ref  (R x R)
    M_merged = merge(M_1, ..., M_T)
    Delta W = U_ref M_merged V_ref^T                                (d_out x d_in)

with R = T * r (each side capped at the weight's own dimension when R exceeds
it, so M_i is then min(d_out, R) x min(d_in, R)). The bases are, per adapted weight, the left singular vectors
of the horizontally stacked B's and the right singular vectors of the
vertically stacked A's (the paper's eq. 3):

    [B_1, ..., B_T] = U_ref S V^T          [A_1; ...; A_T] = U S V_ref^T

They must come from the adapters themselves (CoreSpace.from_adapters), not
from an SVD of the dense task vectors: that would be a different basis.

The paper writes the core matrix of a single adapter as
(U_ref^T B_i)(A_i V_ref) (eq. 8). Here it is computed from the dense task
vector instead, U_ref^T tau_i V_ref, which is the same matrix when
tau_i = B_i A_i, and also covers --sequential-finetuning, where task i's
vector is the running sum of adapters 0..i: every adapter's columns/rows lie
in the span of U_ref/V_ref, so so does the sum, and the round trip
from_core(to_core(tau_i)) loses nothing.

Consequences for the merge functions:
  * a linear merge (average) gives exactly the full-space result (eq. 11);
  * element counts are counts of core-matrix entries (R^2 per weight), so
    mask_and_merge_by_weights' per-task budgets, num_params_all and
    num_unaligned are all in those units under --merge_space core.
"""

import torch

from src.merging.task_vector import TaskVector


def result_name(merge_fn_name: str, merge_space: str) -> str:
    """The name a merge's results are written under: the --merge_fn value,
    suffixed "-core" under --merge_space core so the two never share files."""
    return merge_fn_name if merge_space == "full" else f"{merge_fn_name}-{merge_space}"


def _scaled_factors(adapter: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{"<module>.weight": (A, (alpha / r) * B)} — keyed like the task vectors
    built by src.lora.adapter_deltas, with the LoRA scaling folded into B so
    that B @ A is exactly that adapter's delta."""
    scaling = adapter["config"]["alpha"] / adapter["config"]["r"]
    return {
        f"{name}.weight": (ab["A"], ab["B"] * scaling)
        for name, ab in adapter["modules"].items()
    }


class CoreSpace:
    """Per-weight reference bases U_ref (d_out x R) and V_ref (d_in x R),
    stored in float64 on the CPU."""

    def __init__(self, bases: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        self.bases = bases

    @classmethod
    def from_adapters(cls, adapters: list[dict]) -> "CoreSpace":
        """Reference bases from every task's adapter, in task order (the order
        only fixes the bases' column signs/order, not the space they span)."""
        factors = [_scaled_factors(adapter) for adapter in adapters]
        keys = factors[0].keys()
        for f in factors[1:]:
            assert f.keys() == keys, "every adapter must cover the same weights"

        bases = {}
        for key in keys:
            A_stack = torch.cat([f[key][0] for f in factors], dim=0).double()
            B_stack = torch.cat([f[key][1] for f in factors], dim=1).double()
            # full_matrices=False: R = T * r columns, or min(R, d) when the
            # stacked rank would exceed the weight's own dimensions.
            V_ref = torch.linalg.svd(A_stack, full_matrices=False)[2].T
            U_ref = torch.linalg.svd(B_stack, full_matrices=False)[0]
            bases[key] = (U_ref, V_ref)
        return cls(bases)

    def _basis(self, key):
        if key not in self.bases:
            raise KeyError(
                f"{key!r} has no Core Space basis: --merge_space core only "
                "rotates LoRA-adapted weights"
            )
        return self.bases[key]

    def to_core(self, task_vectors: list[TaskVector]) -> list[TaskVector]:
        """tau_i -> U_ref^T tau_i V_ref, in each task vector's own dtype."""
        rotated = []
        for tv in task_vectors:
            vector = {}
            for key, tau in tv.vector.items():
                U_ref, V_ref = self._basis(key)
                core = U_ref.T @ tau.double().cpu() @ V_ref
                vector[key] = core.to(dtype=tau.dtype)
            rotated.append(TaskVector(vector=vector))
        return rotated

    def from_core(self, task_vector: TaskVector) -> TaskVector:
        """M -> U_ref M V_ref^T, in the core vector's dtype."""
        vector = {}
        for key, core in task_vector.vector.items():
            U_ref, V_ref = self._basis(key)
            delta = U_ref @ core.double().cpu() @ V_ref.T
            vector[key] = delta.to(dtype=core.dtype)
        return TaskVector(vector=vector)
