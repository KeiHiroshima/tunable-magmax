"""Fixtures for the characterization ("golden") tests.

These tests pin the *current* behaviour of the pipeline so the DRY refactor
(unifying the duplicated vision/NLP code paths) can be shown not to change it.
They are the substitute for "keep the implementations separate so nothing
breaks": backward compatibility is asserted here instead of being bought with
duplicated code.

Everything runs on CPU against a few-hundred-parameter synthetic module — no
GPU, no network, no real checkpoints — so the whole file stays fast enough to
run on every edit.

Where a test pins behaviour that is known to be *wrong* and is scheduled to
change in a later phase, its docstring says so explicitly.
"""

import pytest
import torch
from magmax_tiny_model import TinyModel

from src.merging.task_vector import TaskVector

# Shapes chosen to exercise every branch of mask_and_merge_by_weights:
# a 2-D key, a 1-D key, a second 2-D key, and an all-zero key that trips the
# is_freezed_parameter() early-out.
SYNTHETIC_KEYS = {
    "layer1.weight": (4, 5),
    "layer1.bias": (4,),
    "layer2.weight": (3, 4),
    "frozen.weight": (2, 2),
}

N_SYNTHETIC_TASKS = 3


def build_synthetic_task_vectors(n_tasks: int = N_SYNTHETIC_TASKS) -> list:
    """Deterministic task vectors, independent of global RNG state.

    Uses an explicit torch.Generator so that a merge function which itself
    seeds/consumes the global RNG (merge_rnd_mix, mask_and_merge_by_weights)
    cannot change what the *inputs* are.
    """
    generator = torch.Generator().manual_seed(0)
    task_vectors = []
    for _ in range(n_tasks):
        vector = {}
        for key, shape in SYNTHETIC_KEYS.items():
            vector[key] = (
                torch.zeros(shape)
                if key.startswith("frozen")
                else torch.randn(shape, generator=generator)
            )
        task_vectors.append(TaskVector(vector=vector))
    return task_vectors


@pytest.fixture
def synthetic_task_vectors():
    return build_synthetic_task_vectors()


@pytest.fixture
def tiny_checkpoints(tmp_path):
    """(pretrained_path, [finetuned_path, ...]) for real TaskVector round-trips.

    Needed by the tests that exercise apply_to()/TaskVector(ckpt, ckpt), which
    go through torch.load and therefore need actual files on disk.
    """
    from src.utils import torch_save

    generator = torch.Generator().manual_seed(1234)

    base = TinyModel()
    with torch.no_grad():
        for p in base.parameters():
            p.copy_(torch.randn(p.shape, generator=generator))
    pretrained_path = tmp_path / "zeroshot.pt"
    torch_save(base, str(pretrained_path))

    finetuned_paths = []
    for i in range(N_SYNTHETIC_TASKS):
        from src.utils import torch_load

        model = torch_load(str(pretrained_path))
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * (i + 1) * torch.randn(p.shape, generator=generator))
        path = tmp_path / f"finetuned_{i}.pt"
        torch_save(model, str(path))
        finetuned_paths.append(str(path))

    return str(pretrained_path), finetuned_paths
