"""src/paths.py — the one place that spells the checkpoint layout.

test_checkpoint_paths.py verifies the layout through the backends (what
production code actually produces). These tests cover the helper itself: the
pieces it joins, and the edge cases each backend relies on.
"""

import os
from argparse import Namespace

import pytest

from src.config import BASE_DIR
from src.paths import checkpoint_dir, finetuned_path, run_dir_name


def _args(**overrides):
    args = Namespace(
        model="ViT-B-16",
        dataset="CIFAR100",
        epochs=10,
        taskseq_pattern="A",
        seed=3,
        sequential_finetuning=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_run_dir_name_carries_task_order_epochs_and_seed():
    assert run_dir_name(_args()) == "ft-pattern_A-epochs-10-seed:3"


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("taskseq_pattern", "B", "ft-pattern_B-epochs-10-seed:3"),
        ("epochs", 3, "ft-pattern_A-epochs-3-seed:3"),
        ("seed", 5, "ft-pattern_A-epochs-10-seed:5"),
    ],
)
def test_every_run_dimension_changes_the_directory(field, value, expected):
    """Two runs that differ in any of these must not share a directory, or the
    second silently resumes from the first's checkpoints."""
    assert run_dir_name(_args(**{field: value})) == expected


def test_checkpoint_dir_full_layout():
    assert checkpoint_dir(_args(), group="class_incremental", scope="CIFAR100-5") == (
        os.path.join(
            BASE_DIR,
            "checkpoints",
            "ViT-B-16",
            "sequential_finetuning",
            "class_incremental",
            "CIFAR100-5",
            "ft-pattern_A-epochs-10-seed:3",
        )
    )


def test_independent_finetuning_drops_the_sequential_segment():
    """The empty segment must collapse rather than leaving a doubled separator
    or an empty directory component."""
    path = checkpoint_dir(
        _args(sequential_finetuning=False), group="class_incremental", scope="CIFAR100-5"
    )

    assert "sequential_finetuning" not in path
    assert "//" not in path
    assert path == os.path.join(
        BASE_DIR,
        "checkpoints",
        "ViT-B-16",
        "class_incremental",
        "CIFAR100-5",
        "ft-pattern_A-epochs-10-seed:3",
    )


def test_sequential_and_independent_runs_do_not_collide():
    args = _args()
    sequential = checkpoint_dir(args, group="g", scope="s")
    independent = checkpoint_dir(_args(sequential_finetuning=False), group="g", scope="s")

    assert sequential != independent


def test_group_separates_backends_sharing_a_model():
    """The NLP backends share bert-base-uncased, so only `group` keeps a
    classification run from overwriting a seq2seq one."""
    args = _args(model="bert-base-uncased", dataset="LSB")
    cls = checkpoint_dir(args, group="nlp_classification", scope=args.dataset)
    seq = checkpoint_dir(args, group="nlp_seq2seq", scope=args.dataset)

    assert cls != seq


def test_finetuned_path_numbering():
    ckpt_dir = checkpoint_dir(_args(), group="g", scope="s")

    assert finetuned_path(ckpt_dir, 0) == os.path.join(ckpt_dir, "finetuned_0.pt")
    assert finetuned_path(ckpt_dir, 12) == os.path.join(ckpt_dir, "finetuned_12.pt")


def test_every_backend_routes_through_the_shared_helper():
    """The point of Phase 3: no backend may go back to hand-rolling the layout.
    Each one exposes a single `_ckpt_dir`, and that is the only place its
    group/scope is spelled out."""
    from src.backends import (
        nlp_classification_backend,
        nlp_seq2seq_backend,
        vision_backend,
    )

    for backend in (vision_backend, nlp_classification_backend, nlp_seq2seq_backend):
        assert hasattr(backend, "_ckpt_dir"), backend.__name__

    source_files = [
        "src/backends/vision_backend.py",
        "src/backends/nlp_classification_backend.py",
        "src/backends/nlp_seq2seq_backend.py",
    ]
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    for rel in source_files:
        text = (repo_root / rel).read_text()
        assert "ft-pattern_" not in text, (
            f"{rel} spells the run-directory template itself; it must call "
            f"src.paths.checkpoint_dir instead"
        )
        assert "finetuned_{" not in text, (
            f"{rel} builds a checkpoint filename itself; use "
            f"src.paths.finetuned_path instead"
        )
