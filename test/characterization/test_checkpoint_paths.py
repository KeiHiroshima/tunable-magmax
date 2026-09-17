"""Pins the on-disk checkpoint path convention.

This is the highest-stakes guard in the suite. The
`ft-pattern_{}-epochs-{}-seed:{}` template is currently hand-written in four
places (vision finetune, vision merge, and each NLP backend's `_ckpt_dir`);
Phase 3 collapses them into one helper. If that helper spells a path even
slightly differently, every checkpoint already on disk silently becomes
unreachable and the pipeline quietly re-trains from scratch.

The assertions are written relative to src.config.BASE_DIR rather than to an
absolute string, so they hold whatever MAGMAX_BASE_DIR is set to: what is
pinned is the structure underneath it.
"""

import os
from argparse import Namespace

import pytest

from src.config import BASE_DIR

MODEL = "ViT-B-16"
DATASET = "CIFAR100"
EPOCHS = 10
N_SPLITS = 5
PATTERN = "A"
SEED = 3

# The relative layout every backend must agree on.
EXPECTED_RUN_DIR = f"ft-pattern_{PATTERN}-epochs-{EPOCHS}-seed:{SEED}"


def _vision_args(**overrides):
    """Mirrors the flags scripts/finetune.sh and scripts/merge.sh actually pass."""
    args = Namespace(
        model=MODEL,
        dataset=DATASET,
        epochs=EPOCHS,
        n_splits=N_SPLITS,
        taskseq_pattern=PATTERN,
        seed=SEED,
        split_strategy="class",
        sequential_finetuning=True,
        merge_fn="magmax",
        wandb_entity_name="test-entity",
        device="cpu",
        load=None,
        coeff=0.5,
        similarity_metric="labels",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _nlp_args(dataset, **overrides):
    args = Namespace(
        model="bert-base-uncased",
        dataset=dataset,
        epochs=EPOCHS,
        taskseq_pattern=PATTERN,
        seed=SEED,
        sequential_finetuning=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _capture_vision_finetune_paths(monkeypatch, args):
    """Run the real vision finetune() but stub out everything that does work,
    and make every checkpoint look like it already exists so the loop skips
    each split after computing its path."""
    from src.backends import vision_backend

    monkeypatch.setattr(vision_backend.wandb, "init", lambda *a, **k: None)

    queried = []

    def fake_exists(path):
        queried.append(path)
        return True

    monkeypatch.setattr(os.path, "exists", fake_exists)
    vision_backend.finetune(args)
    return queried


def _capture_vision_merge_paths(monkeypatch, args):
    """Run the real vision merge_and_evaluate() but record the checkpoint
    paths it hands to TaskVector instead of loading them."""
    from src.backends import vision_backend

    monkeypatch.setattr(vision_backend.wandb, "init", lambda *a, **k: None)
    monkeypatch.setattr(
        vision_backend, "evaluate_merged_fts_on_target_data", lambda *a, **k: None
    )

    requested = []

    class RecordingTaskVector:
        def __init__(self, pretrained_checkpoint, finetuned_checkpoint):
            requested.append(finetuned_checkpoint)

    monkeypatch.setattr(vision_backend, "TaskVector", RecordingTaskVector)
    vision_backend.merge_and_evaluate(args)
    return requested


def test_vision_finetune_checkpoint_paths(monkeypatch):
    args = _vision_args()
    queried = _capture_vision_finetune_paths(monkeypatch, args)

    expected_dir = os.path.join(
        BASE_DIR,
        "checkpoints",
        MODEL,
        "sequential_finetuning",
        "class_incremental",
        f"{DATASET}-{N_SPLITS}",
        EXPECTED_RUN_DIR,
    )
    assert queried == [
        os.path.join(expected_dir, f"finetuned_{i}.pt") for i in range(N_SPLITS)
    ]


def test_vision_finetune_without_sequential_flag_drops_the_subdirectory(monkeypatch):
    args = _vision_args(sequential_finetuning=False)
    queried = _capture_vision_finetune_paths(monkeypatch, args)

    assert "sequential_finetuning" not in queried[0]
    assert os.path.join("checkpoints", MODEL, "class_incremental") in queried[0]


def test_vision_merge_reads_exactly_what_vision_finetune_wrote(monkeypatch):
    """The two halves of the pipeline build this path independently today.
    They must resolve to the same strings — otherwise merging silently fails
    to find the fine-tuned checkpoints."""
    written = _capture_vision_finetune_paths(monkeypatch, _vision_args())
    monkeypatch.undo()
    read = _capture_vision_merge_paths(monkeypatch, _vision_args())

    assert read == written


@pytest.mark.parametrize(
    "dataset,family_dir",
    [
        ("LSB", "nlp_classification"),
        ("CITB19", "nlp_seq2seq"),
        ("CITB38", "nlp_seq2seq"),
    ],
)
def test_nlp_checkpoint_dirs(dataset, family_dir):
    from src.backends.registry import resolve_backend

    backend = resolve_backend(dataset)
    args = _nlp_args(dataset)

    assert backend._ckpt_dir(args) == os.path.join(
        BASE_DIR,
        "checkpoints",
        args.model,
        "sequential_finetuning",
        family_dir,
        dataset,
        EXPECTED_RUN_DIR,
    )


def test_nlp_checkpoint_dir_without_sequential_flag():
    from src.backends import nlp_classification_backend

    args = _nlp_args("LSB", sequential_finetuning=False)
    assert "sequential_finetuning" not in nlp_classification_backend._ckpt_dir(args)


def test_zeroshot_checkpoint_path():
    from src.config import get_zeroshot_checkpoint

    assert get_zeroshot_checkpoint(MODEL) == os.path.join(
        BASE_DIR, "checkpoints", MODEL, "zeroshot.pt"
    )
