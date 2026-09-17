"""Where checkpoints live.

The layout below used to be hand-written in four places — the vision backend's
fine-tuning and merging halves, and each NLP backend's `_ckpt_dir` — plus a
fifth hand-rolled copy of the zero-shot path. Keeping four spellings in sync by
hand is exactly the kind of thing that silently orphans a directory full of
trained checkpoints, so they all route through here now.

The layout is:

    BASE_DIR/checkpoints/{model}/[sequential_finetuning/]{group}/{scope}/
        ft-pattern_{pattern}-epochs-{epochs}-seed:{seed}/finetuned_{i}.pt

`group` and `scope` are what differ between backends, so each backend supplies
its own (see the `_ckpt_dir` helper each one defines):

    vision        group="{split_strategy}_incremental"  scope="{dataset}-{n_splits}"
    NLP (cls)     group="nlp_classification"            scope="{dataset}"
    NLP (seq2seq) group="nlp_seq2seq"                   scope="{dataset}"
"""

import os

from src.config import BASE_DIR


def run_dir_name(args) -> str:
    """The per-run directory name.

    Everything that distinguishes one training run from another of the same
    model on the same benchmark: task order, epoch budget, seed.
    """
    return (
        f"ft-pattern_{args.taskseq_pattern}"
        f"-epochs-{args.epochs}"
        f"-seed:{args.seed}"
    )


def checkpoint_dir(args, *, group: str, scope: str) -> str:
    """The directory holding one run's per-task checkpoints."""
    # An empty segment collapses away: os.path.join("a", "", "b") == "a/b".
    sequential = "sequential_finetuning" if args.sequential_finetuning else ""
    return os.path.join(
        BASE_DIR,
        "checkpoints",
        args.model,
        sequential,
        group,
        scope,
        run_dir_name(args),
    )


def finetuned_path(ckpt_dir: str, idx: int) -> str:
    """The checkpoint written after fine-tuning on task/split `idx`."""
    return os.path.join(ckpt_dir, f"finetuned_{idx}.pt")
