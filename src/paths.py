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

    vision        group="{split_strategy}_incremental"  scope="{dataset}-{n_splits}{suffix}"
    NLP (cls)     group="nlp_classification"            scope="{dataset}{suffix}"
    NLP (seq2seq) group="nlp_seq2seq"                   scope="{dataset}"

where {suffix} is finetune_mode_suffix(args).
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


def finetune_mode_suffix(args) -> str:
    """The scope suffix that keeps runs of different kinds apart: "" for a
    full run, "-lora" for --finetune_mode lora, "-lora-reuse" for
    --lora_chain reuse. They write files under the same names
    (finetuned_{i}.pt / adapter_{i}.pt), so sharing a directory would make
    one kind skip training on — or merge — the other's files.

    Callers that build `args` by hand without the fields get a full run, or
    the default merge chain."""
    if getattr(args, "finetune_mode", "full") == "full":
        return ""
    if getattr(args, "lora_chain", "merge") == "reuse":
        return f"-{args.finetune_mode}-reuse"
    return f"-{args.finetune_mode}"


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


def adapter_path(ckpt_dir: str, idx: int) -> str:
    """The LoRA adapter written after fine-tuning on task `idx`.

    Used instead of finetuned_path under --finetune_mode lora: only the
    adapter is stored, and the model after task `idx` is rebuilt as the
    zero-shot checkpoint plus adapters 0..idx (src/lora.py).
    """
    return os.path.join(ckpt_dir, f"adapter_{idx}.pt")
