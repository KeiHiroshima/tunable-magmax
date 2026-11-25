"""Check that a vision run's checkpoints are all there, without training.

A job that only merges must merge the checkpoints that already exist — the
ones every other method in the same results directory was merged from
(logs/README.md, rule 3). scripts/vision/finetune.sh would instead train any
split that is missing, quietly mixing in a new checkpoint, so
scripts/vision/run_parallel_experiment.sh calls this in its place for the
variants in merge_only_variants.

    uv run python -m src.check_checkpoints <variant> <dataset> <n_splits> <seed> <epochs>

variant is full, lora (one adapter per split, merged in) or lora-reuse. The
paths come from src/backends/vision_backend.py::split_checkpoint_path, the
function fine-tuning writes through and merging reads through. Exits 1 and
lists what is missing if anything is.

Positional arguments only: test/characterization/test_cli_surface.py checks
every long option a shell script uses against the training/merging CLI.
"""

import os
import sys

from src.config import get_zeroshot_checkpoint

VARIANTS = {
    # variant: (--finetune_mode, --lora_chain), as run_parallel_experiment.sh
    # passes them to finetune.sh and merge.sh.
    "full": ("full", "merge"),
    "lora": ("lora", "merge"),
    "lora-reuse": ("lora", "reuse"),
}


def run_args(variant, dataset, n_splits, seed, epochs, model="ViT-B-16", task_seq="A"):
    """The arguments scripts/vision/finetune.sh parses for this run."""
    from src.args import parse_arguments

    finetune_mode, lora_chain = VARIANTS[variant]
    return parse_arguments([
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--n_splits", str(n_splits),
        "--split_strategy", "class",
        "--sequential-finetuning",
        "--finetune_mode", finetune_mode,
        "--lora_chain", lora_chain,
        "--seed", str(seed),
        "--taskseq_pattern", task_seq,
    ])


def missing_checkpoints(args):
    """Every file merging this run would read that is not there: the
    zero-shot checkpoint, then each split's checkpoint in order."""
    from src.backends.vision_backend import split_checkpoint_path

    paths = [get_zeroshot_checkpoint(args.model)]
    paths += [split_checkpoint_path(args, i) for i in range(args.n_splits)]
    return [p for p in paths if not os.path.exists(p)]


def main(argv):
    if len(argv) not in (5, 6, 7) or argv[0] not in VARIANTS:
        print(__doc__, file=sys.stderr)
        return 2
    variant, dataset, n_splits, seed, epochs, *rest = argv
    args = run_args(variant, dataset, int(n_splits), int(seed), int(epochs), *rest)
    missing = missing_checkpoints(args)
    label = f"{variant} {dataset}-{n_splits} seed {seed}"
    if missing:
        print(f"{label}: {len(missing)} checkpoint(s) missing (merge only: not training them):")
        for p in missing:
            print(f"  {p}")
        return 1
    print(f"{label}: all {args.n_splits} split checkpoints present")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
