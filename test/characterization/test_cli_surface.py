"""Pins the command-line interface that scripts/*.sh depend on.

The shell scripts are the project's real entry points, so every flag they pass
has to keep parsing. These tests reproduce the exact argv the scripts build
under their default variable values, and separately check that no script has
grown a flag the tests don't know about.
"""

import re
from pathlib import Path

import pytest

from src.args import parse_arguments

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

# scripts/finetune.sh with its committed defaults.
FINETUNE_ARGV = [
    "finetune_splitted.py",
    "--model", "ViT-B-16",
    "--dataset", "CIFAR100",
    "--epochs", "10",
    "--n_splits", "5",
    "--split_strategy", "class",
    "--sequential-finetuning",
    "--seed", "3",
    "--results_db", "logs/ViT-B-16/run",
    "--taskseq_pattern", "A",
]

# scripts/merge.sh, masked_magmax_with_targetdata branch (n_splits=5).
MERGE_ARGV = [
    "merge_for_targetdata.py",
    "--model", "ViT-B-16",
    "--dataset", "CIFAR100",
    "--epochs", "10",
    "--n_splits", "5",
    "--split_strategy", "class",
    "--sequential-finetuning",
    "--results_db", "logs/ViT-B-16/run",
    "--taskseq_pattern", "A",
    "--merge_fn", "masked_magmax_with_targetdata",
    "--similarity_metric", "labels",
    "--num_train_data_each_task", "500",
    "--num_target_data", "1000",
    "--seed", "3",
    "--gpu_id", "0",
]


def _parse(argv):
    # argv[0] is the script name, mirroring how the shell scripts invoke these
    # entry points; parse_arguments takes the flags only.
    return parse_arguments(argv[1:])


def test_finetune_script_argv_parses():
    args = _parse(FINETUNE_ARGV)

    assert args.model == "ViT-B-16"
    assert args.dataset == "CIFAR100"
    assert args.n_splits == 5
    assert args.split_strategy == "class"
    assert args.sequential_finetuning is True
    assert args.taskseq_pattern == "A"
    assert args.seed == 3


def test_merge_script_argv_parses():
    args = _parse(MERGE_ARGV)

    assert args.merge_fn == "masked_magmax_with_targetdata"
    assert args.similarity_metric == "labels"
    assert args.num_train_data_each_task == 500
    assert args.num_target_data == 1000
    assert args.gpu_id == 0


def test_device_is_derived_from_gpu_id():
    """args.device is computed inside parse_arguments rather than passed in;
    every backend relies on it existing."""
    args = _parse(MERGE_ARGV)
    assert args.device in ("cpu",) or args.device.startswith("cuda:")


@pytest.mark.parametrize("dataset", ["LSB", "CITB19", "CITB38"])
def test_nlp_datasets_parse_and_resolve_to_a_backend(dataset):
    from src.backends.registry import resolve_backend

    args = _parse(
        ["merge_for_targetdata.py", "--model", "bert-base-uncased", "--dataset", dataset]
    )
    backend = resolve_backend(args.dataset)

    assert hasattr(backend, "finetune")
    assert hasattr(backend, "merge_and_evaluate")


def test_unknown_dataset_is_rejected_with_the_registered_names():
    from src.backends.registry import resolve_backend

    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_backend("NoSuchDataset")


# Flags removed because nothing in the repository implemented them: there was
# no LwF/EWC training code (only a path branch in the merge half), and --save
# was overwritten unconditionally by both entry points before it was read.
REMOVED_FLAGS = [
    "--lwf_lamb",
    "--ewc_lamb",
    "--lamb_case",
    "--save",
    # replaced by --warmup_ratio: a step count cannot suit task schedules that
    # differ by orders of magnitude.
    "--warmup_length",
]


@pytest.mark.parametrize("flag", REMOVED_FLAGS)
def test_removed_flags_are_rejected(flag):
    with pytest.raises(SystemExit):
        _parse(["x", "--model", "ViT-B-16", "--dataset", "CIFAR100", flag, "x"])


# The similarity metrics the paper actually evaluates: Cos / MMD / OT (each in
# a raw and an "_embedded" form) plus Label. "hpo" was removed — it appeared in
# a single sentence of the paper as an unevaluated alternative and in no table.
PAPER_SIMILARITY_METRICS = {
    "labels",
    "cosine",
    "mmd",
    "ot",
    "cosine_embedded",
    "mmd_embedded",
    "ot_embedded",
}


@pytest.mark.parametrize("metric", sorted(PAPER_SIMILARITY_METRICS))
def test_every_evaluated_similarity_metric_still_parses(metric):
    args = _parse(
        ["merge_for_targetdata.py", "--model", "ViT-B-16", "--dataset", "CIFAR100",
         "--similarity_metric", metric]
    )
    assert args.similarity_metric == metric


def test_removed_hpo_similarity_metric_is_rejected():
    """run_optmization (and its optuna dependency) were deleted as unused by
    the paper's experiments. --similarity_metric hpo must now fail loudly at
    argument-parsing time rather than reaching a missing function."""
    with pytest.raises(SystemExit):
        _parse(
            ["merge_for_targetdata.py", "--model", "ViT-B-16", "--dataset", "CIFAR100",
             "--similarity_metric", "hpo"]
        )


# src/backends/vision_backend.py::finetune used to overwrite these two before
# reading them, so whatever the user passed was discarded. --batch_size was the
# one that mattered: it became 32, while the CLI default and the paper's
# reported setting are both 128.
PAPER_TRAINING_DEFAULTS = {"batch_size": 128, "lr": 1e-5, "epochs": 10}


@pytest.mark.parametrize("bad", ["0", "1.0", "1.5", "-0.1"])
def test_warmup_ratio_outside_the_open_unit_interval_is_rejected(bad):
    """At 1.0 the whole schedule is warmup and the cosine decay never runs —
    the failure the old step-count flag produced silently."""
    with pytest.raises(SystemExit):
        _parse(["x", "--model", "ViT-B-16", "--dataset", "CIFAR100",
                "--warmup_ratio", bad])


@pytest.mark.parametrize("good", ["0.01", "0.1", "0.5", "0.99"])
def test_warmup_ratio_accepts_usable_fractions(good):
    args = _parse(["x", "--model", "ViT-B-16", "--dataset", "CIFAR100",
                   "--warmup_ratio", good])

    assert args.warmup_ratio == float(good)


@pytest.mark.parametrize("flag,expected", sorted(PAPER_TRAINING_DEFAULTS.items()))
def test_training_defaults_match_the_paper(flag, expected):
    args = _parse(["x", "--model", "ViT-B-16", "--dataset", "CIFAR100"])

    assert getattr(args, flag) == expected


@pytest.mark.parametrize("flag,value", [("--batch_size", "64"), ("--lr", "3e-5")])
def test_vision_finetune_no_longer_discards_these_flags(flag, value, monkeypatch):
    """Passing a value must reach the training run rather than be overwritten."""
    from src.backends import vision_backend

    args = _parse(["x", "--model", "ViT-B-16", "--dataset", "CIFAR100",
                   "--split_strategy", "class", flag, value])
    before = {"batch_size": args.batch_size, "lr": args.lr}

    monkeypatch.setattr(vision_backend.wandb, "init", lambda *a, **k: None)
    monkeypatch.setattr(vision_backend, "_run_sequential_finetuning", lambda a: None)
    vision_backend.finetune(args)

    assert args.batch_size == before["batch_size"]
    assert args.lr == before["lr"]


def test_no_script_uses_a_flag_the_tests_do_not_cover():
    """Closes the loop: if a shell script grows a new flag, this fails until
    the argv fixtures above are updated to match."""
    covered = set(FINETUNE_ARGV) | set(MERGE_ARGV)

    for script in sorted(SCRIPTS_DIR.glob("*.sh")):
        flags = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9_-]*", script.read_text()))
        assert flags <= covered, f"{script.name} uses uncovered flags: {sorted(flags - covered)}"
