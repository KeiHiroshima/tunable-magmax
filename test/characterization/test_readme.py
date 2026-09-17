"""README.md against the code it documents.

Usage documentation drifts silently: a flag is renamed, a merge method removed,
a default changed, and the README keeps describing the old behaviour until
someone follows it and it does not work. The claims that can be checked
mechanically are checked here.

Only claims with a single source of truth in the code are covered — prose and
rationale are not, and should not be.
"""

import json
import re
from pathlib import Path

import pytest

from src.args import parse_arguments
from src.merging.registry import MERGE_SPECS, SUPPORTED_MERGE_FN_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
README = (REPO_ROOT / "README.md").read_text()

# Rows of the "--merge_fn" table: | `name` | description | NLP support |
MERGE_TABLE_ROW = re.compile(r"^\| `([a-z_]+)` \| (.+?) \| (yes|`LSB` only|not yet) \|$", re.M)


def _merge_table():
    return {name: support for name, _, support in MERGE_TABLE_ROW.findall(README)}


# --- the merge_fn table -----------------------------------------------------


def test_merge_table_lists_every_registered_method():
    assert set(_merge_table()) == set(MERGE_SPECS)


def test_merge_table_reports_nlp_support_correctly():
    """`masked_magmax_with_targetdata` is the subtle one: the registry marks it
    nlp_supported=False because it cannot go through the plain
    task_vectors -> TaskVector path, but the LSB backend implements it with a
    flow of its own. Documenting it as "vision-only" would be wrong."""
    table = _merge_table()
    plain_nlp = {name for name, support in table.items() if support == "yes"}

    assert plain_nlp == SUPPORTED_MERGE_FN_NAMES

    from src.backends import nlp_classification_backend, nlp_seq2seq_backend
    import inspect

    lsb = inspect.getsource(nlp_classification_backend)
    seq2seq = inspect.getsource(nlp_seq2seq_backend)
    assert table["masked_magmax_with_targetdata"] == "`LSB` only"
    assert "_merge_and_evaluate_masked" in lsb
    assert "_merge_and_evaluate_masked" not in seq2seq


# --- defaults ---------------------------------------------------------------


DOCUMENTED_DEFAULTS = {
    "batch_size": 128,
    "warmup_ratio": 0.1,
    "epochs": 10,
    "num_target_data": 1000,
    "n_splits": 2,
    "seed": 5,
    "gpu_id": 0,
    "lr": 1e-5,
    "target_config": "target_data_config",
}


def _readme_mentions(value) -> bool:
    """True if the README states this value somewhere, allowing for the
    spellings a human would write (`1e-5` for Python's 1e-05)."""
    if isinstance(value, float):
        return any(
            abs(float(token) - value) < 1e-12
            for token in re.findall(r"[0-9]*\.?[0-9]+(?:e-?[0-9]+)?", README)
            if _parses_as_float(token)
        )
    return str(value) in README


def _parses_as_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("flag,expected", sorted(DOCUMENTED_DEFAULTS.items()))
def test_documented_defaults_match_the_parser(flag, expected):
    args = parse_arguments(["--model", "ViT-B-16", "--dataset", "CIFAR100"])

    assert getattr(args, flag) == expected
    assert _readme_mentions(expected), f"the README does not state {flag}'s default"


# --- removed options --------------------------------------------------------


REMOVED = ["--warmup_length", "--save", "--lwf_lamb", "--ewc_lamb", "--lamb_case"]


@pytest.mark.parametrize("flag", REMOVED)
def test_removed_options_are_documented_and_actually_gone(flag):
    assert f"`{flag}`" in README, f"{flag} is gone from the code but not the README"

    with pytest.raises(SystemExit):
        parse_arguments(["--model", "m", "--dataset", "d", flag, "1"])


@pytest.mark.parametrize("name", ["select_one_task_vector", "hpo"])
def test_removed_methods_are_documented_and_actually_gone(name):
    assert name in README
    assert name not in MERGE_SPECS

    metrics = parse_arguments(
        ["--model", "m", "--dataset", "d"]
    )  # parses, so the choices below are the live ones
    assert metrics is not None
    with pytest.raises(SystemExit):
        parse_arguments(["--model", "m", "--dataset", "d", "--similarity_metric", "hpo"])


# --- similarity metrics -----------------------------------------------------


def test_every_similarity_metric_is_documented():
    import argparse
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_arguments(["--help"])
    choices = set(
        re.search(r"--similarity_metric \{([^}]+)\}", buf.getvalue()).group(1).split(",")
    )

    for metric in choices:
        assert f"`{metric}`" in README, f"--similarity_metric {metric} is undocumented"


# --- config files -----------------------------------------------------------


def test_every_documented_config_exists():
    documented = set(re.findall(r"`(target_data_config[a-z0-9_]*)`", README))
    # the {5,20,50} row is written with a brace expansion
    documented.discard("target_data_config_split")
    if "target_data_config_split{5,20,50}" in README:
        documented |= {f"target_data_config_split{n}" for n in (5, 20, 50)}

    for name in documented:
        assert (REPO_ROOT / "configs" / f"{name}.json").exists(), name


def test_every_shipped_config_is_documented():
    shipped = {p.stem for p in (REPO_ROOT / "configs").glob("target_data_config*.json")}
    for name in shipped:
        mentioned = f"`{name}`" in README or "target_data_config_split{5,20,50}" in README
        assert mentioned, f"configs/{name}.json is not mentioned in the README"


def test_documented_environment_count():
    n = sum(
        len(c["variants"])
        for c in json.loads(
            (REPO_ROOT / "configs" / "target_data_config.json").read_text()
        )["dataset_configs"]
    )

    assert f"{n} environments" in README


# --- the entry points and backends the README names -------------------------


@pytest.mark.parametrize(
    "dataset,module",
    [
        ("CIFAR100", "vision_backend"),
        ("ImageNetR", "vision_backend"),
        ("LSB", "nlp_classification_backend"),
        ("CITB19", "nlp_seq2seq_backend"),
        ("CITB38", "nlp_seq2seq_backend"),
    ],
)
def test_backend_table_matches_the_registry(dataset, module):
    from src.backends.registry import resolve_backend

    assert resolve_backend(dataset).__name__.endswith(module)
    assert f"`{dataset}`" in README
    assert f"{module}.py" in README


def test_entry_points_exist():
    for name in ("finetune_splitted.py", "merge_for_targetdata.py"):
        assert (REPO_ROOT / name).exists()
        assert name in README

    for name in ("finetune.sh", "merge.sh", "finetune_merge.sh"):
        assert (REPO_ROOT / "scripts" / name).exists()
        assert f"scripts/{name}" in README


def test_documented_result_paths_match_the_writers():
    """Vision writes under --results_db; the NLP backends ignore it and write
    beside the checkpoints. The README says so, and must keep saying so."""
    import inspect

    from src import eval as eval_module
    from src.backends import nlp_classification_backend

    assert "args.results_db" in inspect.getsource(eval_module._save_target_eval_results)
    assert "results_db" not in inspect.getsource(
        nlp_classification_backend.merge_and_evaluate
    )
    assert "NLP ignores `--results_db`" in README
