"""src/target_env.py — the configs/target_data_config*.json schema.

The `-1` conventions in these files used to be re-implemented in src/eval.py
and src/nlp/target_data.py (and are still read, for labelling only, by two
postprocessing notebooks). The parity test below replays both old
implementations against every config file in the repository, so the
consolidation is checked on the real inputs rather than on invented ones.
"""

import glob
import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from src.target_env import TargetEnvSpec, config_path, load_target_envs

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILES = sorted(
    os.path.basename(p)[: -len(".json")]
    for p in glob.glob(str(REPO_ROOT / "configs" / "target_data_config*.json"))
)


def _args(target_config="target_data_config", n_splits=5, num_target_data=1000):
    return Namespace(
        target_config=target_config,
        n_splits=n_splits,
        num_target_data=num_target_data,
    )


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """load_target_envs resolves "configs/..." relative to the working
    directory, exactly as the entry points do."""
    monkeypatch.chdir(REPO_ROOT)


# --- the conventions -------------------------------------------------------


def test_negative_task_count_means_every_task(tmp_path, monkeypatch):
    config = _write_config(tmp_path, monkeypatch, num_task_to_be_fetched=-1, ratio=[-1])

    envs = list(load_target_envs(_args(target_config=config, n_splits=7), n_tasks=7))

    assert [e.n_tasks_fetched for e in envs] == [7]


def test_negative_ratio_means_an_even_split(tmp_path, monkeypatch):
    config = _write_config(tmp_path, monkeypatch, num_task_to_be_fetched=4, ratio=[-1])

    (env,) = load_target_envs(_args(target_config=config), n_tasks=10)

    assert env.ratio == [0.25, 0.25, 0.25, 0.25]


def test_explicit_ratio_is_passed_through(tmp_path, monkeypatch):
    config = _write_config(tmp_path, monkeypatch, num_task_to_be_fetched=3, ratio=[0.6, 0.2, 0.2])

    (env,) = load_target_envs(_args(target_config=config), n_tasks=10)

    assert env.ratio == [0.6, 0.2, 0.2]


def test_both_conventions_combine(tmp_path, monkeypatch):
    """The "all tasks, evenly" entry every config file ends with."""
    config = _write_config(tmp_path, monkeypatch, num_task_to_be_fetched=-1, ratio=[-1])

    (env,) = load_target_envs(_args(target_config=config), n_tasks=4)

    assert env.n_tasks_fetched == 4
    assert env.ratio == [0.25] * 4


def test_ratio_length_must_match_the_task_count(tmp_path, monkeypatch):
    """A hand-edited config with mismatched fields should fail loudly here
    rather than producing a silently truncated mixture: the sampling code zips
    the two together."""
    config = _write_config(tmp_path, monkeypatch, num_task_to_be_fetched=3, ratio=[0.5, 0.5])

    with pytest.raises(AssertionError, match="ratio has 2 entries"):
        list(load_target_envs(_args(target_config=config), n_tasks=10))


# --- num_target_data resolution --------------------------------------------


def test_split_config_supplies_its_own_size():
    args = _args(target_config="target_data_config_split5", n_splits=5, num_target_data=999)

    envs = list(load_target_envs(args, n_tasks=5))

    assert {e.num_target_data for e in envs} == {200, 400, 600, 800, 1000}


def test_mismatched_split_config_falls_back_to_the_flag():
    """target_data_config_split5 sizes environments for a 5-way split. Asked
    for with --n_splits 20 its numbers do not apply, so --num_target_data wins.
    """
    args = _args(target_config="target_data_config_split5", n_splits=20, num_target_data=999)

    envs = list(load_target_envs(args, n_tasks=20))

    assert {e.num_target_data for e in envs} == {999}


def test_configs_without_the_key_do_not_need_n_splits():
    """--n_splits describes the vision class-incremental splits; the NLP
    backends pass configs that never carry a per-entry size, so resolving one
    must not reach for a flag that means nothing to them."""
    args = Namespace(target_config="target_data_config_lsb", num_target_data=500)

    envs = list(load_target_envs(args, n_tasks=15))

    assert {e.num_target_data for e in envs} == {500}


def test_general_config_uses_the_flag():
    args = _args(target_config="target_data_config", n_splits=5, num_target_data=1234)

    envs = list(load_target_envs(args, n_tasks=5))

    assert {e.num_target_data for e in envs} == {1234}


# --- iteration order --------------------------------------------------------


def test_variants_are_yielded_in_file_order():
    """target_id names the result file, so the order these come out in is part
    of the contract."""
    args = _args()
    raw = json.loads((REPO_ROOT / "configs" / "target_data_config.json").read_text())
    expected = [
        v["target_id"] for c in raw["dataset_configs"] for v in c["variants"]
    ]

    assert [e.target_id for e in load_target_envs(args, n_tasks=5)] == expected


def test_seed_comes_from_the_variant_not_the_entry():
    args = _args()
    raw = json.loads((REPO_ROOT / "configs" / "target_data_config.json").read_text())
    expected = [v["random_seed"] for c in raw["dataset_configs"] for v in c["variants"]]

    assert [e.seed for e in load_target_envs(args, n_tasks=5)] == expected


# --- parity with the implementations this replaced --------------------------


def _legacy_vision(args, n_splits):
    """src/eval.py's inline handling, before the consolidation."""
    with open(config_path(args.target_config)) as f:
        configs = json.load(f)["dataset_configs"]
    out = []
    num_target_data = args.num_target_data
    for config in configs:
        n_fetch = (
            n_splits
            if config["num_task_to_be_fetched"] < 0
            else config["num_task_to_be_fetched"]
        )
        ratio = (
            [1.0 / n_fetch for _ in range(n_fetch)]
            if config["ratio_task_to_be_fetched"][0] == -1
            else config["ratio_task_to_be_fetched"]
        )
        num_target_data = (
            config["num_target_data"]
            if args.target_config == f"target_data_config_split{args.n_splits}"
            else num_target_data
        )
        for variant in config["variants"]:
            out.append(
                (variant["target_id"], variant["random_seed"], n_fetch, list(ratio), num_target_data)
            )
    return out


def _legacy_nlp(args, n_tasks):
    """src/nlp/target_data.py's handling, before the consolidation.

    num_to_fetch was normalised inside select_target_tasks, and the ratio was
    resolved against the *resulting* selection length.
    """
    with open(config_path(args.target_config)) as f:
        configs = json.load(f)["dataset_configs"]
    out = []
    for config in configs:
        raw_n = config["num_task_to_be_fetched"]
        n_selected = n_tasks if raw_n < 0 else raw_n
        raw_ratio = config["ratio_task_to_be_fetched"]
        ratio = (
            [1.0 / n_selected for _ in range(n_selected)]
            if raw_ratio[0] == -1
            else raw_ratio
        )
        for variant in config["variants"]:
            out.append(
                (
                    variant["target_id"],
                    variant["random_seed"],
                    n_selected,
                    list(ratio),
                    args.num_target_data,
                )
            )
    return out


def _as_tuples(envs):
    return [
        (e.target_id, e.seed, e.n_tasks_fetched, e.ratio, e.num_target_data)
        for e in envs
    ]


@pytest.mark.parametrize("target_config", CONFIG_FILES)
@pytest.mark.parametrize("n_splits", [5, 20, 50])
def test_parity_with_the_old_vision_handling(target_config, n_splits):
    args = _args(target_config=target_config, n_splits=n_splits)

    assert _as_tuples(load_target_envs(args, n_tasks=n_splits)) == _legacy_vision(
        args, n_splits
    )


@pytest.mark.parametrize("target_config", ["target_data_config", "target_data_config_lsb"])
def test_parity_with_the_old_nlp_handling(target_config):
    """The NLP path only ever used the two general configs, and never read a
    per-entry num_target_data."""
    args = _args(target_config=target_config, n_splits=15)

    assert _as_tuples(load_target_envs(args, n_tasks=15)) == _legacy_nlp(args, 15)


def test_every_config_in_the_repository_loads():
    """A config that cannot be read is a broken experiment, not a broken test."""
    for target_config in CONFIG_FILES:
        n_splits = 5
        if target_config.startswith("target_data_config_split"):
            n_splits = int(target_config.rsplit("split", 1)[1])
        envs = list(load_target_envs(_args(target_config, n_splits), n_tasks=n_splits))

        assert envs, target_config
        assert all(isinstance(e, TargetEnvSpec) for e in envs)
        for e in envs:
            assert e.num_target_data > 0
            assert abs(sum(e.ratio) - 1.0) < 0.02, (target_config, e.target_id)


# --- helper -----------------------------------------------------------------


def _write_config(tmp_path, monkeypatch, *, num_task_to_be_fetched, ratio):
    """Write a one-entry config into a temporary configs/ directory."""
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    name = "tmp_target_config"
    (configs / f"{name}.json").write_text(
        json.dumps(
            {
                "dataset_configs": [
                    {
                        "num_task_to_be_fetched": num_task_to_be_fetched,
                        "ratio_task_to_be_fetched": ratio,
                        "variants": [{"target_id": 1, "random_seed": 42}],
                    }
                ]
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return name
