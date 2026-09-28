"""Reading and aggregating merge/target-environment result JSONs.

Split out of utils.py so that this half — pure stdlib + pandas, no plotting —
can be imported (by build_comparison_table.py, and by tests) without pulling
in matplotlib/seaborn, which utils.py's plotting helpers need but this data
layer never touches. utils.py re-exports everything here so existing
notebook imports (`from utils import fetch_stats, COMPETITOR_ALL_DICT, ...`)
are unaffected.
"""

import json
import os
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Competitors
# ---------------------------------------------------------------------------

# Keys are `--merge_fn` values, which are also the directory names results are
# written under (see src/merging/registry.py). They used to be the *Python
# function* names instead — "merge_max_abs" for magmax, "sum" for average — so
# result files produced before that change live under the old directory names
# and need renaming (or re-running) to be picked up here.
#
# Order matters: build_color_mapping (utils.py) assigns plot colors by
# position, so inserting or removing an entry recolors every competitor after
# it.
COMPETITOR_ALL_DICT = {
    "finetune": "Baseline",
    "random_mix": "Random Mix",
    "average": "Average",
    "ties": "TIES-Merging",
    "magmax": "MAGMAX",
    "masked_magmax_with_targetdata-labels": "Tunable MAGMAX (Labels)",
    "masked_magmax_with_targetdata-cosine_embedded": "Tunable MAGMAX (Cosine)",
    "masked_magmax_with_targetdata-ot_embedded": "Tunable MAGMAX (OT)",
    "masked_magmax_with_targetdata-mmd_embedded": "Tunable MAGMAX (MMD)",
    # NLP's proposed method has no similarity_metric variants (its preference
    # vector is the target environment's own known ratio; see
    # src/nlp/target_data.py), so its competitor key has no "-<metric>" suffix.
    "masked_magmax_with_targetdata": "Tunable MAGMAX",
}

# ---------------------------------------------------------------------------
# Vision key / path utilities
# ---------------------------------------------------------------------------


def parse_competitor_key(key: str) -> tuple[str, str, str | None, str]:
    """Decompose a competitor_dict key into (merge_fn, similarity_metric, metric_name, metric_name_path).

    Example:
        "masked_magmax_with_targetdata-ot_embedded"
        -> ("masked_magmax_with_targetdata", "ot_embedded_", "ot_embedded", "ot_embedded/")

        "magmax"
        -> ("magmax", "", None, "")
    """
    if "masked_magmax_with_targetdata" in key:
        merge_fn = key.split("-")[0]
        metric_name = key.split("-")[1]
        similarity_metric = f"{metric_name}_"
        metric_name_path = f"{metric_name}/"
    else:
        merge_fn = key
        similarity_metric = ""
        metric_name = None
        metric_name_path = ""
    return merge_fn, similarity_metric, metric_name, metric_name_path


def resolve_dir_path(
    dir_path_shared: str, key: str, dir_name: dict | None = None
) -> str:
    """Replace the "DIR_NAME" placeholder in dir_path_shared when dir_name is provided.
    If dir_name is None, returns dir_path_shared as-is."""
    if dir_name is not None:
        return dir_path_shared.replace(
            "DIR_NAME",
            dir_name["proposed"] if "masked" in key else dir_name["other"],
        )
    return dir_path_shared


def build_file_path_list(
    dir_path: str,
    merge_fn: str,
    metric_name_path: str,
    similarity_metric: str,
    lambda_: float,
    target_id: int,
    seed_list: list[int],
) -> list[str]:
    """Build the list of result JSON file paths for all seeds."""
    return [
        f"{dir_path}/{merge_fn}/{metric_name_path}"
        f"{merge_fn}_lambda{lambda_}_{similarity_metric}target{target_id}_seed{seed}.json"
        for seed in seed_list
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json_files(file_path_list: list[str]) -> list[dict]:
    """Load and return a list of JSON objects from the given file paths."""
    results = []
    for file_path in file_path_list:
        with open(file_path, "r") as f:
            results.append(json.load(f))
    return results


def load_target_data_config(config_path: str) -> list[dict]:
    """Load a target_data_config*.json file and return its dataset_configs list."""
    with open(config_path, "r") as f:
        return json.load(f)["dataset_configs"]


# ---------------------------------------------------------------------------
# Backend-agnostic result loading
#
# fetch_files/fetch_files_all/fetch_stats below are vision-specific: they
# assume the `{merge_fn}_lambda{}_{similarity_metric}_target{id}_seed{s}.json`
# naming under a `--results_db` tree, which is what the camera-ready
# notebooks (make_tables.ipynb etc.) are tuned against — left untouched here
# on purpose. NLP results live under a differently-shaped tree (see
# src/backends/nlp_classification_backend.py: no lambda/similarity_metric
# component, since LSB sweeps neither). ResultPath below is the one seam that
# lets new code (build_comparison_table.py) read either shape through a
# single aggregation function instead of growing a second copy of
# fetch_stats.
# ---------------------------------------------------------------------------

ResultPath = Callable[[str, int, int], str]
"""(competitor_key, target_id, seed) -> path to that run's result JSON."""


def vision_result_path(
    dir_path_shared: str, lambda_: float, dir_name: dict | None = None
) -> ResultPath:
    """A ResultPath for the vision `--results_db` layout, built from the same
    parse_competitor_key/resolve_dir_path this module's legacy functions use
    — one file's worth of naming knowledge, not two."""

    def _path(key: str, target_id: int, seed: int) -> str:
        merge_fn, similarity_metric, _, metric_name_path = parse_competitor_key(key)
        dir_path = resolve_dir_path(dir_path_shared, key, dir_name)
        (path,) = build_file_path_list(
            dir_path, merge_fn, metric_name_path, similarity_metric, lambda_, target_id, [seed]
        )
        return path

    return _path


def nlp_result_path(
    model: str,
    dataset: str,
    epochs: int,
    taskseq_pattern: str,
    sequential_finetuning: bool = True,
    base_dir: str | None = None,
) -> ResultPath:
    """A ResultPath for the NLP layout:
    `{ckpt_dir}/{merge_fn}/target{id}_seed{s}.json`
    (src/backends/nlp_classification_backend.py::merge_and_evaluate). The
    competitor key *is* the --merge_fn value — LSB has no similarity_metric
    variants to disambiguate between.

    Unlike vision's --results_db (one fixed, seed-independent root), NLP
    writes results beside checkpoints, and the fine-tuning seed is baked into
    the checkpoint directory itself. That template
    (src/paths.py::checkpoint_dir/run_dir_name) is re-derived by hand here
    instead of imported: postprocessing/ deliberately never imports src/, so
    the analysis notebooks stay free of this project's ML dependencies (see
    test/characterization/test_merge_registry.py::
    test_postprocessing_labels_stay_in_sync_with_the_registry). If that
    template changes, this needs updating too.
    """
    base_dir = base_dir or os.environ.get("MAGMAX_BASE_DIR", "YOUR_BASE_DIR_FOR_CHECKPOINTS")
    sequential = "sequential_finetuning" if sequential_finetuning else ""

    def _path(key: str, target_id: int, seed: int) -> str:
        run_dir = f"ft-pattern_{taskseq_pattern}-epochs-{epochs}-seed:{seed}"
        ckpt_dir = os.path.join(
            base_dir, "checkpoints", model, sequential, "nlp_classification", dataset, run_dir
        )
        return f"{ckpt_dir}/{key}/target{target_id}_seed{seed}.json"

    return _path


def load_overall_accuracy(
    path_fn: ResultPath,
    competitor_dict: dict[str, str],
    target_ids: list[int],
    seed_list: list[int],
) -> pd.DataFrame:
    """Tidy long-form accuracy table: one row per (competitor, target_id,
    seed), read through `path_fn` — the single place that walks this grid and
    opens a result file, regardless of backend. Callers reduce it with
    ordinary pandas grouping instead of a bespoke loop per aggregation.
    """
    rows = []
    for key in competitor_dict:
        for target_id in target_ids:
            for seed in seed_list:
                with open(path_fn(key, target_id, seed)) as f:
                    overall_accuracy = json.load(f)["overall_accuracy"]
                rows.append(
                    {
                        "competitor": competitor_dict[key],
                        "target_id": target_id,
                        "seed": seed,
                        "overall_accuracy": overall_accuracy,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Accuracy aggregation (vision-specific legacy shapes, kept verbatim for the
# notebooks — see load_overall_accuracy above for the backend-agnostic path)
# ---------------------------------------------------------------------------


def fetch_files(
    targetdata_id: list[int],
    competitor_dict: dict,
    dir_path_shared: str,
    seed_list: list[int],
    lambda_: float,
    dir_name: dict | None = None,
) -> tuple[dict, dict]:
    """Collect mean and std of overall_accuracy across seeds for each target_id.

    Pass dir_name to replace the "DIR_NAME" placeholder in dir_path_shared
    (used in postprocessing_targetdata_for_table and split50).
    Set dir_name=None when dir_path_shared is used directly
    (used in postprocessing_targetdata and to_share).
    """
    import numpy as np

    acc_mean_dict = {key: [] for key in competitor_dict}
    std_mean_dict = {key: [] for key in competitor_dict}

    for target_id in targetdata_id:
        print(f"\nProcessing target data ID: {target_id}")
        for key in competitor_dict:
            merge_fn, similarity_metric, _, metric_name_path = parse_competitor_key(key)
            dir_path = resolve_dir_path(dir_path_shared, key, dir_name)
            file_path_list = build_file_path_list(
                dir_path,
                merge_fn,
                metric_name_path,
                similarity_metric,
                lambda_,
                target_id,
                seed_list,
            )
            dict_list = load_json_files(file_path_list)
            acc_mean_dict[key].append(
                np.mean([d["overall_accuracy"] for d in dict_list])
            )
            std_mean_dict[key].append(
                np.std([d["overall_accuracy"] for d in dict_list])
            )

    return acc_mean_dict, std_mean_dict


def fetch_files_all(
    targetdata_id: list[int],
    competitor_dict: dict,
    dir_path_shared: str,
    seed_list: list[int],
    lambda_: float,
    dir_name: dict | None = None,
) -> dict:
    """Collect all overall_accuracy values (target_id x seed) into a flat list per competitor."""
    acc_dict = {key: [] for key in competitor_dict}

    for target_id in targetdata_id:
        for key in competitor_dict:
            merge_fn, similarity_metric, _, metric_name_path = parse_competitor_key(key)
            dir_path = resolve_dir_path(dir_path_shared, key, dir_name)
            file_path_list = build_file_path_list(
                dir_path,
                merge_fn,
                metric_name_path,
                similarity_metric,
                lambda_,
                target_id,
                seed_list,
            )
            dict_list = load_json_files(file_path_list)
            acc_dict[key] += [d["overall_accuracy"] for d in dict_list]

    return acc_dict


def fetch_stats(
    targetdata_id: list[int],
    competitor_dict: dict,
    dir_path_shared: str,
    seed_list: list[int],
    lambda_: float,
    dir_name: dict | None = None,
) -> tuple[dict, dict]:
    """Compute per-competitor mean/std by first averaging over target_ids per seed,
    then aggregating across seeds (used for table output)."""
    mean_dict = {}
    std_dict = {}

    for key in competitor_dict:
        acc_df = pd.DataFrame(index=seed_list, columns=targetdata_id)
        for target_id in targetdata_id:
            merge_fn, similarity_metric, _, metric_name_path = parse_competitor_key(key)
            dir_path = resolve_dir_path(dir_path_shared, key, dir_name)
            file_path_list = build_file_path_list(
                dir_path,
                merge_fn,
                metric_name_path,
                similarity_metric,
                lambda_,
                target_id,
                seed_list,
            )
            for seed, file_path in zip(seed_list, file_path_list):
                with open(file_path, "r") as f:
                    data = json.load(f)
                acc_df.loc[seed, target_id] = data["overall_accuracy"]

        acc_mean = acc_df.mean(axis=1)
        assert acc_mean.shape[0] == len(seed_list), "Mean accuracy shape mismatch."
        mean_dict[key] = acc_mean.mean()
        std_dict[key] = acc_mean.std()

    return mean_dict, std_dict


def acc_dict_to_dataframe(acc_dict: dict, competitor_dict: dict) -> pd.DataFrame:
    """Convert the dict returned by fetch_files / fetch_files_all to a DataFrame
    with competitor display names as columns."""
    df = pd.DataFrame(acc_dict.values(), index=acc_dict.keys()).T
    df.columns = [competitor_dict[key] for key in df.columns]
    return df
