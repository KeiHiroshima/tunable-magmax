import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

CUSTOM_PALETTE = ["#003f5c", "#444e86", "#955196", "#dd5182", "#ff6e54", "#ffa600"]
CUSTOM_HUE = ["#004c6d", "#346888", "#5886a5", "#7aa6c2", "#9dc6e0", "#c1e7ff"]
CUSTOM_DIVERGENT = [
    "#00876c",
    "#6aaa96",
    "#aecdc2",
    "#f1f1f1",
    "#f0b8b8",
    "#e67f83",
    "#d43d51",
]

# Full ordered competitor list used for consistent color assignment across notebooks
COMPETITOR_ALL_DICT = {
    "finetune": "Baseline",
    "select_one_task_vector": "Single task vector",
    "merge_rnd_mix": "Random Mix",
    "sum": "Average",
    "ties": "TIES-Merging",
    "merge_max_abs": "MAGMAX",
    "merge_max_abs_masked_with_targetdata-labels": "Tunable MAGMAX (Labels)",
    "merge_max_abs_masked_with_targetdata-cosine_embedded": "Tunable MAGMAX (Cosine)",
    "merge_max_abs_masked_with_targetdata-ot_embedded": "Tunable MAGMAX (OT)",
    "merge_max_abs_masked_with_targetdata-mmd_embedded": "Tunable MAGMAX (MMD)",
}

# ---------------------------------------------------------------------------
# Plot style setup
# ---------------------------------------------------------------------------


def setup_plot_style():
    """Apply shared matplotlib/seaborn style settings.
    Call at the top of each notebook (%matplotlib inline must be set per-notebook)."""
    plt.style.use("fivethirtyeight")
    rcParams["figure.figsize"] = (16, 5)
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False
    rcParams["font.size"] = 12
    rcParams["savefig.dpi"] = 300
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    plt.rc("xtick", labelsize=11)
    plt.rc("ytick", labelsize=11)
    sns.set_style("whitegrid")
    sns.set_palette(CUSTOM_PALETTE)
    sns.set_palette("deep")
    sns.set_context("notebook")


# ---------------------------------------------------------------------------
# Key / path utilities
# ---------------------------------------------------------------------------


def parse_competitor_key(key: str) -> tuple[str, str, str | None, str]:
    """Decompose a competitor_dict key into (merge_fn, similarity_metric, metric_name, metric_name_path).

    Example:
        "merge_max_abs_masked_with_targetdata-ot_embedded"
        -> ("merge_max_abs_masked_with_targetdata", "ot_embedded_", "ot_embedded", "ot_embedded/")

        "merge_max_abs"
        -> ("merge_max_abs", "", None, "")
    """
    if "merge_max_abs_masked_with_targetdata" in key:
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
# Accuracy aggregation
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


# ---------------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------------


def build_color_mapping(competitor_all_dict: dict | None = None) -> dict:
    """Return a color mapping that assigns consistent colors to each competitor
    based on the fixed ordering in competitor_all_dict."""
    if competitor_all_dict is None:
        competitor_all_dict = COMPETITOR_ALL_DICT
    all_competitors = list(competitor_all_dict.values())
    full_palette = sns.color_palette()[: len(all_competitors)]
    return {name: color for name, color in zip(all_competitors, full_palette)}


def save_figure(save_dir: str, filename: str, dpi: int = 300) -> None:
    """Create save_dir if needed, then save the current figure as both PDF and PNG."""
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), bbox_inches="tight", dpi=dpi)


def plot_overall_accuracy_boxplot(
    acc_mean_df: pd.DataFrame,
    save_dir: str,
    suffix: str,
    competitor_all_dict: dict | None = None,
    figsize: tuple = (8, 8),
    rotation: int = 25,
    ylabel: str = "Overall Accuracy",
    fontsize: int = 24,
    show_axvline: bool = True,
) -> None:
    """Draw a boxplot with colors fixed across notebooks and save to save_dir.

    Parameters
    ----------
    show_axvline : bool
        When True, draw a vertical dashed separator and thicken spine borders
        based on the number of columns (style used in postprocessing_targetdata).
        Set False to omit these decorations (style used in split50).
    """
    color_mapping = build_color_mapping(competitor_all_dict)
    current_colors = [
        color_mapping[col] for col in acc_mean_df.columns if col in color_mapping
    ]

    plt.close("all")
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=acc_mean_df, palette=current_colors)
    plt.xlabel("", fontsize=fontsize)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=rotation, ha="right", fontsize=fontsize * 0.85
    )
    plt.ylabel(ylabel, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize * 0.85)

    if show_axvline:
        if len(acc_mean_df.columns) == 7:
            plt.axvline(x=5.5, color="gray", linestyle="--")
        elif len(acc_mean_df.columns) == 5:
            plt.axvline(x=3.5, color="gray", linestyle="--")
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor("black")

    plt.grid(axis="y")
    plt.tight_layout()
    save_figure(save_dir, f"overall_accuracy_boxplot_{suffix}")


def plot_single_boxplot(
    ax,
    acc_mean_df: pd.DataFrame,
    color_mapping: dict,
    fontsize: int = 24,
    show_ylabel: bool = True,
) -> None:
    """Draw one panel of a multi-panel boxplot figure onto the given axes."""
    current_colors = [
        color_mapping[col] for col in acc_mean_df.columns if col in color_mapping
    ]
    sns.boxplot(data=acc_mean_df, palette=current_colors, ax=ax)
    ax.set_xlabel("", fontsize=fontsize)
    ax.set_xticklabels([])
    ax.set_ylabel("Overall Accuracy" if show_ylabel else "", fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize * 0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")
    ax.grid(axis="y")


def plot_two_boxplots(
    acc_mean_dfs: list[pd.DataFrame],
    suffix: str,
    color_mapping: dict,
    save_dir: str,
    fontsize: int = 24,
    show: bool = False,
) -> None:
    """Plot the first two configs side by side without a legend and save."""
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, (ax, acc_mean_df) in enumerate(zip(axes, acc_mean_dfs[:2])):
        plot_single_boxplot(
            ax, acc_mean_df, color_mapping, fontsize, show_ylabel=(idx == 0)
        )
    plt.tight_layout()
    save_figure(save_dir, f"overall_accuracy_boxplot_first2_{suffix}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_three_boxplots_with_legend(
    acc_mean_dfs: list[pd.DataFrame],
    suffix: str,
    color_mapping: dict,
    all_competitors: list[str],
    save_dir: str,
    fontsize: int = 24,
    show: bool = False,
) -> None:
    """Plot configs 3-5 side by side with a shared legend below and save."""
    plt.close("all")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(24, 10),
        gridspec_kw={"height_ratios": [1, 0.15], "hspace": 0.4},
    )

    all_columns: set[str] = set()
    for acc_mean_df in acc_mean_dfs[2:5]:
        all_columns.update(acc_mean_df.columns)

    for idx, (ax, acc_mean_df) in enumerate(zip(axes[0], acc_mean_dfs[2:5])):
        plot_single_boxplot(
            ax, acc_mean_df, color_mapping, fontsize, show_ylabel=(idx == 0)
        )

    for ax in axes[1]:
        ax.axis("off")

    legend_elements = [
        Patch(facecolor=color_mapping[col], label=col)
        for col in all_competitors
        if col in all_columns
    ]
    axes[1, 1].legend(
        handles=legend_elements,
        ncol=min(4, len(legend_elements)),
        fontsize=fontsize * 0.9,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        bbox_to_anchor=(0.5, 0.15),
        loc="lower center",
    )

    save_figure(save_dir, f"overall_accuracy_boxplot_last3_{suffix}", dpi=300)
    if show:
        plt.show()
    plt.close()
