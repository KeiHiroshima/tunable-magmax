import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Patch

# Result loading/aggregation lives in results.py, which has no matplotlib/
# seaborn dependency (build_comparison_table.py and tests import it directly
# for that reason). Re-exported here so existing notebook imports —
# `from utils import (COMPETITOR_ALL_DICT, setup_plot_style, fetch_stats, ...)`
# mixing data and plotting names in one line — keep working unchanged.
from results import (  # noqa: F401
    COMPETITOR_ALL_DICT,
    acc_dict_to_dataframe,
    build_file_path_list,
    fetch_files,
    fetch_files_all,
    fetch_stats,
    load_json_files,
    load_overall_accuracy,
    load_target_data_config,
    nlp_result_path,
    parse_competitor_key,
    resolve_dir_path,
    vision_result_path,
)

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
