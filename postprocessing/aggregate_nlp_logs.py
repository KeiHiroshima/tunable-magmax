#!/usr/bin/env python
"""Aggregate every StdCL/LSB result copied under logs/nlp into the paper's
Table 2 layout: one row per method, one column per target environment
(D_tar,1..5, i.e. one mixing ratio of configs/target_data_config_lsb.json)
plus "Average", each cell the mean +/- std over seeds.

Unlike build_comparison_table.py (one model/dataset/order per invocation,
read from beside the checkpoints), this walks a copied results tree

    {root}/{model}/{dataset}[-lora]/ft-pattern_{order}-epochs-{e}-seed:{s}/{merge_fn}/target{id}_seed{s}.json

discovers every (model, dataset, order, seed) in it, and prints one table per
(model, dataset, order) plus one per (model, dataset) pooled over orders.

Aggregation follows make_tables.ipynb/build_comparison_table.py: for each
seed, average a column's target ids (and, for the pooled table, its orders)
first, then take mean/std (ddof=1) of those per-seed numbers — so the std is
the spread across seeds only. "Average" is the per-seed mean of the five
D_tar columns, reduced the same way. The all-tasks environment is not one of
the paper's columns; --include_all_tasks appends it after "Average" (it does
not enter "Average").

    uv run python postprocessing/aggregate_nlp_logs.py
    uv run python postprocessing/aggregate_nlp_logs.py --root logs/nlp --out logs/nlp/summary.csv
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import results`

from results import COMPETITOR_ALL_DICT, load_overall_accuracy, load_target_data_config  # noqa: E402

RUN_DIR = re.compile(r"ft-pattern_(?P<order>\w+)-epochs-(?P<epochs>\d+)-seed:(?P<seed>\d+)")
# StdCL/LSB's proposed method has no "-<similarity_metric>" variants.
NLP_COMPETITORS = {k: v for k, v in COMPETITOR_ALL_DICT.items() if "-" not in k}
ALL_TASKS = "All tasks"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT / "logs" / "nlp"), help="the copied results tree")
    p.add_argument("--target_config", default="target_data_config_lsb", help="configs/<name>.json")
    p.add_argument("--include_all_tasks", action="store_true", help="also report the all-tasks environment")
    p.add_argument("--digits", type=int, default=2, help="decimal places, accuracies in %%")
    p.add_argument("--out", help="save every cell as a long-form CSV (setting, order, method, column, mean, std, n_seeds)")
    return p.parse_args()


def discover_runs(root: Path) -> dict[tuple[str, str], dict[str, list[tuple[int, int]]]]:
    """{(model, scope): {order: [(epochs, seed), ...]}} for every run directory under root."""
    runs: dict = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(root.glob("*/*/ft-pattern_*")):
        m = RUN_DIR.fullmatch(run_dir.name)
        if m is None:
            continue
        model, scope = run_dir.parent.parent.name, run_dir.parent.name
        runs[(model, scope)][m["order"]].append((int(m["epochs"]), int(m["seed"])))
    return runs


def logs_result_path(root: Path, model: str, scope: str, order: str, epochs: int):
    """A results.ResultPath over the copied tree (no checkpoints/.../sequential_finetuning prefix)."""

    def _path(key: str, target_id: int, seed: int) -> str:
        run_dir = root / model / scope / f"ft-pattern_{order}-epochs-{epochs}-seed:{seed}"
        return str(run_dir / key / f"target{target_id}_seed{seed}.json")

    return _path


def column_labels(dataset_configs: list[dict]) -> list[tuple[str, str, list[int]]]:
    """(column name, mixing-ratio label, target ids) per environment kind, in config order."""
    columns, k = [], 0
    for config in dataset_configs:
        target_ids = [v["target_id"] for v in config["variants"]]
        if config["num_task_to_be_fetched"] < 0:
            columns.append((ALL_TASKS, "uniform", target_ids))
            continue
        k += 1
        ratio = ", ".join(f"{r:g}" for r in config["ratio_task_to_be_fetched"])
        columns.append((f"D_tar,{k}", f"({ratio})", target_ids))
    return columns


def per_seed_accuracy(long_df: pd.DataFrame, columns, competitors) -> pd.DataFrame:
    """One accuracy per (method, seed, column): the mean over that column's
    target ids (and over orders, if long_df holds several), plus "Average"."""
    col_of = {tid: name for name, _, tids in columns for tid in tids}
    df = long_df.assign(column=long_df["target_id"].map(col_of))
    wide = df.groupby(["competitor", "seed", "column"])["overall_accuracy"].mean().unstack("column")
    main = [name for name, _, _ in columns if name != ALL_TASKS]
    wide["Average"] = wide[main].mean(axis=1)
    order = main + ["Average"] + ([ALL_TASKS] if ALL_TASKS in wide else [])
    return wide[order].reindex(list(competitors.values()), level="competitor")


def summarize(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    grouped = per_seed.groupby(level="competitor", sort=False)
    return grouped.mean(), grouped.std(ddof=1), grouped.size()


def format_table(mean: pd.DataFrame, std: pd.DataFrame, columns, digits: int) -> pd.DataFrame:
    """Table 2-style strings in %, with the best mean of each column marked by '*'."""
    ratio_of = {name: ratio for name, ratio, _ in columns}
    best = mean.round(digits + 2).eq(mean.round(digits + 2).max())
    cells = (
        (mean * 100).map(lambda v: f"{v:.{digits}f}")
        + "±"
        + (std * 100).map(lambda v: f"{v:.{digits}f}")
        + best.map(lambda b: "*" if b else " ")
    )
    cells.columns = pd.MultiIndex.from_tuples([(c, ratio_of.get(c, "")) for c in cells.columns])
    cells.index.name = None
    return cells


def main():
    args = _parse_args()
    root = Path(args.root)
    columns = column_labels(load_target_data_config(str(REPO_ROOT / "configs" / f"{args.target_config}.json")))
    if not args.include_all_tasks:
        columns = [c for c in columns if c[0] != ALL_TASKS]
    target_ids = [tid for _, _, tids in columns for tid in tids]

    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"no ft-pattern_* run directories under {root}")

    records = []
    pd.set_option("display.width", 250, "display.max_columns", None)
    for (model, scope), orders in sorted(runs.items()):
        pooled = []
        for order, epochs_seeds in sorted(orders.items()):
            per_order = []
            for epochs in sorted({e for e, _ in epochs_seeds}):
                seeds = sorted(s for e, s in epochs_seeds if e == epochs)
                path_fn = logs_result_path(root, model, scope, order, epochs)
                per_order.append(load_overall_accuracy(path_fn, NLP_COMPETITORS, target_ids, seeds))
            per_order = pd.concat(per_order).assign(order=order)
            pooled.append(per_order)
            _report(f"{model} {scope}  Order {order}", order, per_order, columns, args, model, scope, records)
        orders_label = ",".join(sorted(orders))
        _report(
            f"{model} {scope}  Orders {orders_label} pooled (per seed: mean over orders)",
            "all", pd.concat(pooled), columns, args, model, scope, records,
        )

    if args.out:
        pd.DataFrame(records).to_csv(args.out, index=False)
        print(f"Saved {len(records)} cells to {args.out}")


def _report(title, order, long_df, columns, args, model, scope, records):
    seeds = sorted(int(s) for s in long_df["seed"].unique())
    mean, std, n = summarize(per_seed_accuracy(long_df, columns, NLP_COMPETITORS))
    print(f"\n### {title}   [mean±std over seeds {seeds}, accuracy %, * = best]")
    print(format_table(mean, std, columns, args.digits).to_string())
    for method in mean.index:
        for column in mean.columns:
            records.append(
                {
                    "model": model,
                    "setting": scope,
                    "order": order,
                    "method": method,
                    "column": column,
                    "mean": mean.loc[method, column],
                    "std": std.loc[method, column],
                    "n_seeds": n[method],
                }
            )


if __name__ == "__main__":
    main()
