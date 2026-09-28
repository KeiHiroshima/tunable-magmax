#!/usr/bin/env python
"""Build the paper's main-comparison table (accuracy per method per target
environment, mean +/- std over seeds) for either backend.

This is the "build a table from result JSONs" half of make_tables.ipynb,
pulled out of the notebook so it works for NLP results too (which live under
a different directory layout than vision's --results_db — see
postprocessing/results.py::nlp_result_path). make_tables.ipynb itself is
unchanged and keeps producing the vision camera-ready CSVs it always has;
this script is the new, backend-agnostic way to get the same kind of table.
It only needs results.py (pandas), not utils.py's matplotlib/seaborn-based
plotting half.

Vision example (reproduces one of make_tables.ipynb's tables):
    uv run python postprocessing/build_comparison_table.py --backend vision \\
        --results_db logs/ViT-B-16/sequential_finetuning/class_incremental/DEFAULT_NAME/CIFAR100-5/taskseq_A \\
        --target_config target_data_config --seeds 3,4,5 \\
        --competitors finetune,random_mix,average,ties,magmax,masked_magmax_with_targetdata-labels

NLP example:
    uv run python postprocessing/build_comparison_table.py --backend nlp \\
        --model bert-base-uncased --dataset LSB --epochs 3 --taskseq_pattern A \\
        --target_config target_data_config_lsb --seeds 3,4,5 \\
        --competitors finetune,random_mix,average,ties,magmax,masked_magmax_with_targetdata
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import results`

from results import (  # noqa: E402
    COMPETITOR_ALL_DICT,
    load_overall_accuracy,
    load_target_data_config,
    nlp_result_path,
    vision_result_path,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", required=True, choices=["vision", "nlp"])
    p.add_argument("--target_config", required=True, help="configs/<name>.json, without the extension")
    p.add_argument("--seeds", required=True, help="comma-separated seeds to average over, e.g. 3,4,5")
    p.add_argument(
        "--competitors",
        help="comma-separated --merge_fn keys (vision: optionally '-<similarity_metric>'); "
        "defaults to every entry in COMPETITOR_ALL_DICT applicable to --backend",
    )
    p.add_argument("--out", default="comparison_table.csv", help="where to save the CSV")

    # vision
    p.add_argument("--results_db", help="vision: the --results_db directory results were written under")
    p.add_argument("--lambda_", type=float, default=0.5, help="vision: the --coeff/lambda merges were run with")

    # nlp
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--dataset", default="LSB")
    p.add_argument("--epochs", type=int)
    p.add_argument("--taskseq_pattern", default="A")
    p.add_argument(
        "--sequential_finetuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="nlp: must match the --sequential-finetuning flag finetune_splitted.py was run with",
    )

    return p.parse_args()


def _default_competitors(backend: str) -> dict[str, str]:
    if backend == "vision":
        return dict(COMPETITOR_ALL_DICT)
    # LSB has no similarity_metric variants of the proposed method — only the
    # plain "masked_magmax_with_targetdata" key exists for it.
    return {k: v for k, v in COMPETITOR_ALL_DICT.items() if "-" not in k}


def _build_path_fn(args):
    if args.backend == "vision":
        if not args.results_db:
            raise SystemExit("--results_db is required for --backend vision")
        return vision_result_path(args.results_db, args.lambda_)

    if not args.epochs:
        raise SystemExit("--epochs is required for --backend nlp")
    return nlp_result_path(
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        taskseq_pattern=args.taskseq_pattern,
        sequential_finetuning=args.sequential_finetuning,
    )


def _config_label(config: dict) -> str:
    if config["num_task_to_be_fetched"] < 0:
        return "all"
    ratio = ", ".join(str(r) for r in config["ratio_task_to_be_fetched"])
    return f"{config['num_task_to_be_fetched']}-task ({ratio})"


def build_comparison_table(
    path_fn, competitor_dict: dict[str, str], dataset_configs: list[dict], seeds: list[int]
) -> pd.DataFrame:
    """One row per competitor, one column per target-environment kind (a
    config entry, averaged over its variants and over seeds) plus "Average".
    Mirrors make_tables.ipynb's aggregation: mean over each config's target
    ids first (one number per seed), then mean/std of that across seeds —
    so every seed contributes equally regardless of how many variants a
    config has.
    """
    columns: dict[str, pd.Series] = {}
    for config in dataset_configs:
        if config["num_task_to_be_fetched"] < 0:
            continue  # the "all tasks" entry isn't part of the main comparison grid
        target_ids = [variant["target_id"] for variant in config["variants"]]

        long_df = load_overall_accuracy(path_fn, competitor_dict, target_ids, seeds)
        per_seed = long_df.groupby(["competitor", "seed"])["overall_accuracy"].mean()
        mean = per_seed.groupby("competitor").mean()
        std = per_seed.groupby("competitor").std()
        columns[_config_label(config)] = mean.map("{:.3f}".format) + " ± " + std.map("{:.3f}".format)

    table = pd.DataFrame(columns).reindex(list(competitor_dict.values()))
    return table


def main():
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    competitor_dict = (
        {k: COMPETITOR_ALL_DICT[k] for k in args.competitors.split(",")}
        if args.competitors
        else _default_competitors(args.backend)
    )
    dataset_configs = load_target_data_config(str(REPO_ROOT / "configs" / f"{args.target_config}.json"))
    path_fn = _build_path_fn(args)

    table = build_comparison_table(path_fn, competitor_dict, dataset_configs, seeds)
    print(table)
    table.to_csv(args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
