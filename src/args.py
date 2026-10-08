import argparse
import random

import numpy as np
import torch
from src.config import DATA_DIR, OPENCLIP_CACHE_DIR
from src.nlp.long_sequence_benchmark import TASK_ORDERS

# Defaults that differ between the StdCL/LSB (T5) pipeline and everything
# else. The flags default to None and are resolved after parsing
# (_resolve_training_defaults), so vision and CITB keep exactly the values
# they had before these benchmarks existed.
_DEFAULT_LR = 1e-5
_DEFAULT_WD = 0.1
_DEFAULT_LR_SCHEDULE = "cosine"
# O-LoRA's T5 setting: AdamW at a constant 1e-3 for LoRA, no weight decay.
# Full fine-tuning at 1e-3 is far too aggressive for AdamW, hence 1e-4.
_T5_DEFAULT_LR = {"full": 1e-4, "lora": 1e-3}
_T5_DEFAULT_WD = 0.0
_T5_DEFAULT_LR_SCHEDULE = "constant"


def warmup_ratio(value: str) -> float:
    """--warmup_ratio must leave room for the cosine decay to actually run.

    At 1.0 the whole schedule is linear warmup: the learning rate climbs to
    --lr on the last step and never decays. Anything above that is worse — it
    never even reaches --lr. This is the failure the previous step-count flag
    produced silently, so it is rejected at the boundary instead.
    """
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError(
            f"--warmup_ratio must be in (0, 1), got {ratio}"
        )
    return ratio


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_training_defaults(parser, args):
    """Fill --lr/--wd/--lr_schedule for whichever pipeline --dataset selects,
    and reject combinations only one pipeline understands."""
    t5 = args.dataset in TASK_ORDERS
    if t5:
        if args.taskseq_pattern not in TASK_ORDERS[args.dataset]:
            parser.error(
                f"--taskseq_pattern {args.taskseq_pattern} is not an order of "
                f"{args.dataset}; choose one of {sorted(TASK_ORDERS[args.dataset])}"
            )
    elif args.finetune_mode != "full":
        parser.error("--finetune_mode lora is only implemented for StdCL/LSB")

    if args.lr is None:
        args.lr = _T5_DEFAULT_LR[args.finetune_mode] if t5 else _DEFAULT_LR
    if args.wd is None:
        args.wd = _T5_DEFAULT_WD if t5 else _DEFAULT_WD
    if args.lr_schedule is None:
        args.lr_schedule = _T5_DEFAULT_LR_SCHEDULE if t5 else _DEFAULT_LR_SCHEDULE


def parse_arguments(argv=None):
    """Parse the CLI. `argv` defaults to None, i.e. sys.argv[1:], so every
    existing call site is unchanged; passing it explicitly lets tests exercise
    the real parser without having to patch sys.argv."""
    parser = argparse.ArgumentParser()

    # DATASETS
    parser.add_argument(
        "--data_location",
        type=str,
        default=DATA_DIR,  # os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )

    # MODEL/TRAINING
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Learning rate. Default 1e-5; for StdCL/LSB 1e-4 under "
            "--finetune_mode full and 1e-3 under --finetune_mode lora."
        ),
    )
    parser.add_argument(
        "--wd", type=float, default=None, help="Weight decay. Default 0.1; 0 for StdCL/LSB."
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default=None,
        choices=["cosine", "constant"],
        help="Default cosine (with --warmup_ratio warmup); constant for StdCL/LSB.",
    )
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="full",
        choices=["full", "lora"],
        help=(
            "StdCL/LSB only. full: train every parameter, save a checkpoint per "
            "task. lora: train a LoRA adapter per task, merged into the weights "
            "before the next one; only adapters are saved."
        ),
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer step (NLP only). Effective batch = --batch_size x this.",
    )
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_ratio",
        type=warmup_ratio,
        default=0.1,
        help=(
            "Fraction of each task's schedule spent in linear warmup before the "
            "cosine decay starts. A ratio rather than a step count because task "
            "lengths differ by orders of magnitude, both between settings and "
            "between tasks of one NLP benchmark."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument("--skip-eval", action="store_true")

    # LOAD/SAVE PATHS
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--results_db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=OPENCLIP_CACHE_DIR,
        help="Directory for caching models from OpenCLIP",
    )

    # CL SPLITS
    parser.add_argument(
        "--n_splits",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--split_strategy", type=str, default=None, choices=[None, "data", "class"]
    )
    parser.add_argument("--sequential-finetuning", action="store_true")

    # OTHER
    parser.add_argument("--seed", default=5, type=int)
    parser.add_argument(
        "--wandb_entity_name", type=str, default="YOUR_WANDB_ENTITY_NAME"
    )

    parser.add_argument(
        "--taskseq_pattern",
        type=str,
        default="A",
        choices=["A", "B", "C", *sorted(o for orders in TASK_ORDERS.values() for o in orders)],
        help=(
            "The task sequence pattern to use: A/B/C for the vision and CITB "
            "benchmarks, O-LoRA's Orders 1-3 for StdCL and 4-6 for LSB."
        ),
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use for training."
    )
    parser.add_argument(
        "--merge_fn", type=str, default="magmax", help="Merging function to use."
    )
    parser.add_argument(
        "--datasets", type=str, default=None, help="Comma-separated list of datasets."
    )

    parser.add_argument(
        "--redo",
        default=False,
        action="store_true",
        help="Whether to redo the experiment even if the results file exists.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Alpha parameter for masked_magmax merging.",
    )

    # TARGET DATASET
    parser.add_argument(
        "--num_target_data",
        type=int,
        default=1000,
        help="Number of data points in the target dataset.",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=1,
        help="Number of target datasets to evaluate on.",
    )
    """parser.add_argument(
        "--num_data_from_tasks",
        type=lambda x: [int(item) for item in x.split(",")],
        default=None,
        help="Comma-separated list of number of data points from each task.",
    )"""
    parser.add_argument(
        "--num_train_data_each_task",
        type=int,  # lambda x: [int(item) for item in x.split(",")],
        default=None,
        help="Comma-separated list of number of training data points from each task.",
    )
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=[
            "labels",
            "cosine",
            "mmd",
            "ot",
            "cosine_embedded",
            "mmd_embedded",
            "ot_embedded",
        ],
        help="Similarity metric to use for masked_magmax_with_targetdata merging.",
    )
    parser.add_argument(
        "--logger_mode",
        type=str,
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Logger mode.",
    )
    parser.add_argument(
        "--scaling_coef",
        type=float,
        default=0.5,
        help="StdCL/LSB: the merged task vector is added to the zero-shot model scaled by this.",
    )
    parser.add_argument(
        "--eval_full_testsets",
        action="store_true",
        help=(
            "StdCL/LSB: also score each single-model merge on every task's whole "
            "test split and report O-LoRA's Average Accuracy."
        ),
    )
    parser.add_argument(
        "--target_config",
        type=str,
        default="target_data_config",
        help="Configuration for target data.",
    )

    parsed_args = parser.parse_args(argv)
    _resolve_training_defaults(parser, parsed_args)
    parsed_args.device = (
        f"cuda:{parsed_args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    seed_everything(parsed_args.seed)

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
