import argparse
import random

import numpy as np
import torch
from src.config import DATA_DIR, OPENCLIP_CACHE_DIR


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
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
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
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
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
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

    # CL METHODS
    parser.add_argument("--lwf_lamb", type=float, default=0.0, help="LWF lambda")
    parser.add_argument("--ewc_lamb", type=float, default=0.0, help="EWC lambda")
    parser.add_argument(
        "--lamb_case",
        type=str,
        default="ascending",
        choices=["ascending", "constant", "decaying", "pow"],
    )

    # OTHER
    parser.add_argument("--seed", default=5, type=int)
    parser.add_argument(
        "--wandb_entity_name", type=str, default="YOUR_WANDB_ENTITY_NAME"
    )

    parser.add_argument(
        "--taskseq_pattern",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="The task sequence pattern to use.",
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
            "hpo",
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
        "--target_config",
        type=str,
        default="target_data_config",
        help="Configuration for target data.",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = (
        f"cuda:{parsed_args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    seed_everything(parsed_args.seed)

    assert parsed_args.lwf_lamb == 0.0 or parsed_args.ewc_lamb == 0.0, (
        "Lambda for LWF and EWC are mutually exclusive"
    )

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
