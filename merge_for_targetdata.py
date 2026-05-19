import os
import warnings

import wandb
from src.args import parse_arguments
from src.config import BASE_DIR, get_zeroshot_checkpoint
from src.eval import evaluate_merged_fts_on_target_data
from src.merging.task_vector import TaskVector
from src.merging.task_vectors import (
    finetune,
    merge_max_abs,
    merge_max_abs_masked_with_targetdata,
    merge_rnd_mix,
    select_one_task_vector,
    ties,
)
from src.utils import setup_logging

warnings.simplefilter("ignore")

# Config
args = parse_arguments()
pretrained_checkpoint = get_zeroshot_checkpoint(args.model)


def main():
    logger = setup_logging(level=args.logger_mode)

    suffix = ""
    if args.lwf_lamb > 0.0:
        method = "lwf"
        args.save = f"checkpoints/{args.model}/lwf"
        suffix = f"-lamb:{args.lwf_lamb}"
    elif args.ewc_lamb > 0.0:
        method = "ewc"
        args.save = f"checkpoints/{args.model}/ewc"
        suffix = f"-lamb:{args.ewc_lamb}"
    elif args.sequential_finetuning:
        method = "seq-ft"
        args.save = f"checkpoints/{args.model}/sequential_finetuning/{args.split_strategy}_incremental"
    else:
        method = "ind-ft"
        args.save = f"checkpoints/{args.model}/{args.split_strategy}_incremental"

    name = f"merging_target-{args.dataset}-{args.n_splits}-{method}"
    args.save = os.path.join(BASE_DIR, args.save)

    task_vectors = [
        TaskVector(
            pretrained_checkpoint,
            f"{args.save}/{args.dataset}-{args.n_splits}/ft-pattern_{args.taskseq_pattern}-epochs-{args.epochs}-seed:{args.seed}{suffix}/finetuned_{_idx}.pt",
        )
        for _idx in range(args.n_splits)
    ]

    merge_fn_dict = {
        "finetune": (finetune, [0.5]),  # , 1.0
        "select_one_task_vector": (select_one_task_vector, [0.5]),  # , 1.0
        "masked_magmax_with_targetdata": (
            merge_max_abs_masked_with_targetdata,
            [0.5],
        ),  # , 1.0
        "magmax": (merge_max_abs, [0.5]),  # , 1.0
        "random_mix": (merge_rnd_mix, [0.5]),  # , 1.0
        "average": (sum, [0.5]),  # , 1.0
        "ties": (ties, [0.5]),  # , 1.0
    }
    f, coeffs = merge_fn_dict[args.merge_fn]

    for coeff in coeffs:
        args.coeff = coeff

        wandb.init(
            project="magmax",
            group="merging-CIL-target",
            entity=args.wandb_entity_name,
            mode="online",
            name=f"{name}-{args.taskseq_pattern}-{args.merge_fn}_lambda{args.coeff}_{suffix}_seed{args.seed}",
            tags=["merging-target", "CIL", f"{args.dataset}", f"{method}"],
            config=args,
        )

        evaluate_merged_fts_on_target_data(
            task_vectors, args, f, coeff, pretrained_checkpoint
        )


if __name__ == "__main__":
    main()
