"""CLIP/ViT class-incremental pipeline.

This is the original finetune_splitted.py + merge_for_targetdata.py content,
moved here unchanged so that existing scripts/*.sh invocations (dataset=
CIFAR100/ImageNetR) behave identically to before this refactor. Only the
outer `args = parse_arguments()` / `setup_logging(...)` calls were removed
from each entry point, since the new top-level finetune_splitted.py /
merge_for_targetdata.py dispatchers now do that once before resolving which
backend to call.
"""

import os
from logging import getLogger

import torch
import wandb
from src.cl_utils import get_dataset_and_classifier_for_split
from src.config import BASE_DIR, get_zeroshot_checkpoint
from src.datasets.common import get_dataloader
from src.datasets.registry import get_dataset
from src.eval import evaluate_merged_fts_on_target_data
from src.merging.task_vector import TaskVector
from src.merging.task_vectors import (
    finetune as merge_fn_finetune,
    merge_max_abs,
    merge_max_abs_masked_with_targetdata,
    merge_rnd_mix,
    select_one_task_vector,
    ties,
)
from src.modeling import ImageEncoder
from src.trainer import (
    build_loss_fn,
    build_optimizer_and_scheduler,
    run_training_epoch,
    setup_model_for_training,
)

logger = getLogger(__name__)


def _run_sequential_finetuning(args):
    train_dataset = args.dataset
    ckpdir = os.path.join(
        args.save,
        f"{train_dataset}-{args.n_splits}",
        f"ft-pattern_{args.taskseq_pattern}-epochs-{args.epochs}-seed:{args.seed}",
    )

    # finetune for each split separately
    for split_idx in range(args.n_splits):
        logger.info(f"\n##### SPLIT {split_idx} #####")
        ft_path = os.path.join(ckpdir, f"finetuned_{split_idx}.pt")
        if os.path.exists(ft_path):
            logger.info(
                f"Skipping finetuning on split {split_idx}, "
                f"ckpt already exists under {ft_path}"
            )
            continue

        assert train_dataset is not None, "Please provide a training dataset."
        if args.load is not None and args.load.endswith("pt"):
            image_encoder = ImageEncoder.load(args.load, keep_lang=True)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f"finetuned_{split_idx - 1}.pt")
            logger.info(f"Loading image encoder from prev task {prev_ckpt=}")
            image_encoder = torch.load(prev_ckpt, weights_only=False)
        else:
            logger.info(f"Building image encoder: {args.model}.")
            image_encoder = ImageEncoder(args, keep_lang=True)

        if split_idx == 0 and not os.path.exists(
            f"{args.save_ssd}/checkpoints/{args.model}/zeroshot.pt"
        ):
            image_encoder.save(f"{args.save_ssd}/checkpoints/{args.model}/zeroshot.pt")

        preprocess_fn = image_encoder.train_preprocess

        dataset = get_dataset(
            train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size,
            args_=args,
        )
        dataset, classification_head = get_dataset_and_classifier_for_split(
            dataset, split_idx, image_encoder, args
        )

        model = setup_model_for_training(
            image_encoder, classification_head, args, freeze_lang=True
        )
        loss_fn = build_loss_fn(args)
        num_batches = len(dataset.train_loader)
        optimizer, scheduler = build_optimizer_and_scheduler(model, args, num_batches)
        params = [p for p in model.parameters() if p.requires_grad]
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None
        )
        n_batches = len(data_loader)

        if args.save is not None:
            os.makedirs(ckpdir, exist_ok=True)

        for epoch in range(args.epochs):
            loss_total = run_training_epoch(
                model, data_loader, optimizer, scheduler, loss_fn, params, epoch, args
            )
            wandb.log(
                {
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/loss": loss_total / n_batches,
                }
            )

        image_encoder = model.module.image_encoder

        if args.save is not None:
            image_encoder.save(ft_path)


def finetune(args):
    args.lr = 1e-5
    args.batch_size = 32

    args.save_ssd = BASE_DIR
    sequential_ft_dir = "sequential_finetuning/" if args.sequential_finetuning else ""
    args.save = f"{args.save_ssd}/checkpoints/{args.model}/{sequential_ft_dir}{args.split_strategy}_incremental"

    wandb.init(
        project="magmax",
        group=f"{args.dataset}-{args.n_splits}"
        if args.split_strategy == "class"
        else f"{args.dataset}-dil",
        entity=args.wandb_entity_name,
        name=f"{args.dataset}-{args.n_splits}-pattern:{args.taskseq_pattern}-seed:{args.seed}",
        config=args,
        reinit="create_new",
        tags=[
            "ft",
            "CIL",
            f"{args.dataset}",
            f"{args.split_strategy}",
            f"{args.n_splits}",
        ],
    )

    _run_sequential_finetuning(args)


def merge_and_evaluate(args):
    pretrained_checkpoint = get_zeroshot_checkpoint(args.model)

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
        "finetune": (merge_fn_finetune, [0.5]),  # , 1.0
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
            name=f"{name}-{args.taskseq_pattern}-{args.merge_fn}_lambda{args.coeff}_{suffix}_seed{args.seed}",
            tags=["merging-target", "CIL", f"{args.dataset}", f"{method}"],
            config=args,
        )

        evaluate_merged_fts_on_target_data(
            task_vectors, args, f, coeff, pretrained_checkpoint
        )
