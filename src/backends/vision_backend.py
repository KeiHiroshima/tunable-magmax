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
from src.config import get_zeroshot_checkpoint
from src.datasets.common import get_dataloader
from src.datasets.registry import get_dataset
from src.eval import evaluate_merged_fts_on_target_data
from src.merging.task_vector import TaskVector
from src.merging.registry import get_merge_spec
from src.modeling import ImageEncoder
from src.paths import checkpoint_dir, finetuned_path
from src.trainer import (
    build_loss_fn,
    build_optimizer_and_scheduler,
    run_training_epoch,
    setup_model_for_training,
)

logger = getLogger(__name__)


def _ckpt_dir(args):
    """Where this vision run's per-split checkpoints live. The fine-tuning and
    merging halves both go through here, so they cannot drift apart."""
    return checkpoint_dir(
        args,
        group=f"{args.split_strategy}_incremental",
        scope=f"{args.dataset}-{args.n_splits}",
    )


def _run_sequential_finetuning(args):
    train_dataset = args.dataset
    ckpdir = _ckpt_dir(args)

    # finetune for each split separately
    for split_idx in range(args.n_splits):
        logger.info(f"\n##### SPLIT {split_idx} #####")
        ft_path = finetuned_path(ckpdir, split_idx)
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
            prev_ckpt = finetuned_path(ckpdir, split_idx - 1)
            logger.info(f"Loading image encoder from prev task {prev_ckpt=}")
            image_encoder = torch.load(prev_ckpt, weights_only=False)
        else:
            logger.info(f"Building image encoder: {args.model}.")
            image_encoder = ImageEncoder(args, keep_lang=True)

        zeroshot_path = get_zeroshot_checkpoint(args.model)
        if split_idx == 0 and not os.path.exists(zeroshot_path):
            image_encoder.save(zeroshot_path)

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

        image_encoder.save(ft_path)


def finetune(args):
    # --lr and --batch_size used to be overwritten here with 1e-5 and 32.
    # The learning rate matched the CLI default anyway, but the batch size did
    # not: it silently replaced the documented default of 128 — the value the
    # paper reports fine-tuning with — and made --batch_size do nothing.
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

    method = "seq-ft" if args.sequential_finetuning else "ind-ft"
    name = f"merging_target-{args.dataset}-{args.n_splits}-{method}"

    ckpt_dir = _ckpt_dir(args)
    task_vectors = [
        TaskVector(pretrained_checkpoint, finetuned_path(ckpt_dir, _idx))
        for _idx in range(args.n_splits)
    ]

    spec = get_merge_spec(args.merge_fn)
    # Every merge method was registered with the same single coefficient; the
    # loop is kept so a sweep can be reinstated by extending this list.
    coeffs = [0.5]

    for coeff in coeffs:
        args.coeff = coeff

        wandb.init(
            project="magmax",
            group="merging-CIL-target",
            entity=args.wandb_entity_name,
            name=f"{name}-{args.taskseq_pattern}-{args.merge_fn}_lambda{args.coeff}_seed{args.seed}",
            tags=["merging-target", "CIL", f"{args.dataset}", f"{method}"],
            config=args,
        )

        evaluate_merged_fts_on_target_data(
            task_vectors, args, spec, coeff, pretrained_checkpoint
        )
