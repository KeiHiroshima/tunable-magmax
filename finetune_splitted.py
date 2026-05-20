import os
import warnings
from logging import getLogger

import torch
import wandb
from src.args import parse_arguments
from src.cl_utils import get_dataset_and_classifier_for_split
from src.config import BASE_DIR
from src.datasets.common import get_dataloader
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder
from src.trainer import (
    build_loss_fn,
    build_optimizer_and_scheduler,
    run_training_epoch,
    setup_model_for_training,
)
from src.utils import setup_logging

warnings.simplefilter("ignore")

logger = getLogger(__name__)


def finetune(args):
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
        if os.path.exists(os.path.join(ckpdir, f"finetuned_{split_idx}.pt")):
            logger.info(
                f"Skipping finetuning on split {split_idx}, "
                f"ckpt already exists under {os.path.join(ckpdir, f'finetuned_{split_idx}.pt')}"
            )
            continue

        assert train_dataset is not None, "Please provide a training dataset."
        if args.load is not None and args.load.endswith("pt"):
            image_encoder = ImageEncoder.load(args.load, keep_lang=True)
        elif args.sequential_finetuning and split_idx != 0:
            prev_ckpt = os.path.join(ckpdir, f"finetuned_{split_idx - 1}.pt")
            logger.info(f"Loading image encoder from prev task {prev_ckpt=}")
            image_encoder = torch.load(prev_ckpt)
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


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logging(level=args.logger_mode)

    args.lr = 1e-5
    args.batch_size = 128

    args.save_ssd = BASE_DIR
    sequential_ft_dir = "sequential_finetuning/" if args.sequential_finetuning else ""
    args.save = f"{args.save_ssd}/checkpoints/{args.model}/{sequential_ft_dir}{args.split_strategy}_incremental"

    """print("=" * 100)
    print(
        f"Finetuning {args.model} on {args.dataset}-{args.n_splits} (pattern: {args.taskseq_pattern})"
    )
    print("=" * 100)"""

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

    finetune(args)
