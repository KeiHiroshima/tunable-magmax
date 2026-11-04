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
from src.config import get_unpacked_zeroshot_checkpoint, get_zeroshot_checkpoint
from src.datasets.common import get_dataloader
from src.datasets.registry import get_dataset
from src.eval import evaluate_merged_fts_on_target_data
from src.lora import (
    LoraConfig,
    adapter_task_vectors,
    extract_adapter,
    inject_lora,
    merge_adapter_into,
)
from src.merging.task_vector import TaskVector
from src.merging.core_space import CoreSpace
from src.merging.registry import get_merge_spec, result_name
from src.modeling import ImageEncoder, unpack_visual_attention
from src.paths import adapter_path, checkpoint_dir, finetuned_path
from src.trainer import (
    build_loss_fn,
    build_optimizer_and_scheduler,
    run_training_epoch,
    setup_model_for_training,
)
from src.utils import torch_save

logger = getLogger(__name__)


def _lora(args):
    # Callers that build `args` by hand without the field get a full run.
    return getattr(args, "finetune_mode", "full") == "lora"


def _ckpt_dir(args):
    """Where this vision run's per-split checkpoints live. The fine-tuning and
    merging halves both go through here, so they cannot drift apart. A LoRA
    run writes adapters rather than whole checkpoints, so it gets its own
    directory instead of sharing the full run's."""
    scope = f"{args.dataset}-{args.n_splits}"
    return checkpoint_dir(
        args,
        group=f"{args.split_strategy}_incremental",
        scope=f"{scope}-lora" if _lora(args) else scope,
    )


def _ensure_unpacked_zeroshot_checkpoint(args) -> str:
    """The base every LoRA model is built on: zeroshot.pt (created first if
    missing, exactly as a full run would) with its image-tower attention
    unpacked into q/k/v layers. Same weights and outputs as zeroshot.pt."""
    zeroshot_path = get_zeroshot_checkpoint(args.model)
    if not os.path.exists(zeroshot_path):
        ImageEncoder(args, keep_lang=True).save(zeroshot_path)
    unpacked_path = get_unpacked_zeroshot_checkpoint(args.model)
    if not os.path.exists(unpacked_path):
        encoder = torch.load(zeroshot_path, weights_only=False)
        torch_save(unpack_visual_attention(encoder), unpacked_path)
    return unpacked_path


def _log_trainable_parameters(image_encoder):
    """How far LoRA cuts the trained parameters: a full run updates the whole
    image tower (the text tower and the zero-shot heads are frozen)."""
    n_lora = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    n_full = sum(
        p.numel()
        for name, p in image_encoder.model.visual.named_parameters()
        if "lora_" not in name
    )
    logger.info(
        f"Trainable parameters: LoRA {n_lora:,} / full fine-tuning {n_full:,} "
        f"= {100 * n_lora / n_full:.3f}%"
    )


def _run_sequential_finetuning(args):
    train_dataset = args.dataset
    ckpdir = _ckpt_dir(args)
    lora_config = LoraConfig() if _lora(args) else None
    if lora_config is not None:
        assert args.load is None, "--load is not supported with --finetune_mode lora"
        lora_base_path = _ensure_unpacked_zeroshot_checkpoint(args)
    # --finetune_mode lora: every adapter so far, in split order. A skipped
    # split still joins the chain, so the next split continues from it.
    adapters_so_far: list[str] = []

    # finetune for each split separately
    for split_idx in range(args.n_splits):
        logger.info(f"\n##### SPLIT {split_idx} #####")
        ft_path = (
            adapter_path(ckpdir, split_idx)
            if lora_config
            else finetuned_path(ckpdir, split_idx)
        )
        if os.path.exists(ft_path):
            logger.info(
                f"Skipping finetuning on split {split_idx}, "
                f"ckpt already exists under {ft_path}"
            )
            adapters_so_far.append(ft_path)
            continue

        assert train_dataset is not None, "Please provide a training dataset."
        if lora_config is not None:
            # Same chain as src/nlp/finetune_nlp.py: the zero-shot model with
            # every earlier adapter merged in, in order (O-LoRA eq. 9).
            image_encoder = torch.load(lora_base_path, weights_only=False)
            if args.sequential_finetuning:
                for path in adapters_so_far:
                    merge_adapter_into(
                        image_encoder, torch.load(path, weights_only=False)
                    )
        elif args.load is not None and args.load.endswith("pt"):
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
        if lora_config is not None:
            # After the head is built: inject_lora freezes everything but the
            # adapters, and only matches the (unpacked) image tower's q/v.
            inject_lora(image_encoder, lora_config)
            _log_trainable_parameters(image_encoder)

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

        if lora_config is not None:
            torch.save(extract_adapter(image_encoder, lora_config), ft_path)
        else:
            image_encoder.save(ft_path)
        adapters_so_far.append(ft_path)


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
    method = "seq-ft" if args.sequential_finetuning else "ind-ft"
    name = f"merging_target-{args.dataset}-{args.n_splits}-{method}"

    ckpt_dir = _ckpt_dir(args)
    if _lora(args):
        # As in the NLP backend: each split's task vector is rebuilt from the
        # adapters, over the adapted q/v weights only, and applied to the
        # unpacked zero-shot model those weights belong to.
        pretrained_checkpoint = _ensure_unpacked_zeroshot_checkpoint(args)
        adapters = [
            torch.load(adapter_path(ckpt_dir, _idx), weights_only=False)
            for _idx in range(args.n_splits)
        ]
        task_vectors = [
            TaskVector(vector=v)
            for v in adapter_task_vectors(adapters, args.sequential_finetuning)
        ]
    else:
        pretrained_checkpoint = get_zeroshot_checkpoint(args.model)
        task_vectors = [
            TaskVector(pretrained_checkpoint, finetuned_path(ckpt_dir, _idx))
            for _idx in range(args.n_splits)
        ]

    # --merge_space core (LoRA only; enforced by src/args.py): the bases come
    # from the adapters, once per run — they do not depend on the target.
    core_space = (
        CoreSpace.from_adapters(adapters) if args.merge_space == "core" else None
    )

    spec = get_merge_spec(args.merge_fn)
    # Every merge method was registered with the same single coefficient; the
    # loop is kept so a sweep can be reinstated by extending this list.
    # TODO(--merge_space core): 0.5 was chosen for full-space merges. Core
    # Space packs the same norm into far fewer entries, so a max/sign-based
    # merge there can come out at a different scale; the coefficient may
    # need retuning for core merges. Not done yet.
    coeffs = [0.5]

    for coeff in coeffs:
        args.coeff = coeff

        wandb.init(
            project="magmax",
            group="merging-CIL-target",
            entity=args.wandb_entity_name,
            name=f"{name}-{args.taskseq_pattern}-{result_name(args.merge_fn, args.merge_space)}_lambda{args.coeff}_seed{args.seed}",
            tags=["merging-target", "CIL", f"{args.dataset}", f"{method}"],
            config=args,
        )

        evaluate_merged_fts_on_target_data(
            task_vectors, args, spec, coeff, pretrained_checkpoint, core_space=core_space
        )
