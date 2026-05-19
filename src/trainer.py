import sys

import torch

from src.datasets.common import maybe_dictionarize
from src.modeling import ImageClassifier
from src.utils import LabelSmoothing, cosine_lr


def setup_model_for_training(image_encoder, classification_head, args, freeze_lang=False):
    """Create ImageClassifier, freeze head (and optionally lang), wrap in DataParallel."""
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    if freeze_lang:
        model.freeze_lang()
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    return model


def build_optimizer_and_scheduler(model, args, num_batches):
    """Return (AdamW optimizer, cosine scheduler) for all trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    return optimizer, scheduler


def build_loss_fn(args):
    """Return LabelSmoothing when args.ls > 0, otherwise CrossEntropyLoss."""
    if args.ls > 0:
        return LabelSmoothing(args.ls)
    return torch.nn.CrossEntropyLoss()


def get_batch_inputs(batch, device):
    """Extract (inputs, labels) from a dictionarized batch, handling both key variants."""
    if "images" in batch:
        return batch["images"].to(device), batch["labels"].to(device)
    return batch["image"].to(device), batch["label"].to(device)


def run_training_epoch(model, data_loader, optimizer, scheduler, loss_fn, params, epoch, args):
    """Run one epoch of standard classification training. Returns total loss."""
    model.cuda()
    model.train()
    num_batches = len(data_loader)
    loss_total = 0.0
    for i, batch in enumerate(data_loader):
        step = i + epoch * num_batches
        scheduler(step)
        optimizer.zero_grad()

        batch = maybe_dictionarize(batch)
        inputs, labels = get_batch_inputs(batch, "cuda:0")

        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss_total += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        sys.stdout.write(
            f"\rEpoch {epoch}/{args.epochs} [{i}/{num_batches}]\tLoss: {loss.item():.6f}"
        )
        sys.stdout.flush()

    return loss_total
