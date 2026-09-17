"""Minimal NLP training loop.

The *loop* is separate from src/trainer.py's because vision's batch handling is
hardcoded to a single-tensor `model(images)` call, which does not fit BERT's
(input_ids, attention_mask) signature. The learning-rate schedule is not
separate: both go through src.trainer.build_scheduler, so --warmup_ratio means
the same thing on either side.
"""

import torch

from src.task_spec import TaskSpec
from src.trainer import build_scheduler


def _run_epochs(model, task: TaskSpec, args, compute_loss):
    model.to(args.device).train()
    total_steps = args.epochs * len(task.train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Shared with the vision trainer. This file used to clamp warmup to 10% of
    # the schedule on its own, which the vision side did not do; expressing it
    # as a ratio makes that the defined behaviour for both.
    scheduler = build_scheduler(optimizer, args, total_steps)
    step = 0
    for _ in range(args.epochs):
        for batch in task.train_loader:
            scheduler(step)
            step += 1
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
    return model


def train_classification_task(model, task: TaskSpec, args):
    loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(model, batch):
        logits = model(batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device))
        return loss_fn(logits, batch["labels"].to(args.device))

    return _run_epochs(model, task, args, compute_loss)


def train_seq2seq_task(model, task: TaskSpec, args):
    def compute_loss(model, batch):
        return model(
            input_ids=batch["input_ids"].to(args.device),
            attention_mask=batch["attention_mask"].to(args.device),
            labels=batch["labels"].to(args.device),
        ).loss

    return _run_epochs(model, task, args, compute_loss)


@torch.no_grad()
def classification_correct_and_total(model, eval_loader, device) -> tuple[int, int]:
    """(num_correct, num_examples) — kept separate from accuracy so callers
    merging results across several tasks (e.g. a target-environment mix) can
    sum exact counts instead of averaging already-rounded per-task ratios.
    """
    model.eval()
    correct, total = 0, 0
    for batch in eval_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        pred = logits.argmax(dim=-1).cpu()
        correct += (pred == batch["labels"]).sum().item()
        total += len(pred)
    return correct, total


def classification_accuracy(model, eval_loader, device) -> float:
    correct, total = classification_correct_and_total(model, eval_loader, device)
    return correct / total


@torch.no_grad()
def seq2seq_eval_loss(model, eval_loader, device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in eval_loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        total_loss += out.loss.item()
        n_batches += 1
    return total_loss / n_batches
