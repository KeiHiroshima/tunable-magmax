"""Minimal NLP training loop.

The *loop* is separate from src/trainer.py's because vision's batch handling is
hardcoded to a single-tensor `model(images)` call, which does not fit a
seq2seq model's (input_ids, attention_mask, labels) signature. The
learning-rate schedule is not separate: both go through
src.trainer.build_scheduler, so --lr_schedule / --warmup_ratio mean the same
thing on either side.
"""

import math
from logging import getLogger

import torch

from src.nlp.long_sequence_benchmark import MAX_TARGET_LENGTH, normalize_answer
from src.task_spec import TaskSpec
from src.trainer import build_scheduler

logger = getLogger(__name__)

LOG_EVERY_STEPS = 50


def _run_epochs(model, task: TaskSpec, args, compute_loss):
    """Train `model` on `task` in place.

    Only parameters with requires_grad are optimised, so a LoRA-injected
    model (src/nlp/lora.py) trains its adapter alone, and a model with frozen
    embeddings leaves them untouched (AdamW's weight decay would otherwise
    still move them).

    --grad_accum_steps micro-batches make one optimizer step, which is how a
    single GPU reproduces O-LoRA's batch of 64 (8 per GPU x 8 GPUs). The
    schedule counts optimizer steps. A final group shorter than
    --grad_accum_steps still steps, with its loss averaged over its own size.
    With --grad_accum_steps 1 this is the plain one-batch-one-step loop.
    """
    model.to(args.device).train()
    accum = getattr(args, "grad_accum_steps", 1)
    n_batches = len(task.train_loader)
    total_steps = args.epochs * math.ceil(n_batches / accum)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = build_scheduler(optimizer, args, total_steps)

    step = 0
    for _ in range(args.epochs):
        optimizer.zero_grad()
        running = 0.0
        for i, batch in enumerate(task.train_loader):
            group_start = (i // accum) * accum
            group_size = min(accum, n_batches - group_start)
            loss = compute_loss(model, batch) / group_size
            loss.backward()
            running += loss.item()
            if i + 1 == group_start + group_size:
                scheduler(step)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % LOG_EVERY_STEPS == 0 or step == total_steps:
                    logger.info(f"{task.name} step {step}/{total_steps} loss {running:.4f}")
                running = 0.0
    return model


def train_seq2seq_task(model, task: TaskSpec, args):
    def compute_loss(model, batch):
        return model(
            input_ids=batch["input_ids"].to(args.device),
            attention_mask=batch["attention_mask"].to(args.device),
            labels=batch["labels"].to(args.device),
        ).loss

    return _run_epochs(model, task, args, compute_loss)


@torch.no_grad()
def generation_correct_and_total(model, eval_loader, tokenizer, device) -> tuple[int, int]:
    """(num_correct, num_examples) under exact match of the greedily generated
    string against the label string, both normalised as O-LoRA does.

    Kept as counts rather than an accuracy so callers merging results across
    several tasks (a target-environment mix) can sum exact counts instead of
    averaging already-rounded per-task ratios.
    """
    model.eval()
    correct, total = 0, 0
    for batch in eval_loader:
        generated = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=1,
            do_sample=False,
        )
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        correct += sum(
            normalize_answer(p) == normalize_answer(t)
            for p, t in zip(predictions, batch["targets"])
        )
        total += len(predictions)
    return correct, total


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
