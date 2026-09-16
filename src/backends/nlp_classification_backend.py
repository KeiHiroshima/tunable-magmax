"""Long Sequence Benchmark (LSB) pipeline: BertClassifier, task-incremental
(each task is already its own dataset, no class-incremental splitting).

Mirrors vision_backend.py's two-stage structure (finetune saves per-task
checkpoints; merge_and_evaluate reloads them) but is otherwise independent —
see the "interfaces are intentionally not unified across backends" note in
src/task_spec.py's usage.
"""

import json
import os
from logging import getLogger

from src.config import BASE_DIR, get_zeroshot_checkpoint
from src.merging.registry import merge_task_vectors
from src.merging.task_vector import TaskVector
from src.nlp.long_sequence_benchmark import TASK_ORDER_PATTERNS, build_lsb_task_sequence
from src.nlp.modeling_nlp import BertClassifier
from src.nlp.trainer_nlp import classification_accuracy, train_classification_task
from src.utils import torch_load, torch_save

logger = getLogger(__name__)


def _ckpt_dir(args):
    seq_dir = "sequential_finetuning/" if args.sequential_finetuning else ""
    return os.path.join(
        BASE_DIR,
        "checkpoints",
        args.model,
        seq_dir,
        "nlp_classification",
        args.dataset,
        f"ft-pattern_{args.taskseq_pattern}-epochs-{args.epochs}-seed:{args.seed}",
    )


def _build_tasks(args):
    return build_lsb_task_sequence(TASK_ORDER_PATTERNS[args.taskseq_pattern], tokenizer_name=args.model)


def finetune(args):
    tasks = _build_tasks(args)
    ckpt_dir = _ckpt_dir(args)
    os.makedirs(ckpt_dir, exist_ok=True)

    zeroshot_path = get_zeroshot_checkpoint(args.model)
    if not os.path.exists(zeroshot_path):
        os.makedirs(os.path.dirname(zeroshot_path), exist_ok=True)
        torch_save(BertClassifier(args.model), zeroshot_path)

    prev_ckpt = zeroshot_path
    for idx, task in enumerate(tasks):
        logger.info(f"\n##### TASK {idx}: {task.name} #####")
        ft_path = os.path.join(ckpt_dir, f"finetuned_{idx}.pt")
        if os.path.exists(ft_path):
            logger.info(f"Skipping finetuning on task {task.name}, ckpt already exists under {ft_path}")
            prev_ckpt = ft_path
            continue

        # --sequential-finetuning has the same meaning as in vision_backend:
        # continue from the previous task's weights, vs. always restart from
        # the pretrained base (independent per-task finetuning).
        load_from = prev_ckpt if args.sequential_finetuning else zeroshot_path
        model = torch_load(load_from, device=args.device)
        model.reset_head(task.num_labels)
        train_classification_task(model, task, args)

        torch_save(model, ft_path)
        prev_ckpt = ft_path


def merge_and_evaluate(args):
    tasks = _build_tasks(args)
    ckpt_dir = _ckpt_dir(args)
    zeroshot_path = get_zeroshot_checkpoint(args.model)
    finetuned_paths = [os.path.join(ckpt_dir, f"finetuned_{i}.pt") for i in range(len(tasks))]

    task_vectors = [TaskVector(zeroshot_path, p) for p in finetuned_paths]
    merged_tv = merge_task_vectors(args.merge_fn, task_vectors)
    merged_model = merged_tv.apply_to(zeroshot_path, scaling_coef=0.5).to(args.device)

    results = {}
    for idx, task in enumerate(tasks):
        # The merged (shared) encoder needs *that task's own trained* head to
        # be evaluated meaningfully — a fresh head would just be random.
        merged_model.head = torch_load(finetuned_paths[idx], device=args.device).head
        acc = classification_accuracy(merged_model, task.eval_loader, args.device)
        results[task.name] = acc
        logger.info(f"{task.name}: merged accuracy = {acc:.4f}")

    out_path = os.path.join(ckpt_dir, f"merge_{args.merge_fn}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved merge results to {out_path}")
    return results
