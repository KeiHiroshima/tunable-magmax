"""CITB / Super-NaturalInstructions pipeline: BERT2BERT, task-incremental.

Same two-stage structure as nlp_classification_backend.py. No head to swap
between tasks (the LM head is shared), so it's simpler in that respect; the
eval metric is generation loss for now (see src/nlp/citb_superni.py's module
docstring / the earlier plan notes on scoping out exact-match/ROUGE-L).
"""

import json
import os
from logging import getLogger

from src.config import BASE_DIR, get_zeroshot_checkpoint
from src.merging.registry import merge_task_vectors
from src.merging.task_vector import TaskVector
from src.nlp.citb_superni import TASK_ORDER_PATTERNS, build_citb_task_sequence
from src.nlp.modeling_nlp import build_bert2bert
from src.nlp.trainer_nlp import seq2seq_eval_loss, train_seq2seq_task
from src.utils import torch_load, torch_save

logger = getLogger(__name__)


def _ckpt_dir(args):
    seq_dir = "sequential_finetuning/" if args.sequential_finetuning else ""
    return os.path.join(
        BASE_DIR,
        "checkpoints",
        args.model,
        seq_dir,
        "nlp_seq2seq",
        args.dataset,
        f"ft-pattern_{args.taskseq_pattern}-epochs-{args.epochs}-seed:{args.seed}",
    )


def _build_tasks(args):
    task_ids = TASK_ORDER_PATTERNS[args.dataset][args.taskseq_pattern]
    return build_citb_task_sequence(task_ids, tokenizer_name=args.model)


def finetune(args):
    tasks = _build_tasks(args)
    ckpt_dir = _ckpt_dir(args)
    os.makedirs(ckpt_dir, exist_ok=True)

    zeroshot_path = get_zeroshot_checkpoint(args.model)
    if not os.path.exists(zeroshot_path):
        os.makedirs(os.path.dirname(zeroshot_path), exist_ok=True)
        torch_save(build_bert2bert(args.model), zeroshot_path)

    prev_ckpt = zeroshot_path
    for idx, task in enumerate(tasks):
        logger.info(f"\n##### TASK {idx}: {task.name} #####")
        ft_path = os.path.join(ckpt_dir, f"finetuned_{idx}.pt")
        if os.path.exists(ft_path):
            logger.info(f"Skipping finetuning on task {task.name}, ckpt already exists under {ft_path}")
            prev_ckpt = ft_path
            continue

        load_from = prev_ckpt if args.sequential_finetuning else zeroshot_path
        model = torch_load(load_from, device=args.device)
        train_seq2seq_task(model, task, args)

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
    for task in tasks:
        loss = seq2seq_eval_loss(merged_model, task.eval_loader, args.device)
        results[task.name] = loss
        logger.info(f"{task.name}: merged eval loss = {loss:.4f}")

    out_path = os.path.join(ckpt_dir, f"merge_{args.merge_fn}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved merge results to {out_path}")
    return results
