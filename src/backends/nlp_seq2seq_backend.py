"""CITB / Super-NaturalInstructions pipeline: BERT2BERT, task-incremental.

Same two-stage structure as nlp_classification_backend.py. No head to swap
between tasks (the LM head is shared), so it's simpler in that respect; the
eval metric is generation loss for now (see src/nlp/citb_superni.py's module
docstring / the earlier plan notes on scoping out exact-match/ROUGE-L).
"""

import json
import os
from logging import getLogger

from src.config import get_zeroshot_checkpoint
from src.merging.registry import merge_task_vectors
from src.merging.task_vector import TaskVector
from src.paths import checkpoint_dir, finetuned_path
from src.nlp.citb_superni import TASK_ORDER_PATTERNS, build_citb_task_sequence
from src.nlp.finetune_nlp import finetune_task_sequence
from src.nlp.modeling_nlp import build_bert2bert
from src.nlp.trainer_nlp import seq2seq_eval_loss, train_seq2seq_task

logger = getLogger(__name__)


def _ckpt_dir(args):
    return checkpoint_dir(args, group="nlp_seq2seq", scope=args.dataset)


def _build_tasks(args):
    task_ids = TASK_ORDER_PATTERNS[args.dataset][args.taskseq_pattern]
    return build_citb_task_sequence(task_ids, tokenizer_name=args.model)


def finetune(args):
    finetune_task_sequence(
        args,
        _build_tasks(args),
        _ckpt_dir(args),
        build_base_model=build_bert2bert,
        train_task=train_seq2seq_task,
        # No prepare_model: the LM head is shared across tasks, so there is
        # nothing to swap between them.
    )


def merge_and_evaluate(args):
    tasks = _build_tasks(args)
    ckpt_dir = _ckpt_dir(args)
    zeroshot_path = get_zeroshot_checkpoint(args.model)
    finetuned_paths = [finetuned_path(ckpt_dir, i) for i in range(len(tasks))]

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
