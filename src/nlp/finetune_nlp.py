"""The task-incremental fine-tuning loop shared by the NLP backends.

The two NLP pipelines (LSB classification, CITB/SuperNI seq2seq) ran the same
22-line loop, identical to the line except for which base model to build, which
training function to call, and whether a fresh classification head is needed
per task. Keeping the loop in one place means the invariants below are
implemented — and tested — once instead of twice.

Loop invariants (see test/characterization/test_finetune_loop.py):

  I1  Task `idx`'s checkpoint goes to finetuned_path(ckpt_dir, idx).
  I2  An existing checkpoint is never retrained or overwritten, so an
      interrupted run can be resumed by re-issuing the same command.
  I3  Under --sequential-finetuning a task continues from the *immediately
      preceding* task's checkpoint, and that holds across skipped tasks too.
      This is the easy one to get wrong: drop the bookkeeping and a resumed run
      silently continues from a stale model, with no error to notice.
  I4  The first task starts from the zero-shot checkpoint.
  I5  Tasks are processed in the order the benchmark defines.
"""

import os
from logging import getLogger
from typing import Any, Callable, Optional

from src.config import get_zeroshot_checkpoint
from src.paths import finetuned_path
from src.task_spec import TaskSpec
from src.utils import torch_load, torch_save

logger = getLogger(__name__)


def _ensure_zeroshot_checkpoint(args, build_base_model: Callable[[str], Any]) -> str:
    """Return the zero-shot checkpoint path, creating it on first use.

    Unlike the vision pipeline — where the pre-trained CLIP weights come from
    open_clip — a BERT-based model has no shared image-text head to build from,
    so "zero-shot" here is just the pre-trained encoder with a fresh head. It is
    materialised once so every task vector is taken against the same base.
    """
    zeroshot_path = get_zeroshot_checkpoint(args.model)
    if not os.path.exists(zeroshot_path):
        os.makedirs(os.path.dirname(zeroshot_path), exist_ok=True)
        torch_save(build_base_model(args.model), zeroshot_path)
    return zeroshot_path


def finetune_task_sequence(
    args,
    tasks: list[TaskSpec],
    ckpt_dir: str,
    *,
    build_base_model: Callable[[str], Any],
    train_task: Callable[[Any, TaskSpec, Any], Any],
    prepare_model: Optional[Callable[[Any, TaskSpec], None]] = None,
) -> None:
    """Fine-tune one model across `tasks` in order, saving one checkpoint each.

    Args:
        tasks: the benchmark's task sequence, already ordered.
        ckpt_dir: where this run's checkpoints go (see src/paths.py).
        build_base_model: model_name -> an untrained model, used only to
            create the zero-shot checkpoint the first time it is needed.
        train_task: (model, task, args) -> None. Trains in place.
        prepare_model: (model, task) -> None, called after loading and before
            training. Classification uses it to swap in a head sized for this
            task; seq2seq has no per-task head and passes None.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    zeroshot_path = _ensure_zeroshot_checkpoint(args, build_base_model)

    # None until the first task's checkpoint exists (I4).
    prev_ckpt: Optional[str] = None

    for idx, task in enumerate(tasks):  # I5
        logger.info(f"\n##### TASK {idx}: {task.name} #####")
        ft_path = finetuned_path(ckpt_dir, idx)  # I1

        if os.path.exists(ft_path):
            logger.info(
                f"Skipping finetuning on task {task.name}, "
                f"ckpt already exists under {ft_path}"
            )
            # Still advances the chain, so the next task continues from this
            # checkpoint rather than from whatever preceded it (I3).
            prev_ckpt = ft_path
            continue  # I2

        # --sequential-finetuning has the same meaning as in vision_backend:
        # continue from the previous task's weights, vs. always restart from
        # the pretrained base (independent per-task finetuning).
        load_from = (
            prev_ckpt if args.sequential_finetuning and prev_ckpt else zeroshot_path
        )
        model = torch_load(load_from, device=args.device)
        if prepare_model is not None:
            prepare_model(model, task)
        train_task(model, task, args)

        torch_save(model, ft_path)
        prev_ckpt = ft_path
