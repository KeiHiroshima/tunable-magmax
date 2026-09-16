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

from torch.utils.data import DataLoader

from src.config import BASE_DIR, get_zeroshot_checkpoint
from src.merging.registry import merge_task_vectors
from src.merging.task_vector import TaskVector
from src.merging.task_vectors import mask_and_merge_by_weights
from src.nlp.long_sequence_benchmark import TASK_ORDER_PATTERNS, build_lsb_task_sequence
from src.nlp.modeling_nlp import BertClassifier
from src.nlp.target_data import (
    build_target_weights,
    resolve_ratio,
    sample_target_eval_subset,
    select_target_tasks,
)
from src.nlp.trainer_nlp import (
    classification_accuracy,
    classification_correct_and_total,
    train_classification_task,
)
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

    if args.merge_fn == "masked_magmax_with_targetdata":
        return _merge_and_evaluate_masked(args, tasks, task_vectors, zeroshot_path, finetuned_paths, ckpt_dir)

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


def _merge_and_evaluate_masked(args, tasks, task_vectors, zeroshot_path, finetuned_paths, ckpt_dir):
    """Tunable MAGMAX for LSB: preference vector = the target environment's
    known task-sampling ratio (see src/nlp/target_data.py's module docstring
    for why no similarity/embedding estimation is needed here).
    """
    with open(f"configs/{args.target_config}.json") as f:
        dataset_configs = json.load(f)["dataset_configs"]

    out_dir = os.path.join(ckpt_dir, "masked_magmax_with_targetdata")
    os.makedirs(out_dir, exist_ok=True)

    results_by_target = {}
    for config in dataset_configs:
        num_to_fetch = config["num_task_to_be_fetched"]
        for variant in config["variants"]:
            target_id = variant["target_id"]
            seed = variant["random_seed"]

            task_idx_selected = select_target_tasks(len(tasks), num_to_fetch, seed)
            ratio = resolve_ratio(config["ratio_task_to_be_fetched"], len(task_idx_selected))
            weights_each_task = build_target_weights(len(tasks), task_idx_selected, ratio)

            merged_tv, num_unaligned, num_params_all = mask_and_merge_by_weights(
                task_vectors, weights_each_task
            )
            merged_model = merged_tv.apply_to(zeroshot_path, scaling_coef=0.5).to(args.device)

            # Exact micro-average: sum correct / sum total over the actual
            # sampled examples, not a mean of per-task ratios/accuracies.
            num_data_each_task, taskwise_accuracies = {}, {}
            overall_correct, overall_total = 0, 0
            for i, r in zip(task_idx_selected, ratio):
                task = tasks[i]
                target_subset = sample_target_eval_subset(
                    task, round(args.num_target_data * r), seed
                )
                merged_model.head = torch_load(finetuned_paths[i], device=args.device).head
                correct, total = classification_correct_and_total(
                    merged_model, DataLoader(target_subset, batch_size=32), args.device
                )
                num_data_each_task[task.name] = total
                taskwise_accuracies[task.name] = correct / total
                overall_correct += correct
                overall_total += total

            overall_accuracy = overall_correct / overall_total
            logger.info(f"target {target_id}: overall_accuracy={overall_accuracy:.4f}")

            results_by_target[target_id] = {
                "target_id": target_id,
                "num_task_to_be_fetched": num_to_fetch,
                "ratio_task_to_be_fetched": ratio,
                "task_idx_selected": task_idx_selected,
                "weights_each_task": weights_each_task,
                "num_data_each_task": num_data_each_task,
                "taskwise_accuracies": taskwise_accuracies,
                "overall_correct": overall_correct,
                "overall_total": overall_total,
                "overall_accuracy": overall_accuracy,
                "num_unaligned": num_unaligned,
                "num_params_all": num_params_all,
            }
            with open(os.path.join(out_dir, f"target{target_id}_seed{args.seed}.json"), "w") as f:
                json.dump(results_by_target[target_id], f, indent=2)

    logger.info(f"Saved masked_magmax_with_targetdata results to {out_dir}")
    return results_by_target
