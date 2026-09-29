"""Long Sequence Benchmark (LSB) pipeline: BertClassifier, task-incremental
(each task is already its own dataset, no class-incremental splitting).

Mirrors vision_backend.py's two-stage structure (finetune saves per-task
checkpoints; merge_and_evaluate reloads them). merge_and_evaluate also mirrors
vision's src/eval.py::evaluate_merged_fts_on_target_data: every --merge_fn is
evaluated against every target environment in --target_config, branching on
MergeSpec.needs_target_data (src/merging/registry.py) exactly as vision does.
The one real difference is *why* the target-data branch needs no similarity
estimate here — see src/nlp/target_data.py's module docstring — so this file
skips vision's train_subset_each_task/similarity_metric machinery entirely.
"""

import json
import os
from logging import getLogger

from torch.utils.data import DataLoader

from src.config import get_zeroshot_checkpoint
from src.merging.registry import apply_merge, get_merge_spec
from src.merging.task_vector import TaskVector
from src.merging.task_vectors import mask_and_merge_by_weights
from src.nlp.finetune_nlp import finetune_task_sequence
from src.nlp.long_sequence_benchmark import TASK_ORDER_PATTERNS, build_lsb_task_sequence
from src.nlp.modeling_nlp import BertClassifier
from src.nlp.target_data import build_target_weights, sample_target_eval_subset
from src.nlp.trainer_nlp import classification_correct_and_total, train_classification_task
from src.paths import checkpoint_dir, finetuned_path
from src.target_env import load_target_envs, select_target_tasks
from src.utils import derive_seed, has_evaluation_result, torch_load

logger = getLogger(__name__)


def _ckpt_dir(args):
    return checkpoint_dir(args, group="nlp_classification", scope=args.dataset)


def _build_tasks(args):
    return build_lsb_task_sequence(TASK_ORDER_PATTERNS[args.taskseq_pattern], tokenizer_name=args.model)


def finetune(args):
    finetune_task_sequence(
        args,
        _build_tasks(args),
        _ckpt_dir(args),
        build_base_model=BertClassifier,
        train_task=train_classification_task,
        # Each LSB task has its own label space, so the shared encoder gets a
        # freshly sized head before every task.
        prepare_model=lambda model, task: model.reset_head(task.num_labels),
    )


def _evaluate_on_target_env(merged_tv, zeroshot_path, tasks, finetuned_paths, task_idx_selected, env, args):
    """Apply merged_tv, then score it on this one target environment's mixture
    (env.ratio of each selected task's own eval split). Returns
    (num_data_each_task, taskwise_accuracies, overall_correct, overall_total)."""
    merged_model = merged_tv.apply_to(zeroshot_path, scaling_coef=0.5).to(args.device)

    num_data_each_task, taskwise_accuracies = {}, {}
    overall_correct, overall_total = 0, 0
    for i, r in zip(task_idx_selected, env.ratio):
        task = tasks[i]
        target_subset = sample_target_eval_subset(task, round(env.num_target_data * r), env.seed)
        # The merged (shared) encoder needs *that task's own trained* head to
        # be evaluated meaningfully — a fresh head would just be random.
        merged_model.head = torch_load(finetuned_paths[i], device=args.device).head
        correct, total = classification_correct_and_total(
            merged_model, DataLoader(target_subset, batch_size=32), args.device
        )
        num_data_each_task[task.name] = total
        taskwise_accuracies[task.name] = correct / total
        overall_correct += correct
        overall_total += total

    return num_data_each_task, taskwise_accuracies, overall_correct, overall_total


def merge_and_evaluate(args):
    tasks = _build_tasks(args)
    ckpt_dir = _ckpt_dir(args)
    zeroshot_path = get_zeroshot_checkpoint(args.model)
    finetuned_paths = [finetuned_path(ckpt_dir, i) for i in range(len(tasks))]
    task_vectors = [TaskVector(zeroshot_path, p) for p in finetuned_paths]

    spec = get_merge_spec(args.merge_fn)
    # Every baseline (finetune/random_mix/average/ties/magmax) merges the same
    # way regardless of which target environment it is later scored against,
    # so it is computed once here rather than inside the loop below. Only the
    # proposed method's weights — and therefore its merged vector — depend on
    # the target environment (its preference vector *is* that environment's
    # task-sampling ratio; see src/nlp/target_data.py), so it is left to be
    # recomputed per environment. Merged lazily, so a resumed run whose
    # environments are all finished does no merging at all.
    fixed_merged_tv = None

    out_dir = os.path.join(ckpt_dir, args.merge_fn)
    os.makedirs(out_dir, exist_ok=True)

    results_by_target = {}
    for env in load_target_envs(args, n_tasks=len(tasks)):
        # Same skip-if-finished rule as vision's src/eval.py, so an interrupted
        # run resumes instead of re-evaluating every environment.
        out_path = os.path.join(out_dir, f"target{env.target_id}_seed{args.seed}.json")
        if has_evaluation_result(out_path):
            logger.info(f"Result file {out_path} already exists. Skipping evaluation.")
            with open(out_path) as f:
                results_by_target[env.target_id] = json.load(f)
            continue

        task_idx_selected = select_target_tasks(len(tasks), env.n_tasks_fetched, env.seed)

        if spec.needs_target_data:
            weights_each_task = build_target_weights(len(tasks), task_idx_selected, env.ratio)
            merged_tv, num_unaligned, num_params_all = mask_and_merge_by_weights(
                task_vectors, weights_each_task, seed=derive_seed(args.seed, env.target_id)
            )
        else:
            if fixed_merged_tv is None:
                fixed_merged_tv = apply_merge(spec, task_vectors)
            merged_tv = fixed_merged_tv

        # Exact micro-average: sum correct / sum total over the actual
        # sampled examples, not a mean of per-task ratios/accuracies.
        num_data_each_task, taskwise_accuracies, overall_correct, overall_total = _evaluate_on_target_env(
            merged_tv, zeroshot_path, tasks, finetuned_paths, task_idx_selected, env, args
        )
        overall_accuracy = overall_correct / overall_total
        logger.info(f"{args.merge_fn} target {env.target_id}: overall_accuracy={overall_accuracy:.4f}")

        result = {
            "target_id": env.target_id,
            "num_task_to_be_fetched": env.n_tasks_fetched,
            "ratio_task_to_be_fetched": env.ratio,
            "task_idx_selected": task_idx_selected,
            "num_data_each_task": num_data_each_task,
            "taskwise_accuracies": taskwise_accuracies,
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "overall_accuracy": overall_accuracy,
        }
        if spec.needs_target_data:
            # Only the proposed method has a preference vector to report and
            # elements that missed their task's budget (Algorithm 1's random
            # leftover assignment) to account for.
            result["weights_each_task"] = weights_each_task
            result["num_unaligned"] = num_unaligned
            result["num_params_all"] = num_params_all

        results_by_target[env.target_id] = result
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    logger.info(f"Saved {args.merge_fn} results to {out_dir}")
    return results_by_target
