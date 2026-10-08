"""StdCL / LSB pipeline: O-LoRA's continual-learning text classification
benchmarks with T5, task-incremental (each task is already its own dataset,
no class-incremental splitting). See src/nlp/long_sequence_benchmark.py for
the benchmarks, the prompt, and the data.

Two fine-tuning modes (--finetune_mode):

    full  every parameter but the shared embedding is trained (t5-base); a
          whole checkpoint is saved per task, as vision does.
    lora  a LoRA adapter per task, merged into the weights before the next
          task starts (t5-large); only adapters are saved, and each task's
          model — hence its task vector — is rebuilt from them here.

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

import torch
from torch.utils.data import DataLoader

from src.config import get_zeroshot_checkpoint
from src.merging.registry import apply_merge, get_merge_spec
from src.merging.task_vector import TaskVector
from src.merging.task_vectors import mask_and_merge_by_weights
from src.nlp.finetune_nlp import finetune_task_sequence
from src.nlp.long_sequence_benchmark import EVAL_BATCH_SIZE, build_task_sequence
from src.nlp.lora import LoraConfig, adapters_to_vector
from src.nlp.modeling_nlp import build_t5, freeze_shared_embeddings
from src.nlp.target_data import build_target_weights, sample_target_eval_subset
from src.nlp.trainer_nlp import generation_correct_and_total, train_seq2seq_task
from src.paths import adapter_path, checkpoint_dir, finetuned_path
from src.target_env import load_target_envs, select_target_tasks
from src.utils import derive_seed, has_evaluation_result, torch_load

logger = getLogger(__name__)


def _ckpt_dir(args):
    # A LoRA run and a full run of the same model/order/seed must not share a
    # directory: they write different files, and a merge would read whichever
    # kind it expects.
    scope = (
        args.dataset
        if args.finetune_mode == "full"
        else f"{args.dataset}-{args.finetune_mode}"
    )
    return checkpoint_dir(args, group="nlp_classification", scope=scope)


def _build_tasks(args):
    return build_task_sequence(
        args.dataset,
        args.taskseq_pattern,
        tokenizer_name=args.model,
        batch_size=args.batch_size,
    )


def _lora_config(args):
    return LoraConfig() if args.finetune_mode == "lora" else None


def _tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(args.model)


def finetune(args):
    finetune_task_sequence(
        args,
        _build_tasks(args),
        _ckpt_dir(args),
        build_base_model=build_t5,
        train_task=train_seq2seq_task,
        # No per-task head (every task generates its label as text); the
        # shared embedding is frozen as in O-LoRA. Under LoRA every base
        # parameter is frozen anyway.
        prepare_model=lambda model, task: freeze_shared_embeddings(model),
        lora=_lora_config(args),
    )


def _load_task_vectors(args, ckpt_dir, zeroshot_path, n_tasks):
    """One TaskVector per task: the model after task i minus the zero-shot
    model. Under LoRA that is the running sum of adapters 0..i (or adapter i
    alone without --sequential-finetuning), over the adapted weights only."""
    if args.finetune_mode == "full":
        return [
            TaskVector(zeroshot_path, finetuned_path(ckpt_dir, i))
            for i in range(n_tasks)
        ]

    adapters = [
        torch.load(adapter_path(ckpt_dir, i), weights_only=False)
        for i in range(n_tasks)
    ]
    if args.sequential_finetuning:
        return [
            TaskVector(vector=adapters_to_vector(adapters[: i + 1]))
            for i in range(n_tasks)
        ]
    return [TaskVector(vector=adapters_to_vector([a])) for a in adapters]


@torch.no_grad()
def _apply_task_vector(merged_tv, zeroshot_path, scaling_coef, device):
    """zero-shot + scaling_coef * merged_tv, over the keys merged_tv has.

    TaskVector.apply_to computes the same thing but prints a warning for
    every key the vector lacks — under LoRA that is every parameter but the
    q/v projections — so this backend applies it itself.
    """
    model = torch_load(zeroshot_path)
    state = model.state_dict()
    for key, delta in merged_tv.vector.items():
        state[key] = state[key] + scaling_coef * delta
    model.load_state_dict(state)
    return model.to(device)


def _score(model, task, dataset, tokenizer, args):
    loader = DataLoader(
        dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=task.eval_loader.collate_fn
    )
    return generation_correct_and_total(model, loader, tokenizer, args.device)


def _evaluate_on_target_env(
    merged_model, tasks, task_idx_selected, env, tokenizer, args
):
    """Score merged_model on this one target environment's mixture (env.ratio
    of each selected task's own test split). Returns
    (num_data_each_task, taskwise_accuracies, overall_correct, overall_total)."""
    num_data_each_task, taskwise_accuracies = {}, {}
    overall_correct, overall_total = 0, 0
    for i, r in zip(task_idx_selected, env.ratio):
        task = tasks[i]
        target_subset = sample_target_eval_subset(
            task, round(env.num_target_data * r), env.seed
        )
        correct, total = _score(merged_model, task, target_subset, tokenizer, args)
        num_data_each_task[task.name] = total
        taskwise_accuracies[task.name] = correct / total
        overall_correct += correct
        overall_total += total

    return num_data_each_task, taskwise_accuracies, overall_correct, overall_total


def _evaluate_on_full_testsets(merged_model, tasks, tokenizer, args, out_path):
    """O-LoRA's Average Accuracy: every task's whole test split, then the
    unweighted mean over tasks (paper §4.1.2)."""
    if has_evaluation_result(out_path, key="average_accuracy"):
        logger.info(
            f"Result file {out_path} already exists. Skipping full-test-set evaluation."
        )
        return
    taskwise = {}
    for task in tasks:
        correct, total = _score(
            merged_model, task, task.eval_loader.dataset, tokenizer, args
        )
        taskwise[task.name] = correct / total
        logger.info(
            f"{args.merge_fn} {task.name}: accuracy={taskwise[task.name]:.4f} ({total} examples)"
        )
    result = {
        "taskwise_accuracies": taskwise,
        "average_accuracy": sum(taskwise.values()) / len(taskwise),
    }
    logger.info(f"{args.merge_fn}: average_accuracy={result['average_accuracy']:.4f}")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


def merge_and_evaluate(args):
    tasks = _build_tasks(args)
    ckpt_dir = _ckpt_dir(args)
    zeroshot_path = get_zeroshot_checkpoint(args.model)
    task_vectors = _load_task_vectors(args, ckpt_dir, zeroshot_path, len(tasks))
    tokenizer = _tokenizer(args)

    spec = get_merge_spec(args.merge_fn)
    # Every baseline (finetune/random_mix/average/ties/magmax) merges the same
    # way regardless of which target environment it is later scored against,
    # so it is computed once here rather than inside the loop below. Only the
    # proposed method's weights — and therefore its merged vector — depend on
    # the target environment (its preference vector *is* that environment's
    # task-sampling ratio; see src/nlp/target_data.py), so it is left to be
    # recomputed per environment. Merged lazily, so a resumed run whose
    # environments are all finished does no merging at all.
    fixed_merged_model = None

    def _fixed_merged_model():
        nonlocal fixed_merged_model
        if fixed_merged_model is None:
            fixed_merged_model = _apply_task_vector(
                apply_merge(spec, task_vectors),
                zeroshot_path,
                args.scaling_coef,
                args.device,
            )
        return fixed_merged_model

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

        task_idx_selected = select_target_tasks(
            len(tasks), env.n_tasks_fetched, env.seed
        )

        if spec.needs_target_data:
            weights_each_task = build_target_weights(
                len(tasks), task_idx_selected, env.ratio
            )  # weight is obviously just the ratio of each task in the target enironment which is given in task incremental learning
            merged_tv, num_unaligned, num_params_all = mask_and_merge_by_weights(
                task_vectors,
                weights_each_task,
                seed=derive_seed(args.seed, env.target_id),
            )
            merged_model = _apply_task_vector(
                merged_tv, zeroshot_path, args.scaling_coef, args.device
            )
        else:
            merged_model = _fixed_merged_model()

        # Exact micro-average: sum correct / sum total over the actual
        # sampled examples, not a mean of per-task ratios/accuracies.
        num_data_each_task, taskwise_accuracies, overall_correct, overall_total = (
            _evaluate_on_target_env(
                merged_model, tasks, task_idx_selected, env, tokenizer, args
            )
        )
        overall_accuracy = overall_correct / overall_total
        logger.info(
            f"{args.merge_fn} target {env.target_id}: overall_accuracy={overall_accuracy:.4f}"
        )

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

    # The paper's own metric, for the methods that yield one model. The
    # proposed method's model depends on the target environment, so it has
    # no single Average Accuracy (its all-tasks-uniform environment is the
    # closest counterpart).
    if args.eval_full_testsets and not spec.needs_target_data:
        _evaluate_on_full_testsets(
            _fixed_merged_model(),
            tasks,
            tokenizer,
            args,
            os.path.join(out_dir, f"full_testsets_seed{args.seed}.json"),
        )

    logger.info(f"Saved {args.merge_fn} results to {out_dir}")
    return results_by_target
