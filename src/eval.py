import json
import os
from logging import getLogger

import torch
from src.config import get_zeroshot_checkpoint
from src.datasets.common import (
    construct_target_dataset,
    construct_train_subset_each_task,
)
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.merging.registry import apply_merge
from src.modeling import ImageClassifier, ImageEncoder
from src.target_env import load_target_envs

from . import utils

logger = getLogger(__name__)


def eval_given_dataset(image_encoder, dataset, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    if args.model == "ViT-L-14":
        flag_data_parallel = True
        device = list(range(torch.cuda.device_count()))
        print("Using devices", device)
        model = torch.nn.DataParallel(model, device_ids=device)
    else:
        flag_data_parallel = False

    acc_list = []
    zero_data_task = 0
    # Totals for the overall (micro-averaged) accuracy. Accumulating them here
    # replaces a second pass over the same data, which ran the identical
    # inference again purely to count correct predictions.
    all_correct, all_n = 0, 0

    for task_idx, test_data in enumerate(dataset):
        if test_data is None:
            acc_list.append(0.0)
            zero_data_task += 1
        else:
            dataloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=16,
            )

            metrics = utils.do_eval(
                model,
                dataloader,
                args.device,
                flag_data_parallel=flag_data_parallel,
            )
            acc_list.append(metrics["top1"])
            all_correct += metrics["correct"]
            all_n += metrics["n"]

        print(f"Task {task_idx} Accuracy: {acc_list[-1]:.4f}")

    # Average over tasks: every task counts the same regardless of how many
    # examples this target environment draws from it.
    avg_acc = sum(acc_list) / (len(acc_list) - zero_data_task)
    acc_list.append(avg_acc)
    print(f"Avg Accuracy: {avg_acc:.4f}")

    # Overall: every *example* counts the same, so a task the environment draws
    # more heavily from weighs more.
    overall_acc = all_correct / all_n
    acc_list.append(overall_acc)

    print(f"Overall Accuracy over all tasks: {overall_acc:.4f}")

    return [float(a) for a in acc_list]


def _compute_similarity_weights(
    task_vectors, train_subset_each_task, target_dataset_meta, spec, args
):
    """Call a target-data-driven merge, returning
    (merged_tv, weights_each_task, similarity_metric)."""
    assert spec.needs_target_data, f"{spec.name} does not take target data"
    return spec.fn(
        task_vectors,
        train_subset_each_task,
        target_dataset_meta,
        similarity_metric=args.similarity_metric,
        args=args,
    )


def _build_merged_encoder(
    task_vectors,
    spec,
    args,
    pretrained_checkpoint,
    scaling_coef,
    train_subset_each_task=None,
    target_dataset_meta=None,
):
    """Merge task vectors and apply them to the pretrained model.

    Returns:
        image_encoder:     the merged image encoder.
        log_data_merge_fn: dict with merging metadata for JSON logging.
    """
    if spec.needs_target_data:
        merged_tv, weights_each_task, similarity_metric = _compute_similarity_weights(
            task_vectors, train_subset_each_task, target_dataset_meta, spec, args
        )
        log_data_merge_fn = {
            "merging_function": spec.name,
            "similarity_metric": similarity_metric,
            "weights_each_task": [float(v) for v in weights_each_task],
            "scaling_coefficient": float(scaling_coef),
        }
    else:
        # apply_merge owns the "finetune" (no merging) case and the post-merge
        # division for "average", so both backends get identical treatment.
        merged_tv = apply_merge(spec, task_vectors)
        log_data_merge_fn = {
            "merging_function": spec.name,
            "scaling_coefficient": float(scaling_coef),
        }

    image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    return image_encoder, log_data_merge_fn


def _save_target_eval_results(log_data, spec, args, suffix_dir, file_name):
    """Persist evaluation results as JSON, merging with any pre-existing content."""
    out_dir = f"{args.results_db}/{spec.name}/{suffix_dir}"
    os.makedirs(out_dir, exist_ok=True)
    json_path = f"{out_dir}{file_name}"
    existing = json.loads(open(json_path).read()) if os.path.exists(json_path) else {}
    log_data.update(existing)
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)
    logger.info(f"Target data evaluation results saved to {json_path}")


def evaluate_merged_fts_on_target_data(
    task_vectors, args, spec, scaling_coef, pretrained_checkpoint=None
):
    assert args.num_train_data_each_task is not None, (
        "Please provide num_train_data_each_task for constructing train subsets."
    )

    if pretrained_checkpoint is None:
        pretrained_checkpoint = get_zeroshot_checkpoint(args.model)

    preprocess_fn = ImageEncoder(args, keep_lang=True).train_preprocess

    dataset = get_dataset(
        args.dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        args_=args,
    )

    train_subset_each_task = construct_train_subset_each_task(
        dataset,
        n_splits=args.n_splits,
        num_train_data_each_task=args.num_train_data_each_task,
        seed=args.seed,
    )

    is_similarity_merge = spec.needs_target_data

    for env in load_target_envs(args, n_tasks=args.n_splits):
        # merge_max_abs_masked_with_targetdata names its own log file after
        # this, and reads it off args rather than taking it as a parameter.
        args.target_id = env.target_id

        suffix = f"{args.similarity_metric}_" if is_similarity_merge else ""
        suffix_dir = f"{args.similarity_metric}/" if is_similarity_merge else ""
        file_name = (
            f"{spec.name}_lambda{scaling_coef}_{suffix}"
            f"target{env.target_id}_seed{args.seed}.json"
        )

        if os.path.exists(
            f"{args.results_db}/{spec.name}/{suffix_dir}{file_name}"
        ):
            logger.info(
                f"Result file {file_name} already exists in {args.results_db}. Skipping evaluation."
            )
            continue
        elif args.n_splits == 50 and env.target_id == 26:
            # Skip cifar100-50 target data id 26 due to too few data
            logger.info(
                "Skipping evaluation for cifar100-50 target data id 26 due to too few data."
            )
            continue

        (
            task_idx_selected,
            num_data_each_task,
            target_dataset_meta,
            target_dataset_test,
        ) = construct_target_dataset(
            dataset,
            n_splits=args.n_splits,
            num_data=env.num_target_data,
            ratio_data_from_task=env.ratio,
            seed=env.seed,
        )
        logger.info(
            f"Target_id {env.target_id}: {env.n_tasks_fetched} tasks are fetched, {env.ratio}"
        )

        image_encoder, log_data_merge_fn = _build_merged_encoder(
            task_vectors,
            spec,
            args,
            pretrained_checkpoint,
            scaling_coef,
            train_subset_each_task=train_subset_each_task,
            target_dataset_meta=target_dataset_meta,
        )

        acc_list = eval_given_dataset(
            image_encoder, target_dataset_test, args.dataset, args
        )
        acc_list = [float(acc) for acc in acc_list]
        # Logging
        log_data = {
            "overall_accuracy": acc_list[-1],
            "average_accuracy": acc_list[-2],
            "taskwise_accuracies": acc_list[:-2],
        }

        # Use the actual similarity_metric returned by the merge function (may
        # differ from args.similarity_metric if the function overrides it).
        actual_metric = log_data_merge_fn.get(
            "similarity_metric", args.similarity_metric
        )
        actual_suffix = f"{actual_metric}_" if is_similarity_merge else ""
        file_name = (
            f"{spec.name}_lambda{args.coeff}_{actual_suffix}"
            f"target{args.target_id}_seed{args.seed}.json"
        )

        log_data.update(log_data_merge_fn)
        log_data.update(
            {
                "target_dataset_info": {
                    "target_id": int(env.target_id),
                    "seed_target_data": int(env.seed),
                    "num_task_to_be_fetched": int(env.n_tasks_fetched),
                    "ratio_task_to_be_fetched": [float(v) for v in env.ratio],
                    "task_idx_selected": [int(v) for v in task_idx_selected],
                    "num_meta_data": int(len(target_dataset_meta)),
                    "num_data_each_task": [float(n) for n in num_data_each_task],
                    "num_target_data": float(sum(num_data_each_task)),
                },
            }
        )
        log_data.update(
            {
                "train_data_info": f"{[int(len(onetask)) for onetask in train_subset_each_task]}",
                "model": args.model,
            }
        )

        _save_target_eval_results(log_data, spec, args, suffix_dir, file_name)

        del image_encoder
        torch.cuda.empty_cache()
