import json
import os
from logging import getLogger

import torch
import tqdm

from src.config import get_zeroshot_checkpoint
from src.datasets.common import (
    construct_target_dataset,
    maybe_dictionarize,
)
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.trainer import get_batch_inputs

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

            acc_list.append(
                utils.do_eval(
                    model,
                    dataloader,
                    args.device,
                    flag_data_parallel=flag_data_parallel,
                )["top1"]
            )

        print(f"Task {task_idx} Accuracy: {acc_list[-1]:.4f}")

    avg_acc = sum(acc_list) / (len(acc_list) - zero_data_task)
    acc_list.append(avg_acc)
    print(f"Avg Accuracy: {avg_acc:.4f}")

    # overall accuracy with concatenated data
    all_correct, all_n = 0.0, 0.0
    model.eval()
    device_to_pass = "cuda:0" if flag_data_parallel else args.device
    for task_idx, test_data in enumerate(dataset):
        if test_data is None:
            continue
        dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
        )

        for data in tqdm.tqdm(dataloader):
            data = maybe_dictionarize(data)
            x, y = get_batch_inputs(data, device_to_pass)

            logits = utils.get_logits(x, model)

            if flag_data_parallel:
                pred = logits.argmax(dim=1, keepdim=True).to("cpu")
                y = y.to("cpu")
            else:
                pred = logits.argmax(dim=1, keepdim=True).to(device_to_pass)

            all_correct += pred.eq(y.view_as(pred)).sum().item()
            all_n += y.size(0)

            del x, y, logits, pred
            torch.cuda.empty_cache()

    overall_acc = all_correct / all_n
    acc_list.append(overall_acc)

    print(f"Overall Accuracy over all tasks: {overall_acc:.4f}")

    return [float(a) for a in acc_list]


def _compute_similarity_weights(
    task_vectors, train_subset_each_task, target_dataset_meta, merging_f, args
):
    """Call similarity-based merging to obtain (merged_tv, weights_each_task, similarity_metric).

    Returns None when *merging_f* is not ``merge_max_abs_masked_with_targetdata``.
    """
    if merging_f.__name__ != "merge_max_abs_masked_with_targetdata":
        return None
    return merging_f(
        task_vectors,
        train_subset_each_task,
        target_dataset_meta,
        similarity_metric=args.similarity_metric,
        args=args,
    )


def _build_merged_encoder(
    task_vectors,
    merging_f,
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
    if merging_f.__name__ == "select_one_task_vector":
        classification_head = get_classification_head(args, args.dataset)
        merged_tv, task_vector_idx = merging_f(
            classification_head,
            task_vectors,
            target_dataset_meta,
            args.device,
            model_name=args.model,
        )
        log_data_merge_fn = {
            "merging_function": merging_f.__name__,
            "selected_task_vector_idx": task_vector_idx,
            "scaling_coefficient": scaling_coef,
        }
    elif merging_f.__name__ == "merge_max_abs_masked_with_targetdata":
        merged_tv, weights_each_task, similarity_metric = _compute_similarity_weights(
            task_vectors, train_subset_each_task, target_dataset_meta, merging_f, args
        )
        log_data_merge_fn = {
            "merging_function": merging_f.__name__,
            "similarity_metric": similarity_metric,
            "weights_each_task": [float(v) for v in weights_each_task],
            "scaling_coefficient": float(scaling_coef),
        }
    elif merging_f.__name__ == "finetune":
        merged_tv = task_vectors[-1]
        log_data_merge_fn = {
            "merging_function": "finetune_only",
            "scaling_coefficient": float(scaling_coef),
        }
    else:
        merged_tv = merging_f(task_vectors)
        log_data_merge_fn = {
            "merging_function": merging_f.__name__,
            "scaling_coefficient": scaling_coef,
        }

    if args.merge_fn == "average":
        merged_tv /= len(task_vectors)

    image_encoder = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    return image_encoder, log_data_merge_fn


def _save_target_eval_results(log_data, merging_f, args, suffix_dir, file_name):
    """Persist evaluation results as JSON, merging with any pre-existing content."""
    out_dir = f"{args.results_db}/{merging_f.__name__}/{suffix_dir}"
    os.makedirs(out_dir, exist_ok=True)
    json_path = f"{out_dir}{file_name}"
    existing = json.loads(open(json_path).read()) if os.path.exists(json_path) else {}
    log_data.update(existing)
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)
    logger.info(f"Target data evaluation results saved to {json_path}")


def evaluate_merged_fts_on_target_data(
    task_vectors, args, merging_f, scaling_coef, pretrained_checkpoint=None
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

    train_subset_each_task = dataset._construct_train_subset_each_task(
        n_splits=args.n_splits,
        num_train_data_each_task=args.num_train_data_each_task,
        seed=args.seed,
    )

    with open(os.path.abspath("configs", f"{args.target_config}.json"), "r") as f:
        target_data_configs = json.load(f)["dataset_configs"]

    is_similarity_merge = merging_f.__name__ == "merge_max_abs_masked_with_targetdata"

    for config in target_data_configs:
        num_task_to_be_fetched = (
            args.n_splits
            if config["num_task_to_be_fetched"] < 0
            else config["num_task_to_be_fetched"]
        )
        ratio_task_to_be_fetched = (
            [1.0 / num_task_to_be_fetched for _ in range(num_task_to_be_fetched)]
            if config["ratio_task_to_be_fetched"][0] == -1
            else config["ratio_task_to_be_fetched"]
        )
        args.num_target_data = (
            config["num_target_data"]
            if args.target_config == f"target_data_config_split{args.n_splits}"
            else args.num_target_data
        )

        for variant in config["variants"]:
            target_id = variant["target_id"]
            args.target_id = target_id
            random_seed_for_target = variant["random_seed"]

            suffix = f"{args.similarity_metric}_" if is_similarity_merge else ""
            suffix_dir = f"{args.similarity_metric}/" if is_similarity_merge else ""
            file_name = (
                f"{merging_f.__name__}_lambda{scaling_coef}_{suffix}"
                f"target{target_id}_seed{args.seed}.json"
            )

            if os.path.exists(
                f"{args.results_db}/{merging_f.__name__}/{suffix_dir}{file_name}"
            ):
                logger.info(
                    f"Result file {file_name} already exists in {args.results_db}. Skipping evaluation."
                )
                continue
            elif args.n_splits == 50 and target_id == 26:
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
                dataset_name=args.dataset,
                dataset=dataset,
                n_splits=args.n_splits,
                num_data=args.num_target_data,
                ratio_data_from_task=ratio_task_to_be_fetched,
                seed=random_seed_for_target,
            )
            logger.info(
                f"Target_id {target_id}: {num_task_to_be_fetched} tasks are fetched, {ratio_task_to_be_fetched}"
            )

            image_encoder, log_data_merge_fn = _build_merged_encoder(
                task_vectors,
                merging_f,
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
                f"{merging_f.__name__}_lambda{args.coeff}_{actual_suffix}"
                f"target{args.target_id}_seed{args.seed}.json"
            )

            log_data.update(log_data_merge_fn)
            log_data.update(
                {
                    "target_dataset_info": {
                        "target_id": int(target_id),
                        "seed_target_data": int(random_seed_for_target),
                        "num_task_to_be_fetched": int(num_task_to_be_fetched),
                        "ratio_task_to_be_fetched": [
                            float(v) for v in ratio_task_to_be_fetched
                        ],
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

            _save_target_eval_results(log_data, merging_f, args, suffix_dir, file_name)
