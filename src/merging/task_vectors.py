import json
import random
import warnings
from logging import getLogger
from pathlib import Path

import torch
from src.args import parse_arguments
from src.config import get_zeroshot_checkpoint
from src.datasets.common import get_task_classes
from src.datasets.registry import get_dataset
from src.merging.similarity import (
    FeatureCost,
    compute_cosine_similarity,
    compute_mmd_similarity,
    compute_otdd_similarity,
    count_labels,
    run_optmization,
)
from src.merging.task_vector import TaskVector
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import do_eval, is_freezed_parameter

warnings.simplefilter("ignore")


# Config
args = parse_arguments()
pretrained_checkpoint = get_zeroshot_checkpoint(args.model)
logger = getLogger("root")


def merge_rnd_mix(task_vectors):
    """Randomly mix multiple task vectors together."""
    if len(task_vectors) < 2:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            _rand_indices = torch.randint(
                0, len(task_vectors), task_vectors[0].vector[key].shape
            )
            new_vector[key] = task_vectors[0].vector[key] * (_rand_indices == 0)
            for i in range(1, len(task_vectors)):
                new_vector[key] += task_vectors[i].vector[key] * (_rand_indices == i)

    return TaskVector(vector=new_vector)


def merge_max_abs(task_vectors):
    """Mix multiple task vectors together by highest parameter value."""
    if len(task_vectors) < 2:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}

        # Iterate over keys in the first task vector
        for key in task_vectors[0].vector:
            # Get the initial tensor for the current key
            max_abs_tensor = task_vectors[0].vector[key]

            # Iterate over the remaining task vectors
            for task_vector in task_vectors[1:]:
                current_tensor = task_vector.vector[key]

                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(
                    current_tensor.abs() >= max_abs_tensor.abs(),
                    current_tensor,
                    max_abs_tensor,
                )

            # Assign the final tensor to the new_vector dictionary
            new_vector[key] = max_abs_tensor

    return TaskVector(vector=new_vector)


def merge_max_abs_masked_with_targetdata(
    task_vectors,
    train_subset_each_task,
    target_data,
    similarity_metric="cosine",
    args=None,
):
    # calculate similarity scores between train_data and target_data
    similarity_score_list = [None for _ in range(len(train_subset_each_task))]
    distance_metric = {
        "labels": count_labels,
        "cosine": compute_cosine_similarity,
        "mmd": compute_mmd_similarity,
        "ot": compute_otdd_similarity,
        "hpo": run_optmization,
    }
    similarity_metric_key = (
        similarity_metric.split("_")[0]
        if "_" in similarity_metric
        else similarity_metric
    )

    if similarity_metric == "hpo":
        similarity_score_list = distance_metric[similarity_metric_key](
            task_vectors=task_vectors,
            target_dataset_meta=target_data,
            args=args,
        )
    else:
        if similarity_metric == "labels":
            preprocess_fn = ImageEncoder(args, keep_lang=True).train_preprocess
            class_order = get_dataset(
                args.dataset,
                preprocess_fn,
                location=args.data_location,
                batch_size=args.batch_size,
                args_=args,
            ).default_class_order
            task_class_dict = {
                i: get_task_classes(class_order, args.n_splits, i)
                for i in range(args.n_splits)
            }

        for i, train_subset_onetask in enumerate(train_subset_each_task):
            if similarity_metric == "labels":
                similarity_score = distance_metric[similarity_metric_key](
                    target_data, task_class_dict=task_class_dict, task_idx=i
                )
            else:
                if "embedded" in similarity_metric:
                    logger.debug(
                        f"Using {similarity_metric_key} similarity with embedded data."
                    )
                    encoder = task_vectors[i].apply_to(
                        pretrained_checkpoint, scaling_coef=1.0
                    )
                    feature_cost = FeatureCost(
                        src_embedding=encoder,
                        tgt_embedding=encoder,
                        device=args.device,
                    )
                else:
                    logger.debug(
                        f"Using {similarity_metric_key} similarity with raw data."
                    )
                    feature_cost = None

                similarity_score = distance_metric[similarity_metric_key](
                    train_subset_onetask,
                    target_data,
                    feature_cost=feature_cost,
                    device=args.device,
                )

            similarity_score_list[i] = similarity_score

    # calculate weights based on similarity scores
    total_score = sum(similarity_score_list)
    weights_each_task = [
        similarity_score_list[i] / total_score
        for i in range(len(similarity_score_list))
    ]
    logger.debug(f"total_score: {total_score}")

    logger.debug(f"similarity metric: {similarity_metric}")
    logger.debug(f"similarity_score_list: {similarity_score_list}")
    logger.debug(f"weights_each_task: {weights_each_task}")

    # masked MAGMAX merging with calculated number of elements per task vector
    # 1. calculate number of elements to take from each task vector
    # 2. perform masked MAGMAX merging
    with torch.no_grad():
        new_vector = {}
        num_unaligned_accum_dict = {
            f"task_{i + 1}": 0 for i in range(len(task_vectors))
        }
        num_params_all = sum(
            [task_vectors[0].vector[key].numel() for key in task_vectors[0].vector]
        )

        for _, key in enumerate(task_vectors[0].vector):
            num_elements = task_vectors[0].vector[key].numel()

            if is_freezed_parameter([tv.vector[key] for tv in task_vectors]):
                new_vector[key] = torch.zeros_like(task_vectors[0].vector[key])
                continue

            elements_per_task_list = [
                int(num_elements * weight) for weight in weights_each_task
            ]
            remainder = num_elements - sum(elements_per_task_list)
            for i in range(remainder):
                elements_per_task_list[i % len(elements_per_task_list)] += 1

            # Stack all tensors for the current key
            all_tensors = torch.stack(
                [
                    tv.vector[key]
                    if elements_per_task_list[i] > 0
                    else torch.zeros_like(tv.vector[key])
                    for i, tv in enumerate(task_vectors)
                ]
            )
            if all_tensors.dim() == 1:
                all_tensors = all_tensors[:, None]

            logger.debug(f"{key} elements_per_task_list: {elements_per_task_list}")
            logger.debug(f"Shape of all_tensors: {all_tensors.shape}")

            # Get top absolute values and their corresponding task indices
            _, task_indices = torch.topk(all_tensors.abs(), k=1, dim=0)

            # Initial winners are the tasks with the highest absolute value
            winner_indices = task_indices[0]

            # Iterate downwards from the last task to the second task (task_id=1)
            for i in range(len(task_vectors) - 1, -1, -1):
                # Mask for elements won by the current task
                is_winner = winner_indices == i
                num_won = is_winner.sum().item()
                elements_per_task = elements_per_task_list[i]

                if num_won > elements_per_task:
                    logger.debug(
                        f"Key: {key}, Task {i} won {num_won} elements > {elements_per_task}"
                    )

                    if i > 0:
                        # Indices of elements won by this task
                        won_indices = torch.where(is_winner.flatten())[0]

                        # Randomly choose which ones to drop
                        num_to_drop = num_won - elements_per_task
                        perm = torch.randperm(won_indices.numel())
                        drop_indices_local = perm[:num_to_drop]
                        indices_to_drop = won_indices[drop_indices_local]

                        # Find the best replacement from prior tasks
                        prior_tensors = all_tensors[:i, ...].flatten(start_dim=1)

                        prior_tensors_at_drop = prior_tensors[:, indices_to_drop]

                        # Find new winners from prior tasks
                        _, new_winners_local = torch.max(
                            prior_tensors_at_drop.abs(), dim=0
                        )

                        # Update winner_indices for the dropped positions
                        original_shape_indices = torch.unravel_index(
                            indices_to_drop, all_tensors[0].shape
                        )
                        winner_indices[original_shape_indices] = new_winners_local
                    elif i == 0:
                        logger.debug(
                            "reassign elements in the first task vector to not enough elements from other task vectors"
                        )
                        # Indices of elements won by this task
                        won_indices = torch.where(is_winner.flatten())[0]

                        # Randomly choose which ones to drop
                        num_to_drop = num_won - elements_per_task
                        perm = torch.randperm(won_indices.numel())
                        drop_indices_local = perm[:num_to_drop]
                        indices_to_drop = won_indices[drop_indices_local]

                        # Find the replacement candidates from other tasks
                        prior_tensors = all_tensors[1:, ...].flatten(start_dim=1)
                        prior_tensors_at_drop = prior_tensors[:, indices_to_drop]

                        # Find new winners from other tasks if other task vectors are not taken elements as much as elements_per_task[task_index]
                        num_elements_needed = [
                            elements_per_task_list[j]
                            - (winner_indices == j).sum().item()
                            for j in range(1, len(task_vectors))
                        ]  # length: num_tasks-1

                        new_winners_local = torch.zeros_like(indices_to_drop)
                        pool_selected_indices = set()

                        _, winner_indices_local = torch.max(
                            prior_tensors_at_drop.abs(), dim=0
                        )
                        # winner_indices_local and j index are relative task_id: task2=0,...,taskN=N-2

                        num_actual_aligned = [
                            0 for _ in range(len(num_elements_needed))
                        ]
                        for j, num_needed in zip(
                            range(len(num_elements_needed) - 1, -1, -1),
                            reversed(num_elements_needed),
                        ):
                            if num_needed > 0:
                                candidates_indices = torch.where(
                                    winner_indices_local == j
                                )[0]

                                # check if candidates_indices is in pool_selected_indices
                                candidates_indices = torch.tensor(
                                    [
                                        idx
                                        for idx in candidates_indices.tolist()
                                        if idx not in pool_selected_indices
                                    ]
                                )

                                if len(candidates_indices) > num_needed:
                                    perm_ = torch.randperm(len(candidates_indices))
                                    selected_indices = candidates_indices[
                                        perm_[:num_needed]
                                    ].tolist()

                                    index_drop_local = candidates_indices[
                                        perm_[num_needed:]
                                    ].tolist()
                                    num_actual_aligned[j] = num_needed

                                else:
                                    selected_indices, index_drop_local = (
                                        candidates_indices.tolist(),
                                        candidates_indices.tolist(),
                                    )
                                    num_actual_aligned[j] = len(selected_indices)

                                # update winner_indices_local at [num_needed:]
                                prior_tensors_at_drop_local = prior_tensors_at_drop[
                                    :, index_drop_local
                                ]
                                _, winner_indices_local_local = torch.max(
                                    prior_tensors_at_drop_local.abs(), dim=0
                                )
                                winner_indices_local[index_drop_local] = (
                                    winner_indices_local_local
                                )

                                new_winners_local[selected_indices] = (
                                    j + 1
                                )  # this is tv index, task 2 = 1
                                pool_selected_indices.update(set(selected_indices))

                        indices_unselected = set(
                            list(range(len(indices_to_drop)))
                        ).difference(
                            pool_selected_indices,
                        )
                        for j, (num_aligned, num_needed) in enumerate(
                            zip(num_actual_aligned, num_elements_needed)
                        ):
                            if num_aligned < num_needed:
                                num_unaligned_accum_dict[f"task_{j + 2}"] += (
                                    num_needed - num_aligned
                                )  # for logging
                                selected_indices_random = random.sample(
                                    list(indices_unselected),
                                    k=num_needed - num_aligned,
                                )

                                new_winners_local[selected_indices_random] = (
                                    j + 1
                                )  # this is tv index, task 2 = 1

                                pool_selected_indices.update(
                                    set(selected_indices_random)
                                )
                                # update indices_unselected
                                indices_unselected = indices_unselected.difference(
                                    set(selected_indices_random)
                                )

                            elif num_needed == 0:
                                logger.debug(
                                    f"Task idx {j + 1} doesn't need elements any more"
                                )

                        # Update winner_indices for the dropped positions
                        original_shape_indices = torch.unravel_index(
                            indices_to_drop, all_tensors[0].shape
                        )
                        winner_indices[original_shape_indices] = new_winners_local
                    else:
                        raise ValueError("Unexpected task index")
                else:
                    logger.debug(
                        f"Key: {key}, Task {i} won {num_won} elements <= {elements_per_task}"
                    )

            # Gather the final values from the winning tensors
            merged_tensor = all_tensors.gather(0, winner_indices[None, :]).squeeze(0)

            new_vector[key] = merged_tensor

            winner_indices_distribution = [
                (winner_indices == i).sum().item() for i in range(len(task_vectors))
            ]
            logger.debug(
                f"{key} num elements per task:\n winner_indices_distribution: {winner_indices_distribution}, elements_per_task_list: {elements_per_task_list}"
            )
            assert winner_indices_distribution == elements_per_task_list, (
                f"winner_indices_distribution: {winner_indices_distribution}, elements_per_task_list: {elements_per_task_list}"
            )

    logger.info(f"num_unaligned_accum_dict: {num_unaligned_accum_dict}")

    if args.results_db:
        log_dir = (
            Path(args.results_db)
            / "merge_max_abs_masked_with_targetdata"
            / args.similarity_metric
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (
            log_dir
            / f"merge_max_abs_masked_with_targetdata_lambda{args.coeff}_{similarity_metric}_target{args.target_id}_seed{args.seed}.json"
        )
        existing = json.loads(log_path.read_text()) if log_path.exists() else {}
        existing["num_unaligned"] = num_unaligned_accum_dict
        existing["num_params_all"] = num_params_all
        log_path.write_text(json.dumps(existing, indent=4))
        logger.info(f"Saved num_unaligned log to {log_path}")

    return TaskVector(vector=new_vector), weights_each_task, similarity_metric


def ties(task_vectors):
    if len(task_vectors) < 2:
        return task_vectors[0]

    reset_type = "topk"
    reset_thresh = 20
    resolve = "mass"
    merge = "dis-mean"
    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(tv.vector) for tv in task_vectors]
    )

    print(
        f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}"
    )

    merged_flat_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_tv = vector_to_state_dict(
        merged_flat_tv, task_vectors[0].vector, remove_keys=[]
    )

    return TaskVector(vector=merged_tv)


def select_one_task_vector(
    classification_head,
    task_vectors: list[TaskVector],
    target_data: torch.utils.data.Dataset,
    device: torch.device,
    model_name=None,
):
    if model_name == "ViT-L-14":
        flag_data_parallel = True
        device = list(range(torch.cuda.device_count()))
        print("Using devices", device)
    else:
        flag_data_parallel = False

    selected_tv, task_idx_selected, acc_champ = task_vectors[0], -1, 0.0

    dataloader_target = torch.utils.data.DataLoader(
        target_data, batch_size=128, shuffle=False, num_workers=4
    )
    with torch.no_grad():
        for i, tv in enumerate(task_vectors):
            image_encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)
            model = ImageClassifier(image_encoder, classification_head)

            if flag_data_parallel:
                model = torch.nn.DataParallel(model, device_ids=device)

            acc = do_eval(
                model, dataloader_target, device, flag_data_parallel=flag_data_parallel
            )["top1"]
            logger.debug(f"Task vector {i}: accuracy on target data: {acc:.4f}")

            if acc > acc_champ:
                logger.debug(
                    f"Task vector {i} takes the lead with accuracy: {acc:.4f} (previous best: {acc_champ:.4f})"
                )
                selected_tv = tv
                task_idx_selected = i
                acc_champ = acc

    return selected_tv, task_idx_selected


def finetune():
    logger.debug("Finetune only mode selected.")
    pass
