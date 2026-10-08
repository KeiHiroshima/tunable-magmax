import collections
import glob
import math
import os
from logging import getLogger
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.target_env import select_target_tasks

logger = getLogger(__name__)


def get_balanced_data_incremental_subset_indices(dataset, n_splits, split_idx):
    n_classes = torch.unique(torch.tensor(dataset.targets)).shape[0]

    def get_subset_indices(X_dataset):
        subset_indices = []

        for c in range(n_classes):
            mask = [_c == c for _c in X_dataset.targets]
            samples_from_c = torch.tensor(mask).nonzero().flatten()
            start_idx = int(split_idx * len(samples_from_c) / n_splits)
            end_idx = int((split_idx + 1) * len(samples_from_c) / n_splits)
            subset_indices.append(samples_from_c[start_idx:end_idx])

        return subset_indices

    train_subset_indices = get_subset_indices(dataset.train_dataset)
    test_subset_indices = get_subset_indices(dataset.test_dataset)

    return train_subset_indices, test_subset_indices


def get_class_incremental_classes_and_subset_indices(dataset, n_splits, split_idx):
    assert 0 <= split_idx < n_splits

    n_classes = torch.unique(torch.tensor(dataset.train_dataset.targets)).shape[0]

    start_class_idx = math.floor(n_classes / n_splits * split_idx)
    end_class_idx = math.floor(n_classes / n_splits * (split_idx + 1))

    class_order = (
        dataset.default_class_order
        if hasattr(dataset, "default_class_order")
        else list(range(n_classes))
    )
    classes = sorted(class_order[start_class_idx:end_class_idx])

    train_mask = [c in classes for c in dataset.train_dataset.targets]
    test_mask = [c in classes for c in dataset.test_dataset.targets]

    train_subset_indices = torch.tensor(train_mask).nonzero().flatten()
    test_subset_indices = torch.tensor(test_mask).nonzero().flatten()

    return classes, train_subset_indices, test_subset_indices


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {"images": batch[0], "labels": batch[1]}
    elif len(batch) == 3:
        batch = {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(
        image_encoder, device_ids=[x for x in range(torch.cuda.device_count())]
    )
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch["images"].cuda())

            all_data["features"].append(features.cpu())

            for key, val in batch.items():
                if key == "images":
                    continue
                if hasattr(val, "cpu"):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = "train" if is_train else "val"
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f"{image_encoder.cache_dir}/{dname}/{split}"
        cached_files = glob.glob(f"{cache_dir}/*")
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f"Getting features from {cache_dir}")
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f"Did not find cached features at {cache_dir}. Building from scratch.")
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print("Not caching because no cache directory was passed.")
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching data at {cache_dir}")
            for name, val in data.items():
                torch.save(val, f"{cache_dir}/{name}.pt")
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data["features"])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data["features"] = torch.from_numpy(data["features"]).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(
            feature_dataset, batch_size=args.batch_size, shuffle=is_train
        )
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader


def get_task_classes(class_order, n_splits, split_idx):
    """Determines the class labels for a specific task."""
    num_classes = len(class_order)
    if num_classes % n_splits != 0:
        raise ValueError("n_splits must evenly divide the number of classes.")
    classes_per_task = num_classes // n_splits
    start_class_idx = split_idx * classes_per_task
    end_class_idx = (split_idx + 1) * classes_per_task
    return class_order[start_class_idx:end_class_idx]


def get_subset_indices_with_classes(dataset, classes):
    """Gets the indices of samples belonging to a set of classes."""
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, classes))[0]
    return indices


class ClassIncrementalDataset(Protocol):
    """What the samplers below need from a dataset wrapper.

    CIFAR100 and ImageNetR both provide these, which is why they can share the
    sampling code rather than each carrying a copy of it.
    """

    train_dataset: Any  # torchvision-style, exposing `.targets`
    test_dataset: Any
    default_class_order: list  # class ids in the order tasks consume them


def construct_train_subset_each_task(
    dataset: ClassIncrementalDataset,
    n_splits: int,
    num_train_data_each_task: int,
    seed: int = 42,
) -> list:
    """One fixed-size sample of each task's training data.

    Used to measure how similar a target environment is to each task, not for
    training — see src/merging/similarity.py.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_subsets = []
    for split_idx in range(n_splits):
        task_classes = get_task_classes(dataset.default_class_order, n_splits, split_idx)
        # self.train_dataset may be an ImageFolder, which looks like it has no
        # `targets` — but its DatasetFolder parent does, and populates it.
        task_indices = get_subset_indices_with_classes(dataset.train_dataset, task_classes)

        num_sampled = _clamp_to_available(
            num_train_data_each_task, len(task_indices), f"train task {split_idx}"
        )
        sampled_task_indices = np.random.choice(task_indices, num_sampled, replace=False)
        train_subsets.append(Subset(dataset.train_dataset, sampled_task_indices))

    return train_subsets


def _clamp_to_available(requested: int, available: int, what: str) -> int:
    """Take what there is when a task cannot cover the request.

    ImageNet-R's classes are unevenly sized, so the mixing ratio cannot always
    be met exactly; the paper notes this and reports it "followed as much as
    possible". Shrinking a target environment changes the experiment, so unlike
    the original — which clamped silently in one dataset and raised in the
    other — it is logged.
    """
    if requested <= available:
        return requested
    logger.warning(
        f"{what}: {requested} samples requested but only {available} available; "
        f"taking {available}. The mixing ratio will not be met exactly."
    )
    return available


def _plan_samples_per_task(
    n_splits: int, num_data: int, ratio_data_from_task: list, seed: int
) -> tuple[list, list]:
    """Decide how many examples come from each task.

    Returns (task_idx_selected, num_data_each_task), the latter indexed by task
    so unselected tasks hold 0.

    The per-task shares are shuffled before being handed out, so which of the
    chosen tasks gets the largest share is itself random — a ratio of
    (0.8, 0.2) over tasks (3, 1) is as likely to mean "80% of task 1" as "80%
    of task 3".
    """
    num_data_each_task = [0] * n_splits
    task_idx_selected = select_target_tasks(n_splits, len(ratio_data_from_task), seed)

    num_data_to_be_selected = (
        np.floor(np.array(ratio_data_from_task) * num_data).astype(int).tolist()
    )
    # Flooring loses a few examples; hand them back out so the total matches.
    remainder = num_data - sum(num_data_to_be_selected)
    for _ in range(remainder):
        num_data_to_be_selected[np.random.randint(0, len(num_data_to_be_selected))] += 1

    num_data_to_be_selected_shuffled = np.random.permutation(num_data_to_be_selected)
    for i, task_idx in enumerate(task_idx_selected):
        num_data_each_task[task_idx] = num_data_to_be_selected_shuffled[i]

    logger.info(
        f"task_idx_selected: {task_idx_selected}, "
        f"num_data_each_task: {num_data_each_task}"
    )
    logger.debug(f"sum: {sum(num_data_each_task)}, num_data: {num_data}")
    return task_idx_selected, num_data_each_task


def construct_target_dataset(
    dataset: ClassIncrementalDataset,
    n_splits: int,
    num_data: int,
    ratio_data_from_task: list,
    seed: int = 42,
):
    """Build one target environment out of the test set.

    Draws `num_data` examples split across the tasks named by
    `ratio_data_from_task`, then splits off 10% of each task's draw as the meta
    dataset the preference vector is estimated from.

    Returns (task_idx_selected, num_data_each_task, meta_data, test_data_list),
    where `num_data_each_task` reports what was actually taken — which is less
    than planned for any task that ran short — and `test_data_list` holds None
    for tasks this environment does not draw from.
    """
    np.random.seed(seed)

    task_idx_selected, num_data_each_task = _plan_samples_per_task(
        n_splits, num_data, ratio_data_from_task, seed
    )

    meta_data_indices = []
    test_data_list = []
    for split_idx, num_samples in zip(range(n_splits), num_data_each_task):
        if num_samples == 0:
            test_data_list.append(None)
            continue

        task_classes = get_task_classes(dataset.default_class_order, n_splits, split_idx)
        task_indices = get_subset_indices_with_classes(dataset.test_dataset, task_classes)

        num_samples = _clamp_to_available(
            num_samples, len(task_indices), f"target task {split_idx}"
        )
        sampled_task_indices = np.random.choice(task_indices, num_samples, replace=False)

        # NOTE: int() floors, so a task contributing fewer than 10 examples
        # yields no meta data at all — which is why src/eval.py skips
        # cifar100-50's all-tasks target.
        num_meta = int(0.1 * len(sampled_task_indices))

        meta_data_indices.extend(sampled_task_indices[:num_meta])
        test_data_list.append(Subset(dataset.test_dataset, sampled_task_indices[num_meta:]))

        logger.debug(
            f"Task {split_idx} selected {num_samples} samples from "
            f"{len(task_classes)} classes\n"
            f"meta data size: {num_meta}, test data size: {num_samples - num_meta}."
        )

    np.random.shuffle(meta_data_indices)
    meta_data = Subset(dataset.test_dataset, meta_data_indices)

    # Report what was actually taken rather than what was planned: the two
    # differ whenever a task ran short.
    num_data_each_task = [len(td) if td is not None else 0 for td in test_data_list]

    return task_idx_selected, num_data_each_task, meta_data, test_data_list
