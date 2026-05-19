import collections
import glob
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


def construct_target_dataset(
    dataset_name: str,
    dataset,
    n_splits: int,
    num_data: int,
    ratio_data_from_task,
    seed: int = 42,
):
    """
    Factory function to construct a target dataset from the test set of a specified dataset.
    """

    if dataset_name == "CIFAR100":
        return dataset._construct_target_cifar100_dataset(
            num_data=num_data,
            n_splits=n_splits,
            num_data_from_tasks=None,
            ratio_data_from_task=ratio_data_from_task,
            seed=seed,
        )
    elif dataset_name == "ImageNetR":
        return dataset._construct_target_imagenetr_dataset(
            num_data=num_data,
            n_splits=n_splits,
            num_data_from_tasks=None,
            ratio_data_from_task=ratio_data_from_task,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported for target dataset construction."
        )
