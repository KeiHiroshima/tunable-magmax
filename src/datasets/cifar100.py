import logging
import os
import random
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100

from src.datasets.common import get_subset_indices_with_classes, get_task_classes

class_order_dict = {
    "A": [
        70,
        89,
        11,
        13,
        63,
        53,
        86,
        57,
        41,
        43,
        14,
        98,
        52,
        73,
        95,
        96,
        33,
        16,
        39,
        74,
        25,
        88,
        35,
        28,
        79,
        82,
        72,
        4,
        30,
        17,
        59,
        97,
        36,
        38,
        29,
        55,
        83,
        7,
        22,
        48,
        19,
        47,
        2,
        44,
        67,
        71,
        34,
        84,
        6,
        46,
        61,
        8,
        80,
        10,
        49,
        15,
        68,
        9,
        99,
        40,
        27,
        45,
        51,
        37,
        21,
        64,
        92,
        24,
        60,
        31,
        5,
        91,
        93,
        90,
        65,
        66,
        77,
        20,
        58,
        62,
        23,
        76,
        75,
        42,
        0,
        26,
        87,
        50,
        3,
        56,
        81,
        1,
        94,
        69,
        18,
        78,
        54,
        12,
        85,
        32,
    ],
    "B": [
        95,
        20,
        64,
        82,
        59,
        30,
        49,
        65,
        10,
        57,
        87,
        84,
        1,
        62,
        16,
        77,
        78,
        53,
        11,
        35,
        5,
        68,
        50,
        38,
        54,
        76,
        55,
        63,
        4,
        39,
        51,
        58,
        33,
        43,
        36,
        73,
        91,
        61,
        14,
        29,
        74,
        81,
        56,
        42,
        0,
        37,
        32,
        48,
        7,
        86,
        13,
        90,
        99,
        45,
        88,
        52,
        18,
        27,
        98,
        46,
        12,
        85,
        22,
        93,
        83,
        72,
        9,
        70,
        75,
        89,
        31,
        60,
        92,
        8,
        44,
        24,
        66,
        67,
        34,
        79,
        41,
        2,
        17,
        19,
        28,
        23,
        47,
        71,
        21,
        15,
        97,
        25,
        6,
        40,
        26,
        96,
        80,
        94,
        3,
        69,
    ],
    "C": [
        80,
        97,
        98,
        57,
        60,
        8,
        22,
        78,
        88,
        36,
        14,
        49,
        66,
        16,
        64,
        84,
        85,
        31,
        79,
        7,
        72,
        53,
        73,
        48,
        83,
        10,
        18,
        55,
        46,
        4,
        56,
        37,
        9,
        69,
        5,
        67,
        89,
        92,
        41,
        32,
        39,
        29,
        70,
        75,
        47,
        45,
        93,
        28,
        30,
        34,
        82,
        76,
        42,
        77,
        12,
        17,
        51,
        23,
        95,
        58,
        11,
        87,
        52,
        35,
        33,
        50,
        3,
        94,
        62,
        13,
        26,
        63,
        6,
        38,
        90,
        1,
        15,
        19,
        91,
        20,
        43,
        81,
        0,
        2,
        21,
        24,
        74,
        71,
        86,
        54,
        96,
        61,
        59,
        99,
        40,
        68,
        25,
        65,
        44,
        27,
    ],
}  # A = default

logger = getLogger(__name__)


class CIFAR100:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
        args=None,
        **kwargs,
    ):
        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.test_dataset.classes

        if args.taskseq_pattern not in class_order_dict:
            raise ValueError(f"Unknown taskseq_pattern: {args.taskseq_pattern}")
        self.default_class_order = class_order_dict[args.taskseq_pattern]

    def _construct_train_subset_each_task(
        self,
        n_splits: int,
        num_train_data_each_task: int,
        seed: int = 42,
    ):
        """
        Constructs a list of subsets of the CIFAR100 train dataset, one for each task.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_subsets = []
        for split_idx in range(n_splits):
            task_classes = get_task_classes(
                self.default_class_order, n_splits, split_idx
            )
            task_indices = get_subset_indices_with_classes(
                self.train_dataset, task_classes
            )

            sampled_task_indices = np.random.choice(
                task_indices, num_train_data_each_task, replace=False
            )
            train_subsets.append(Subset(self.train_dataset, sampled_task_indices))

        return train_subsets

    def _construct_target_cifar100_dataset(
        self,
        num_data: int,
        n_splits: int,
        num_data_from_tasks: list = None,
        ratio_data_from_task: list = None,
        seed: int = 42,
    ):
        """
        Constructs a subset of the CIFAR100 test dataset by sampling from each task.
        """
        np.random.seed(seed)

        # Determine the number of samples to draw from each task
        if num_data_from_tasks is not None:
            if len(num_data_from_tasks) != n_splits:
                raise ValueError("Length of num_data_from_tasks must match n_splits.")
            if sum(num_data_from_tasks) != num_data:
                logging.warning(
                    f"Sum of num_data_from_tasks ({sum(num_data_from_tasks)}) does not equal num_data ({num_data})."
                )
            samples_per_task = num_data_from_tasks

        elif ratio_data_from_task is not None:
            # assign num task ramdomly
            num_data_each_task = [0] * n_splits
            task_idx_selected = random.sample(
                range(n_splits), len(ratio_data_from_task)
            )
            num_data_to_be_selected = (
                np.floor(np.array(ratio_data_from_task) * num_data).astype(int).tolist()
            )
            # adjust last element to match total num_data

            remainder = num_data - sum(num_data_to_be_selected)
            for _ in range(remainder):
                num_data_to_be_selected[
                    np.random.randint(0, len(num_data_to_be_selected))
                ] += 1

            num_data_to_be_selected_shuffled = np.random.permutation(
                num_data_to_be_selected
            )

            for i, task_idx in enumerate(task_idx_selected):
                num_data_each_task[task_idx] = num_data_to_be_selected_shuffled[i]

            logger.info(
                f"task_idx_selected: {task_idx_selected}, \nnum_data_each_task: {num_data_each_task}"
            )
            logger.debug(f"sum: {sum(num_data_each_task)}, num_data: {num_data}")

        else:
            # uniform distribution
            num_data_each_task = [num_data // n_splits] * n_splits
            remainder = num_data % n_splits
            for i in range(remainder):
                num_data_each_task[np.random.randint(0, n_splits)] += 1

        meta_data_indices = []
        test_data_list = []
        for split_idx, num_samples in zip(range(n_splits), num_data_each_task):
            if num_samples == 0:
                test_data_list.append(None)
                continue
            else:
                task_classes = get_task_classes(
                    self.default_class_order, n_splits, split_idx
                )
                task_indices = get_subset_indices_with_classes(
                    self.test_dataset, task_classes
                )

                if len(task_indices) < num_samples:
                    raise ValueError(
                        f"Task {split_idx} (classes {task_classes}) has only {len(task_indices)} test samples, "
                        f"but {num_samples} were requested."
                    )

                sampled_task_indices = np.random.choice(
                    task_indices, num_samples, replace=False
                )

                num_meta = int(0.1 * len(sampled_task_indices))
                # error: 10% for meta data = 0.4 = 0 for cifar100-50 with all task data

                meta_data_indices.extend(sampled_task_indices[:num_meta])
                test_data_list.append(
                    Subset(self.test_dataset, sampled_task_indices[num_meta:])
                )

                logging.debug(
                    f"Task {split_idx} selected {num_samples} samples from classes {len(task_classes)}\nmeta data size: {num_meta}, test data size: {num_samples - num_meta}."
                )

        np.random.shuffle(meta_data_indices)
        meta_data = Subset(self.test_dataset, meta_data_indices)

        return (
            task_idx_selected,
            num_data_each_task,
            meta_data,
            test_data_list,
        )
