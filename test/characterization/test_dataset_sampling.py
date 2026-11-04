"""src/datasets/common.py — building a target environment out of a test set.

CIFAR-100 and ImageNet-R each carried a copy of this (~135 lines apiece). The
copies had drifted on exactly one question — what to do when a task cannot
cover its share — with CIFAR-100 raising and ImageNet-R quietly taking what
there was. The paper settles it for ImageNet-R-20/50 ("the mixing ratio could
not be strictly satisfied due to the shortage of test data; however, it was
followed as much as possible"), so the shared version takes what there is, and
says so in the log.

These tests run against a stand-in dataset rather than real CIFAR-100, so they
need no download and finish in milliseconds.
"""

import logging
from argparse import Namespace

import numpy as np
import pytest

from src.datasets.common import (
    construct_target_dataset,
    construct_train_subset_each_task,
)
from src.target_env import load_target_envs, select_target_tasks

# CIFAR-100's test set: 100 classes, 100 images each.
CIFAR_CLASSES = 100
CIFAR_TEST_PER_CLASS = 100


class FakeSplit:
    """The only thing the samplers read off a split is `.targets`."""

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


class FakeDataset:
    """Satisfies the ClassIncrementalDataset protocol."""

    def __init__(self, n_classes=20, per_class_train=50, per_class_test=30):
        self.train_dataset = FakeSplit(
            [c for c in range(n_classes) for _ in range(per_class_train)]
        )
        self.test_dataset = FakeSplit(
            [c for c in range(n_classes) for _ in range(per_class_test)]
        )
        self.default_class_order = list(range(n_classes))


def _uneven_dataset(per_class_test):
    """A test split whose classes are different sizes, like ImageNet-R's."""
    dataset = FakeDataset(n_classes=len(per_class_test))
    dataset.test_dataset = FakeSplit(
        [c for c, n in enumerate(per_class_test) for _ in range(n)]
    )
    return dataset


# --- the planning step ------------------------------------------------------


def test_requested_total_is_distributed_across_the_chosen_tasks():
    dataset = FakeDataset(n_classes=20, per_class_test=30)

    selected, counts, meta, tests = construct_target_dataset(
        dataset, n_splits=5, num_data=100, ratio_data_from_task=[0.5, 0.5], seed=42
    )

    assert len(selected) == 2
    # `counts` reports the evaluation half only; the meta half is returned
    # separately. See test_counts_exclude_the_meta_split.
    assert sum(counts) + len(meta) == 100
    assert [i for i, c in enumerate(counts) if c > 0] == sorted(selected)


def test_unchosen_tasks_get_no_data():
    dataset = FakeDataset(n_classes=20, per_class_test=30)

    selected, counts, _, tests = construct_target_dataset(
        dataset, n_splits=5, num_data=100, ratio_data_from_task=[0.5, 0.5], seed=42
    )

    for task in range(5):
        if task not in selected:
            assert counts[task] == 0
            assert tests[task] is None


def test_rounding_remainder_is_redistributed():
    """A ratio of thirds floors to 33+33+33 = 99 of 100; the leftover has to go
    somewhere or the environment is one example short of what was asked for."""
    dataset = FakeDataset(n_classes=20, per_class_test=30)

    _, counts, meta, _ = construct_target_dataset(
        dataset, n_splits=5, num_data=100, ratio_data_from_task=[1 / 3] * 3, seed=42
    )

    assert sum(counts) + len(meta) == 100


def test_shares_are_shuffled_across_the_chosen_tasks():
    """Which chosen task gets the biggest share is itself random: (0.8, 0.2)
    does not mean the lower-numbered task gets 80%."""
    dataset = FakeDataset(n_classes=20, per_class_test=30)
    biggest_is_first = set()

    for seed in range(20):
        selected, counts, _, _ = construct_target_dataset(
            dataset, n_splits=10, num_data=100, ratio_data_from_task=[0.8, 0.2], seed=seed
        )
        biggest_is_first.add(counts[selected[0]] > counts[selected[1]])

    assert biggest_is_first == {True, False}, "the share assignment is not shuffled"


# --- the meta split ---------------------------------------------------------


def test_ten_percent_of_each_task_becomes_meta_data():
    dataset = FakeDataset(n_classes=20, per_class_test=30)

    _, counts, meta, tests = construct_target_dataset(
        dataset, n_splits=5, num_data=100, ratio_data_from_task=[0.5, 0.5], seed=42
    )

    kept = sum(len(t) for t in tests if t is not None)
    assert len(meta) + kept == 100
    assert len(meta) == pytest.approx(10, abs=2)


def test_counts_exclude_the_meta_split():
    """`num_data_each_task` describes the evaluation set, not the whole draw.

    This is the ImageNet-R behaviour, adopted for both. CIFAR-100 used to
    return the *planned* draw instead, so the results JSON reported the meta
    examples twice — once in "num_meta_data" and again inside
    "num_target_data". Nothing downstream reads those fields, so the only
    effect is that the recorded numbers are now consistent with each other.
    """
    dataset = FakeDataset(n_classes=20, per_class_test=100)

    _, counts, meta, tests = construct_target_dataset(
        dataset, n_splits=5, num_data=200, ratio_data_from_task=[0.5, 0.5], seed=42
    )

    assert sum(counts) == sum(len(t) for t in tests if t is not None)
    assert sum(counts) + len(meta) == 200
    assert len(meta) > 0  # otherwise the assertion above proves nothing


def test_a_task_contributing_under_ten_examples_yields_no_meta_data():
    """int() floors the 10%, so a task giving 9 examples gives 0 meta. This is
    why src/eval.py skips cifar100-50's all-tasks target: spread 200 examples
    over 50 tasks and every task lands under the threshold."""
    dataset = FakeDataset(n_classes=50, per_class_test=30)

    _, _, meta, _ = construct_target_dataset(
        dataset, n_splits=50, num_data=200, ratio_data_from_task=[1 / 50] * 50, seed=42
    )

    assert len(meta) == 0


# --- clamping ---------------------------------------------------------------


def test_clamping_shrinks_the_draw_and_warns(caplog):
    """ImageNet-R's classes are unevenly sized; a task that cannot cover its
    share gives what it has, and the shortfall is logged rather than silent."""
    dataset = _uneven_dataset([100] * 4 + [3] * 4)  # tasks 2,3 are tiny

    with caplog.at_level(logging.WARNING):
        _, counts, _, _ = construct_target_dataset(
            dataset, n_splits=4, num_data=200, ratio_data_from_task=[0.5, 0.5], seed=7
        )

    assert sum(counts) < 200, "nothing was clamped; the fixture no longer bites"
    assert any("requested but only" in r.message for r in caplog.records)


def test_reported_counts_are_what_was_actually_taken():
    """The counts feed the results JSON, so they must describe the environment
    that was evaluated, not the one that was planned."""
    dataset = _uneven_dataset([100] * 4 + [3] * 4)

    _, counts, _, tests = construct_target_dataset(
        dataset, n_splits=4, num_data=200, ratio_data_from_task=[0.5, 0.5], seed=7
    )

    for task, subset in enumerate(tests):
        assert counts[task] == (len(subset) if subset is not None else 0)


def test_no_clamping_leaves_the_plan_intact(caplog):
    dataset = FakeDataset(n_classes=20, per_class_test=100)

    with caplog.at_level(logging.WARNING):
        _, counts, _, _ = construct_target_dataset(
            dataset, n_splits=5, num_data=100, ratio_data_from_task=[0.5, 0.5], seed=42
        )

    assert not [r for r in caplog.records if "requested but only" in r.message]


# --- CIFAR-100 is unaffected by adopting clamping ---------------------------


@pytest.mark.parametrize(
    "target_config,n_splits,default_num_target_data",
    [
        ("target_data_config", 5, 1000),
        ("target_data_config", 20, 500),
        ("target_data_config", 50, 200),
        ("target_data_config_split5", 5, 1000),
        ("target_data_config_split20", 20, 500),
        ("target_data_config_split50", 50, 200),
    ],
)
def test_cifar100_never_clamps_in_any_configured_setting(
    target_config, n_splits, default_num_target_data
):
    """Why this refactor is numerics-neutral for CIFAR-100.

    CIFAR-100 used to raise where ImageNet-R clamped. Adopting clamping only
    changes behaviour where the old code would have raised — and across every
    shipped config it never gets close enough to. The sizes are exact (100
    images per class), so this can be checked without the data.
    """
    available_per_task = CIFAR_CLASSES // n_splits * CIFAR_TEST_PER_CLASS
    args = Namespace(
        target_config=target_config,
        n_splits=n_splits,
        num_target_data=default_num_target_data,
    )

    for env in load_target_envs(args, n_tasks=n_splits):
        selected = select_target_tasks(n_splits, env.n_tasks_fetched, env.seed)
        np.random.seed(env.seed)
        planned = np.floor(np.array(env.ratio) * env.num_target_data).astype(int).tolist()
        for _ in range(env.num_target_data - sum(planned)):
            planned[np.random.randint(0, len(planned))] += 1

        assert max(planned) <= available_per_task, (
            f"{target_config} n_splits={n_splits} target={env.target_id}: "
            f"{max(planned)} requested per task, {available_per_task} available"
        )
        assert len(selected) == len(planned)


def test_cifar100_split50_has_no_headroom():
    """Recorded because it is the tight one: each task is asked for exactly
    what it has. Adding a larger entry to target_data_config_split50 would
    start clamping CIFAR-100, which the test above would then catch."""
    available_per_task = CIFAR_CLASSES // 50 * CIFAR_TEST_PER_CLASS
    args = Namespace(
        target_config="target_data_config_split50", n_splits=50, num_target_data=200
    )

    biggest = max(
        max(np.floor(np.array(env.ratio) * env.num_target_data).astype(int))
        for env in load_target_envs(args, n_tasks=50)
    )

    assert biggest == available_per_task == 200


# --- training subsets -------------------------------------------------------


def test_train_subsets_are_one_per_task_and_sized_as_asked():
    dataset = FakeDataset(n_classes=20, per_class_train=50)

    subsets = construct_train_subset_each_task(
        dataset, n_splits=5, num_train_data_each_task=100, seed=42
    )

    assert len(subsets) == 5
    assert all(len(s) == 100 for s in subsets)


def test_train_subsets_clamp_when_a_task_is_short(caplog):
    dataset = FakeDataset(n_classes=20, per_class_train=5)

    with caplog.at_level(logging.WARNING):
        subsets = construct_train_subset_each_task(
            dataset, n_splits=5, num_train_data_each_task=100, seed=42
        )

    assert all(len(s) == 20 for s in subsets)  # 4 classes x 5 images
    assert any("requested but only" in r.message for r in caplog.records)


def test_train_subsets_are_deterministic():
    dataset = FakeDataset(n_classes=20, per_class_train=50)

    first = construct_train_subset_each_task(dataset, 5, 100, seed=42)
    second = construct_train_subset_each_task(dataset, 5, 100, seed=42)

    assert [list(s.indices) for s in first] == [list(s.indices) for s in second]


# --- both datasets go through the shared code -------------------------------


def test_neither_dataset_class_carries_its_own_sampler():
    from src.datasets.cifar100 import CIFAR100
    from src.datasets.imagenetr import ImageNetR

    for cls in (CIFAR100, ImageNetR):
        assert not hasattr(cls, "_construct_train_subset_each_task"), cls.__name__
        assert not [m for m in dir(cls) if m.startswith("_construct_target")], cls.__name__
