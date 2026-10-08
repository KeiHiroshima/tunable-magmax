"""The `configs/target_data_config*.json` schema.

Users author these files by hand, so the schema is an external interface — but
it had no single definition. Four places knew that a negative
`num_task_to_be_fetched` means "all tasks" (src/eval.py, src/nlp/target_data.py
and two postprocessing notebooks), and two of them separately knew that
`ratio_task_to_be_fetched == [-1]` means "an even split". The conventions are
resolved here once, and `load_target_envs` hands back plain values with no
sentinels left in them.

Schema:

    {"dataset_configs": [
        {"num_task_to_be_fetched":   int,    # -1 = every task
         "ratio_task_to_be_fetched": [float] # [-1] = uniform over the fetched tasks
         "num_target_data":          int,    # optional; see below
         "variants": [{"target_id": int, "random_seed": int}, ...]}
    ]}

Each entry describes one *kind* of target environment (how many tasks, in what
proportion); its `variants` are independent draws of that kind, each with its
own id and seed.

What this module does NOT do: draw the actual examples. How many to take
from each chosen task, and which ones, differs between the backends (the
vision datasets shuffle the ratio across the chosen tasks and redistribute
rounding remainders; the NLP benchmarks take a plain proportion of each task's
eval split), so that stays with them — see src/datasets/cifar100.py and
src/nlp/target_data.py.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Iterator

CONFIG_DIR = "configs"


@dataclass(frozen=True)
class TargetEnvSpec:
    """One target environment, with every sentinel already resolved."""

    target_id: int
    seed: int
    n_tasks_fetched: int
    ratio: list[float]
    num_target_data: int

    def __post_init__(self):
        assert len(self.ratio) == self.n_tasks_fetched, (
            f"target {self.target_id}: ratio has {len(self.ratio)} entries but "
            f"{self.n_tasks_fetched} tasks are fetched"
        )


def config_path(target_config: str) -> str:
    return os.path.join(CONFIG_DIR, f"{target_config}.json")


def _resolve_n_tasks(raw: int, n_tasks: int) -> int:
    """A negative count means "every task"."""
    return n_tasks if raw < 0 else raw


def _resolve_ratio(raw: list[float], n_fetched: int) -> list[float]:
    """`[-1]` means "split evenly across the fetched tasks"."""
    if raw[0] == -1:
        return [1.0 / n_fetched for _ in range(n_fetched)]
    return list(raw)


def _resolve_num_target_data(config: dict, args) -> int:
    """How many examples this environment holds.

    The split-specific config files (target_data_config_split{5,20,50}.json)
    size each environment in proportion to how many tasks it draws from, and
    carry the number per entry. The general files do not, and defer to
    --num_target_data. The filename check preserves the original behaviour:
    a per-entry number is honoured only by the config written for this
    --n_splits, so pointing --target_config at another split's file falls back
    to the flag rather than silently using that split's sizes.

    Configs without the key are settled before `--n_splits` is consulted, since
    that flag describes the vision class-incremental splits and means nothing
    to the NLP benchmarks, whose configs never carry the key.
    """
    if "num_target_data" not in config:
        return args.num_target_data
    if args.target_config == f"target_data_config_split{args.n_splits}":
        return config["num_target_data"]
    return args.num_target_data


def load_target_envs(args, n_tasks: int) -> Iterator[TargetEnvSpec]:
    """Yield every target environment in `args.target_config`, in file order.

    File order is what assigns target ids to environments, and those ids name
    the result files, so it must stay stable.

    Args:
        n_tasks: how many tasks the benchmark has — n_splits for the vision
            class-incremental splits, len(tasks) for the NLP benchmarks. Used
            to resolve "all tasks".
    """
    with open(config_path(args.target_config), "r") as f:
        dataset_configs = json.load(f)["dataset_configs"]

    for config in dataset_configs:
        n_fetched = _resolve_n_tasks(config["num_task_to_be_fetched"], n_tasks)
        ratio = _resolve_ratio(config["ratio_task_to_be_fetched"], n_fetched)
        num_target_data = _resolve_num_target_data(config, args)

        for variant in config["variants"]:
            yield TargetEnvSpec(
                target_id=variant["target_id"],
                seed=variant["random_seed"],
                n_tasks_fetched=n_fetched,
                ratio=ratio,
                num_target_data=num_target_data,
            )


def select_target_tasks(n_tasks: int, num_to_fetch: int, seed: int) -> list[int]:
    """Which tasks this target environment is a mixture of.

    Seeded from the environment's own seed, so the mixture is a property of the
    environment alone — not of the merge method being evaluated, of how much
    randomness ran before it, or of whether earlier targets were skipped on a
    resumed run. The vision datasets used the process-wide RNG here and were
    subject to all three; see
    test/characterization/test_target_task_selection.py.

    `num_to_fetch` arrives already resolved by load_target_envs, so the
    "-1 means all tasks" convention is not repeated here.
    """
    return random.Random(seed).sample(range(n_tasks), num_to_fetch)
