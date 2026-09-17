"""Target-environment construction for task-incremental NLP benchmarks (LSB).

Vision's masked_magmax_with_targetdata infers weights_each_task from a
*similarity* between a sampled target mixture and each task's own labels
(src/merging/similarity.py::count_labels), because vision's tasks are
class-incremental splits of one shared dataset — the target sample's true
task membership must be inferred from its class labels.

LSB tasks are already separate datasets (task-incremental), so a target
environment's task membership is known by construction: we choose which
tasks it draws from and in what ratio, then sample from exactly those
tasks. weights_each_task is therefore just that ratio, directly — no
similarity/embedding computation needed.

Which environments exist, what their ratios are, and which tasks each one
mixes all come from src/target_env.py — the vision and NLP pipelines agree on
those. This module covers what is specific to the NLP benchmarks: turning the
ratio into a preference vector, and drawing the examples.
"""

import random

from torch.utils.data import Subset

from src.task_spec import TaskSpec


def build_target_weights(n_tasks: int, task_idx_selected: list[int], ratio: list[float]) -> list[float]:
    """The preference vector: weights_each_task[i] = the fraction of this
    target environment's traffic that comes from task i (0 for tasks not
    present in this environment). This *is* the preference vector — no
    further estimation step."""
    weights = [0.0] * n_tasks
    for idx, r in zip(task_idx_selected, ratio):
        weights[idx] = r
    return weights


def sample_target_eval_subset(task: TaskSpec, num_examples: int, seed: int) -> Subset:
    """A small held-out sample from this task's own eval split, standing in
    for "the amount of this task's traffic the target environment sees"."""
    dataset = task.eval_loader.dataset
    n = min(num_examples, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), n)
    return Subset(dataset, indices)
