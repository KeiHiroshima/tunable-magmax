"""src/nlp/target_data.py: preference-vector construction from a target
environment's known task-sampling ratio (no similarity/embedding estimate).

Task selection itself now lives in src/target_env.py, shared with the vision
datasets; test/characterization/test_target_task_selection.py covers it.
"""

from src.nlp.target_data import build_target_weights
from src.target_env import select_target_tasks


def test_select_target_tasks_picks_requested_count_deterministically():
    a = select_target_tasks(n_tasks=15, num_to_fetch=3, seed=42)
    b = select_target_tasks(n_tasks=15, num_to_fetch=3, seed=42)
    assert a == b  # same seed -> same selection
    assert len(a) == 3
    assert len(set(a)) == 3  # no duplicates
    assert all(0 <= i < 15 for i in a)


def test_select_target_tasks_can_take_every_task():
    """`num_to_fetch` arrives already resolved from src/target_env.py, so "all
    tasks" reaches this function as the task count, not as -1."""
    assert sorted(select_target_tasks(n_tasks=5, num_to_fetch=5, seed=1)) == [0, 1, 2, 3, 4]


def test_build_target_weights_zero_pads_unselected_tasks():
    weights = build_target_weights(n_tasks=5, task_idx_selected=[3, 1], ratio=[0.8, 0.2])
    assert weights == [0.0, 0.2, 0.0, 0.8, 0.0]
    assert sum(weights) == 1.0
