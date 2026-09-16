"""src/nlp/target_data.py: preference-vector construction from a target
environment's known task-sampling ratio (no similarity/embedding estimate).
"""

from src.nlp.target_data import build_target_weights, resolve_ratio, select_target_tasks


def test_select_target_tasks_picks_requested_count_deterministically():
    a = select_target_tasks(n_tasks=15, num_to_fetch=3, seed=42)
    b = select_target_tasks(n_tasks=15, num_to_fetch=3, seed=42)
    assert a == b  # same seed -> same selection
    assert len(a) == 3
    assert len(set(a)) == 3  # no duplicates
    assert all(0 <= i < 15 for i in a)


def test_select_target_tasks_all_when_negative():
    assert select_target_tasks(n_tasks=5, num_to_fetch=-1, seed=1) == [0, 1, 2, 3, 4]


def test_resolve_ratio_passthrough():
    assert resolve_ratio([0.8, 0.2], num_selected=2) == [0.8, 0.2]


def test_resolve_ratio_uniform_when_negative_one():
    assert resolve_ratio([-1], num_selected=4) == [0.25, 0.25, 0.25, 0.25]


def test_build_target_weights_zero_pads_unselected_tasks():
    weights = build_target_weights(n_tasks=5, task_idx_selected=[3, 1], ratio=[0.8, 0.2])
    assert weights == [0.0, 0.2, 0.0, 0.8, 0.0]
    assert sum(weights) == 1.0
