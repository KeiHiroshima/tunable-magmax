"""evaluate_merged_fts_on_target_data (src/eval.py): which merges are rebuilt
per target environment.

A merge that does not take target data is the same model in every
environment, so it is built once per run (as the NLP backend does); only the
preference-weighted merges depend on the environment and are rebuilt for
each. Re-merging the baselines per environment cost minutes per environment
(TIES over 50 task vectors of a full ViT-B-16) for an identical result.

The data, model and evaluation are stubbed out: only the loop's bookkeeping
is under test.
"""

import json
from types import SimpleNamespace

import pytest

from src import eval as eval_module
from src.merging.registry import get_merge_spec
from src.target_env import TargetEnvSpec

TARGET_IDS = [1, 2, 3]


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    calls = []

    def _build(task_vectors, spec, args, pretrained_checkpoint, scaling_coef, **kwargs):
        calls.append(args.target_id)
        log = {"merging_function": spec.name, "scaling_coefficient": scaling_coef}
        if spec.needs_target_data:
            log.update(similarity_metric=args.similarity_metric, weights_each_task=[1.0])
        return object(), log

    envs = [
        TargetEnvSpec(target_id=i, n_tasks_fetched=1, ratio=[1.0], seed=0, num_target_data=10)
        for i in TARGET_IDS
    ]
    monkeypatch.setattr(eval_module, "_build_merged_encoder", _build)
    monkeypatch.setattr(eval_module, "load_target_envs", lambda args, n_tasks: iter(envs))
    monkeypatch.setattr(
        eval_module, "ImageEncoder", lambda args, keep_lang: SimpleNamespace(train_preprocess=None)
    )
    monkeypatch.setattr(eval_module, "get_dataset", lambda *a, **k: None)
    monkeypatch.setattr(eval_module, "construct_train_subset_each_task", lambda *a, **k: [[0]])
    monkeypatch.setattr(
        eval_module, "construct_target_dataset", lambda *a, **k: ([0], [10], [0] * 10, None)
    )
    monkeypatch.setattr(eval_module, "eval_given_dataset", lambda *a, **k: [0.5, 0.5, 0.5])
    return calls


def _args(tmp_path, merge_fn):
    return SimpleNamespace(
        num_train_data_each_task=1, model="ViT-B-16", dataset="CIFAR100", data_location=None,
        batch_size=1, n_splits=5, seed=3, results_db=str(tmp_path), similarity_metric="labels",
        coeff=0.5, merge_fn=merge_fn, merge_space="full",
    )


def _run(tmp_path, merge_fn):
    eval_module.evaluate_merged_fts_on_target_data(
        task_vectors=None, args=_args(tmp_path, merge_fn), spec=get_merge_spec(merge_fn),
        scaling_coef=0.5, pretrained_checkpoint="unused",
    )


@pytest.mark.parametrize("merge_fn", ["ties", "magmax", "average"])
def test_a_baseline_is_merged_once_and_scored_in_every_environment(tmp_path, stubbed, merge_fn):
    _run(tmp_path, merge_fn)

    assert stubbed == [TARGET_IDS[0]]  # merged once, at the first environment
    for target_id in TARGET_IDS:
        saved = json.loads(
            (tmp_path / merge_fn / f"{merge_fn}_lambda0.5_target{target_id}_seed3.json").read_text()
        )
        assert saved["merging_function"] == merge_fn
        assert saved["target_dataset_info"]["target_id"] == target_id


@pytest.mark.parametrize(
    "merge_fn", ["weighted_ties_with_targetdata", "masked_magmax_with_targetdata"]
)
def test_a_preference_weighted_merge_is_rebuilt_per_environment(tmp_path, stubbed, merge_fn):
    _run(tmp_path, merge_fn)

    assert stubbed == TARGET_IDS


def test_a_finished_run_merges_nothing(tmp_path, stubbed):
    _run(tmp_path, "ties")
    stubbed.clear()

    _run(tmp_path, "ties")

    assert stubbed == []
