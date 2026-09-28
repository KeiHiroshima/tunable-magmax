"""merge_and_evaluate's unified target-environment loop
(src/backends/nlp_classification_backend.py).

Before this, only masked_magmax_with_targetdata was scored against
configs/target_data_config*.json; every other --merge_fn (finetune,
random_mix, average, ties, magmax) was evaluated once on each task's own full
eval split and never touched a target environment at all — making the
baselines incomparable to the proposed method on the same mixture. This file
checks the fix: every merge_fn now goes through the same per-environment loop,
branching only on MergeSpec.needs_target_data (src/merging/registry.py), and
a merge_fn that does not need target data is merged once and reused across
every target environment rather than being recomputed per environment.
"""

import json
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, Dataset

from src.backends import nlp_classification_backend as backend
from src.merging.task_vector import TaskVector
from src.merging.registry import apply_merge
from src.nlp.modeling_nlp import BertClassifier
from src.task_spec import TaskSpec
from src.utils import torch_save


class _TinyTaskDataset(Dataset):
    """A stand-in for LSB's HF-datasets-backed eval split: same dict-batch
    shape (input_ids/attention_mask/labels), tiny and offline."""

    def __init__(self, n: int, num_labels: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, 1000, (n, 8), generator=g)
        self.attention_mask = torch.ones(n, 8, dtype=torch.long)
        self.labels = torch.randint(0, num_labels, (n,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def _make_task(name: str, num_labels: int = 2, n_eval: int = 4, seed: int = 0) -> TaskSpec:
    eval_ds = _TinyTaskDataset(n_eval, num_labels, seed)
    return TaskSpec(
        name=name,
        train_loader=None,
        eval_loader=DataLoader(eval_ds, batch_size=4),
        task_type="classification",
        num_labels=num_labels,
    )


def _write_checkpoints(tmp_path, num_labels=2):
    """A zeroshot BertClassifier plus two divergent 'finetuned' variants,
    saved the way finetune_splitted.py would."""
    base = BertClassifier()
    base.reset_head(num_labels)
    zeroshot_path = tmp_path / "zeroshot.pt"
    torch_save(base, str(zeroshot_path))

    finetuned_paths = []
    for i in range(2):
        model = torch.load(zeroshot_path, weights_only=False)
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_((0.01 * (i + 1)) * torch.randn_like(p))
        path = tmp_path / f"finetuned_{i}.pt"
        torch_save(model, str(path))
        finetuned_paths.append(path)

    return str(zeroshot_path), finetuned_paths


def _write_target_config(tmp_path, monkeypatch, *, num_task_to_be_fetched, ratios, target_ids):
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "tmp_lsb_target.json").write_text(
        json.dumps(
            {
                "dataset_configs": [
                    {
                        "num_task_to_be_fetched": num_task_to_be_fetched,
                        "ratio_task_to_be_fetched": ratio,
                        "variants": [{"target_id": tid, "random_seed": 42}],
                    }
                    for ratio, tid in zip(ratios, target_ids)
                ]
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return "tmp_lsb_target"


def _args(tmp_path, merge_fn, target_config, seed=7):
    return Namespace(
        merge_fn=merge_fn,
        device="cpu",
        seed=seed,
        model="bert-base-uncased",
        target_config=target_config,
        num_target_data=8,
    )


def _setup(tmp_path, monkeypatch, merge_fn, target_config, seed=7):
    tasks = [_make_task("task_a", seed=1), _make_task("task_b", seed=2)]
    zeroshot_path, _ = _write_checkpoints(tmp_path)

    monkeypatch.setattr(backend, "_build_tasks", lambda args: tasks)
    monkeypatch.setattr(backend, "_ckpt_dir", lambda args: str(tmp_path))
    monkeypatch.setattr(backend, "get_zeroshot_checkpoint", lambda model: zeroshot_path)

    return _args(tmp_path, merge_fn, target_config, seed=seed)


def test_baseline_merge_fn_is_scored_against_the_target_environment(tmp_path, monkeypatch):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.5, 0.5]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, "average", target_config)

    results = backend.merge_and_evaluate(args)

    out_path = tmp_path / "average" / f"target1_seed{args.seed}.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved == results[1]
    assert 0.0 <= saved["overall_accuracy"] <= 1.0
    assert saved["overall_total"] == sum(saved["num_data_each_task"].values())
    # Only the proposed method reports a preference vector / unaligned counts.
    assert "weights_each_task" not in saved
    assert "num_unaligned" not in saved


def test_baseline_merge_is_computed_once_and_reused_across_environments(tmp_path, monkeypatch):
    target_config = _write_target_config(
        tmp_path,
        monkeypatch,
        num_task_to_be_fetched=2,
        ratios=[[0.5, 0.5], [0.8, 0.2]],
        target_ids=[1, 2],
    )
    args = _setup(tmp_path, monkeypatch, "magmax", target_config)

    calls = []
    real_apply_merge = apply_merge

    def _counting_apply_merge(spec, task_vectors, **kwargs):
        calls.append(spec.name)
        return real_apply_merge(spec, task_vectors, **kwargs)

    monkeypatch.setattr(backend, "apply_merge", _counting_apply_merge)

    backend.merge_and_evaluate(args)

    assert calls == ["magmax"]  # merged once, not once per target environment
    assert (tmp_path / "magmax" / f"target1_seed{args.seed}.json").exists()
    assert (tmp_path / "magmax" / f"target2_seed{args.seed}.json").exists()


def test_proposed_method_uses_the_environments_ratio_as_its_preference_vector(tmp_path, monkeypatch):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.8, 0.2]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, "masked_magmax_with_targetdata", target_config)

    results = backend.merge_and_evaluate(args)

    saved = results[1]
    assert saved["weights_each_task"] == [0.8, 0.2]
    assert "num_unaligned" in saved
    assert "num_params_all" in saved
