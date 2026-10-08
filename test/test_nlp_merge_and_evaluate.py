"""merge_and_evaluate's unified target-environment loop
(src/backends/nlp_classification_backend.py), run on a tiny T5 with the
t5-small tokenizer (downloaded on first run).

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

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

from src.backends import nlp_classification_backend as backend
from src.merging.registry import apply_merge
from src.nlp.long_sequence_benchmark import Seq2SeqCollator, build_prompt
from src.nlp.modeling_nlp import freeze_shared_embeddings
from src.task_spec import TaskSpec
from src.utils import torch_save


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("t5-small")


def _tiny_t5():
    """T5's real vocabulary (so the t5-small tokenizer's ids fit) on a tiny,
    randomly initialised body: fast enough to generate with on CPU."""
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32128, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_decoder_layers=1,
        num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    return T5ForConditionalGeneration(config)


def _make_task(name, tokenizer, n_eval=4):
    """A stand-in for a StdCL/LSB task: the same {"source", "target"}
    examples and collator, tiny and offline."""
    labels = ["positive", "negative"]
    examples = [
        {"source": build_prompt("SST-2", f"example {i} of {name}", labels), "target": labels[i % 2]}
        for i in range(n_eval)
    ]
    return TaskSpec(
        name=name,
        train_loader=None,
        eval_loader=DataLoader(examples, batch_size=4, collate_fn=Seq2SeqCollator(tokenizer)),
        task_type="seq2seq",
        num_labels=len(labels),
        labels=labels,
    )


def _write_checkpoints(tmp_path):
    """A zeroshot T5 plus two divergent 'finetuned' variants, saved the way
    finetune_splitted.py would under --finetune_mode full."""
    zeroshot_path = tmp_path / "zeroshot.pt"
    torch_save(_tiny_t5(), str(zeroshot_path))

    finetuned_paths = []
    for i in range(2):
        model = torch.load(zeroshot_path, weights_only=False)
        freeze_shared_embeddings(model)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
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


def _args(tmp_path, merge_fn, target_config, seed=7, eval_full_testsets=False):
    return Namespace(
        merge_fn=merge_fn,
        device="cpu",
        seed=seed,
        model="t5-small",
        finetune_mode="full",
        sequential_finetuning=True,
        scaling_coef=0.5,
        eval_full_testsets=eval_full_testsets,
        target_config=target_config,
        num_target_data=8,
    )


def _setup(tmp_path, monkeypatch, tokenizer, merge_fn, target_config, seed=7, **kwargs):
    tasks = [_make_task("task_a", tokenizer), _make_task("task_b", tokenizer)]
    zeroshot_path, _ = _write_checkpoints(tmp_path)

    monkeypatch.setattr(backend, "_build_tasks", lambda args: tasks)
    monkeypatch.setattr(backend, "_ckpt_dir", lambda args: str(tmp_path))
    monkeypatch.setattr(backend, "_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(backend, "get_zeroshot_checkpoint", lambda model: zeroshot_path)

    return _args(tmp_path, merge_fn, target_config, seed=seed, **kwargs)


def test_baseline_merge_fn_is_scored_against_the_target_environment(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.5, 0.5]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, tokenizer, "average", target_config)

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


def test_baseline_merge_is_computed_once_and_reused_across_environments(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path,
        monkeypatch,
        num_task_to_be_fetched=2,
        ratios=[[0.5, 0.5], [0.8, 0.2]],
        target_ids=[1, 2],
    )
    args = _setup(tmp_path, monkeypatch, tokenizer, "magmax", target_config)

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


def test_proposed_method_uses_the_environments_ratio_as_its_preference_vector(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.8, 0.2]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, tokenizer, "masked_magmax_with_targetdata", target_config)

    results = backend.merge_and_evaluate(args)

    saved = results[1]
    assert saved["weights_each_task"] == [0.8, 0.2]
    assert "num_unaligned" in saved
    assert "num_params_all" in saved


def test_finished_environment_is_skipped_on_rerun(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.5, 0.5]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, tokenizer, "magmax", target_config)
    out_path = tmp_path / "magmax" / f"target1_seed{args.seed}.json"
    out_path.parent.mkdir()
    finished = {"target_id": 1, "overall_accuracy": 0.123}
    out_path.write_text(json.dumps(finished))

    calls = []
    monkeypatch.setattr(backend, "apply_merge", lambda *a, **k: calls.append(1))

    results = backend.merge_and_evaluate(args)

    assert calls == []  # every environment finished: nothing merged
    assert results[1] == finished
    assert json.loads(out_path.read_text()) == finished  # not overwritten


def test_unfinished_results_file_is_re_evaluated(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.5, 0.5]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, tokenizer, "magmax", target_config)
    out_path = tmp_path / "magmax" / f"target1_seed{args.seed}.json"
    out_path.parent.mkdir()
    out_path.write_text('{"target_id": 1')  # a write cut off mid-way

    results = backend.merge_and_evaluate(args)

    saved = json.loads(out_path.read_text())
    assert "overall_accuracy" in saved
    assert saved == results[1]


def test_full_testsets_report_the_papers_average_accuracy(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.5, 0.5]], target_ids=[1]
    )
    args = _setup(tmp_path, monkeypatch, tokenizer, "average", target_config, eval_full_testsets=True)

    backend.merge_and_evaluate(args)

    saved = json.loads((tmp_path / "average" / f"full_testsets_seed{args.seed}.json").read_text())
    assert set(saved["taskwise_accuracies"]) == {"task_a", "task_b"}
    assert saved["average_accuracy"] == pytest.approx(
        sum(saved["taskwise_accuracies"].values()) / 2
    )


def test_proposed_method_has_no_single_model_to_score_on_full_testsets(tmp_path, monkeypatch, tokenizer):
    target_config = _write_target_config(
        tmp_path, monkeypatch, num_task_to_be_fetched=2, ratios=[[0.5, 0.5]], target_ids=[1]
    )
    args = _setup(
        tmp_path, monkeypatch, tokenizer, "masked_magmax_with_targetdata", target_config,
        eval_full_testsets=True,
    )

    backend.merge_and_evaluate(args)

    assert not (tmp_path / "masked_magmax_with_targetdata" / f"full_testsets_seed{args.seed}.json").exists()
