"""Confirms src/merging/task_vector.py — written for the CLIP/ViT pipeline —
works unmodified on BERT state dicts. This is the key claim behind reusing
it as-is for NLP instead of writing a parallel merging implementation.
"""

import sys

import torch

from src.nlp.modeling_nlp import BertClassifier
from src.utils import torch_load, torch_save


def _set_fake_cli_args(monkeypatch):
    # src/merging/task_vector.py calls parse_arguments() at *import* time, so
    # it needs sys.argv to look like a real invocation even though this test
    # never touches the parsed args themselves.
    monkeypatch.setattr(sys, "argv", ["test", "--model", "bert-base-uncased", "--dataset", "dummy"])


def test_task_vector_roundtrip_on_bert_classifier(tmp_path, monkeypatch):
    _set_fake_cli_args(monkeypatch)
    from src.merging.task_vector import TaskVector

    base_path = tmp_path / "base.pt"
    finetuned_path = tmp_path / "finetuned.pt"

    model = BertClassifier()
    model.reset_head(num_labels=2)
    torch_save(model, str(base_path))

    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.01 * torch.randn_like(p))
    torch_save(model, str(finetuned_path))

    tv = TaskVector(str(base_path), str(finetuned_path))
    assert len(tv.vector) > 0

    merged = tv.apply_to(str(base_path), scaling_coef=1.0)

    finetuned = torch_load(str(finetuned_path))
    key = "encoder.embeddings.word_embeddings.weight"
    # coef=1.0 means "fully apply the delta": merged should equal finetuned exactly.
    assert torch.allclose(merged.state_dict()[key], finetuned.state_dict()[key], atol=1e-6)
