"""Confirms src/merging/task_vector.py — written for the CLIP/ViT pipeline —
works unmodified on BERT state dicts. This is the key claim behind reusing
it as-is for NLP instead of writing a parallel merging implementation.
"""

import torch

from src.merging.task_vector import TaskVector
from src.nlp.modeling_nlp import BertClassifier
from src.utils import torch_load, torch_save


def test_task_vector_roundtrip_on_bert_classifier(tmp_path):
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


def test_mask_and_merge_by_weights_on_bert_task_vectors(tmp_path):
    """mask_and_merge_by_weights (extracted from
    merge_max_abs_masked_with_targetdata for src/backends/
    nlp_classification_backend.py's masked_magmax_with_targetdata path) runs
    on plain BERT task vectors. weights=[1.0, 0.0] is a checkable edge case:
    every element should come from task A, none from task B."""
    from src.merging.task_vectors import mask_and_merge_by_weights

    base_path = tmp_path / "base.pt"
    base_model = BertClassifier()
    base_model.reset_head(num_labels=2)
    torch_save(base_model, str(base_path))

    finetuned_paths = []
    for i in range(2):
        model = torch_load(str(base_path))
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_((0.01 * (i + 1)) * torch.randn_like(p))
        path = tmp_path / f"finetuned_{i}.pt"
        torch_save(model, str(path))
        finetuned_paths.append(str(path))

    task_vectors = [TaskVector(str(base_path), p) for p in finetuned_paths]

    merged_tv, num_unaligned, num_params_all = mask_and_merge_by_weights(
        task_vectors, weights_each_task=[1.0, 0.0], seed=0
    )

    key = "encoder.embeddings.word_embeddings.weight"
    assert torch.allclose(merged_tv.vector[key], task_vectors[0].vector[key], atol=1e-6)
    assert num_params_all == sum(v.numel() for v in task_vectors[0].vector.values())
