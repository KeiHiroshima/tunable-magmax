"""Confirms src/merging/task_vector.py — written for the CLIP/ViT pipeline —
works unmodified on T5 state dicts. This is the key claim behind reusing it
as-is for NLP instead of writing a parallel merging implementation. Runs on a
tiny randomly initialised T5, offline.
"""

import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.merging.task_vector import TaskVector
from src.nlp.modeling_nlp import freeze_shared_embeddings
from src.utils import torch_load, torch_save

KEY = "encoder.block.0.layer.0.SelfAttention.q.weight"


def _tiny_t5():
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_decoder_layers=2,
        num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    return T5ForConditionalGeneration(config)


def _finetuned_variants(base_path, tmp_path, n):
    paths = []
    for i in range(n):
        model = torch_load(str(base_path))
        freeze_shared_embeddings(model)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_((0.01 * (i + 1)) * torch.randn_like(p))
        path = tmp_path / f"finetuned_{i}.pt"
        torch_save(model, str(path))
        paths.append(str(path))
    return paths


def test_task_vector_roundtrip_on_t5(tmp_path):
    base_path = tmp_path / "base.pt"
    torch_save(_tiny_t5(), str(base_path))
    (finetuned_path,) = _finetuned_variants(base_path, tmp_path, 1)

    tv = TaskVector(str(base_path), finetuned_path)
    merged = tv.apply_to(str(base_path), scaling_coef=1.0)

    finetuned = torch_load(finetuned_path)
    # coef=1.0 means "fully apply the delta": merged should equal finetuned exactly.
    for key in (KEY, "shared.weight", "lm_head.weight"):
        assert torch.allclose(merged.state_dict()[key], finetuned.state_dict()[key], atol=1e-6)
    # the frozen, tied embedding has a zero task vector under every alias
    assert all(torch.all(tv.vector[k] == 0) for k in ("shared.weight", "lm_head.weight"))


def test_mask_and_merge_by_weights_on_t5_task_vectors(tmp_path):
    """mask_and_merge_by_weights (the proposed method's merge) runs on plain
    T5 task vectors. weights=[1.0, 0.0] is a checkable edge case: every
    element should come from task A, none from task B."""
    from src.merging.task_vectors import mask_and_merge_by_weights

    base_path = tmp_path / "base.pt"
    torch_save(_tiny_t5(), str(base_path))
    task_vectors = [TaskVector(str(base_path), p) for p in _finetuned_variants(base_path, tmp_path, 2)]

    merged_tv, num_unaligned, num_params_all = mask_and_merge_by_weights(
        task_vectors, weights_each_task=[1.0, 0.0], seed=0
    )

    assert torch.allclose(merged_tv.vector[KEY], task_vectors[0].vector[KEY], atol=1e-6)
    assert num_params_all == sum(v.numel() for v in task_vectors[0].vector.values())
