"""Confirms the NLP model builders produce what the pipelines expect: T5 in
fp32 with its shared embedding freezable (StdCL/LSB), and BERT2BERT (CITB).
Downloads t5-small / bert-base-uncased on first run.
"""

import torch

from src.nlp.modeling_nlp import build_bert2bert, build_t5, freeze_shared_embeddings


def test_t5_loads_in_fp32_with_olora_dropout():
    model = build_t5("t5-small")

    assert next(model.parameters()).dtype == torch.float32
    assert model.config.dropout_rate == 0.1


def test_freeze_shared_embeddings_freezes_exactly_the_tied_embedding():
    model = build_t5("t5-small")
    freeze_shared_embeddings(model)

    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"shared.weight"}  # the one tensor lm_head and embed_tokens alias
    input_ids = torch.randint(0, 32000, (2, 8))
    loss = model(input_ids=input_ids, labels=input_ids[:, :3].contiguous()).loss
    loss.backward()
    assert model.shared.weight.grad is None


def test_bert2bert_forward_and_loss():
    model = build_bert2bert()
    input_ids = torch.randint(0, 30000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.randint(0, 30000, (2, 8))

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert out.loss.item() > 0
    n_params = sum(p.numel() for p in model.parameters())
    assert 200e6 < n_params < 260e6  # ~247M: encoder + decoder
