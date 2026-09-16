"""Confirms the NLP model wrappers run a forward pass and sit in the same
"Base" parameter-count regime as ViT-B/16 (~86M) — the point of using
bert-base-uncased instead of a larger/smaller checkpoint.
"""

import torch

from src.nlp.modeling_nlp import BertClassifier, build_bert2bert


def test_bert_classifier_forward_and_scale():
    model = BertClassifier()
    model.reset_head(num_labels=3)
    input_ids = torch.randint(0, 30000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    logits = model(input_ids, attention_mask)

    assert logits.shape == (2, 3)
    n_params = sum(p.numel() for p in model.parameters())
    assert 100e6 < n_params < 120e6  # ~110M, ViT-B/16 is ~86M


def test_bert2bert_forward_and_loss():
    model = build_bert2bert()
    input_ids = torch.randint(0, 30000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.randint(0, 30000, (2, 8))

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert out.loss.item() > 0
    n_params = sum(p.numel() for p in model.parameters())
    assert 200e6 < n_params < 260e6  # ~247M: encoder + decoder, ~2x BertClassifier
