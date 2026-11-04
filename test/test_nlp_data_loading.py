"""Confirms the CITB data loader actually works against the real HF-hosted
datasets (not mocks) — network access required, results cached under
datasets/hf_cache after the first run. StdCL/LSB read O-LoRA's local
CL_Benchmark instead; test/test_nlp_benchmark.py covers them.
"""

import pytest
from transformers import AutoTokenizer

from src.nlp.citb_superni import (
    INSTR_DIALOG_PLUS_PLUS_TASKS,
    INSTR_DIALOG_TASKS,
    load_superni_task,
)


@pytest.fixture(scope="module")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_citb_task_id_counts():
    # InstrDialog / InstrDialog++ sizes from the CITB paper (Zhang et al., 2023).
    assert len(INSTR_DIALOG_TASKS) == 19
    assert len(INSTR_DIALOG_PLUS_PLUS_TASKS) == 38
    assert set(INSTR_DIALOG_TASKS) <= set(INSTR_DIALOG_PLUS_PLUS_TASKS)


def test_load_superni_task(bert_tokenizer):
    task = load_superni_task(INSTR_DIALOG_TASKS[0], bert_tokenizer, batch_size=4)
    assert task.task_type == "seq2seq"
    batch = next(iter(task.train_loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].ndim == 2  # target token ids, not class indices
