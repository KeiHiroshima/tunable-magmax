"""Confirms the src/nlp data loaders actually work against the real
HF-hosted datasets (not mocks) — network access required, results cached
under datasets/hf_cache after the first run.
"""

import pytest
from transformers import AutoTokenizer

from src.nlp.citb_superni import (
    INSTR_DIALOG_PLUS_PLUS_TASKS,
    INSTR_DIALOG_TASKS,
    load_superni_task,
)
from src.nlp.long_sequence_benchmark import _SPECIAL_DEFS, TASK_DEFS, load_lsb_task


@pytest.fixture(scope="module")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_lsb_task_defs_cover_15_tasks():
    assert len(TASK_DEFS) + len(_SPECIAL_DEFS) == 15


def test_citb_task_id_counts():
    # InstrDialog / InstrDialog++ sizes from the CITB paper (Zhang et al., 2023).
    assert len(INSTR_DIALOG_TASKS) == 19
    assert len(INSTR_DIALOG_PLUS_PLUS_TASKS) == 38
    assert set(INSTR_DIALOG_TASKS) <= set(INSTR_DIALOG_PLUS_PLUS_TASKS)


def test_load_lsb_pair_task(bert_tokenizer):
    # "rte" exercises the (text_a, text_b) pair path through TASK_DEFS.
    task = load_lsb_task("rte", bert_tokenizer, batch_size=4)
    assert task.task_type == "classification"
    assert task.num_labels == 2
    batch = next(iter(task.train_loader))
    assert batch["input_ids"].shape == (4, 256)
    assert batch["attention_mask"].shape == (4, 256)
    assert batch["labels"].shape == (4,)


def test_load_lsb_special_task(bert_tokenizer):
    # "copa" exercises the custom _encode_copa path (columns don't fit the
    # generic single/pair shape used by the rest of TASK_DEFS).
    task = load_lsb_task("copa", bert_tokenizer, batch_size=2)
    assert task.num_labels == 2
    batch = next(iter(task.train_loader))
    assert batch["input_ids"].shape[0] == 2


def test_load_superni_task(bert_tokenizer):
    task = load_superni_task(INSTR_DIALOG_TASKS[0], bert_tokenizer, batch_size=4)
    assert task.task_type == "seq2seq"
    batch = next(iter(task.train_loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].ndim == 2  # target token ids, not class indices
