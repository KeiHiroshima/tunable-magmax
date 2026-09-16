"""Long Sequence Benchmark: 15 text classification tasks (Razdaibiedina et al.,
Progressive Prompts, ICLR 2023), loaded via Hugging Face `datasets`.

Column names below were confirmed against each dataset's actual schema
(`datasets-server.huggingface.co/first-rows`) rather than assumed — GLUE/
SuperGLUE `test` splits ship with labels hidden (-1), so evaluation uses
`validation` (`validation_matched` for MNLI) instead.
"""

import random

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.nlp.modeling_nlp import BASE_MODEL_NAME
from src.task_spec import TaskSpec

# name -> (repo_id, config, text column(s), label column, num_classes, (train_split, eval_split))
TASK_DEFS = {
    "yelp":    ("Yelp/yelp_review_full", "yelp_review_full", "text", "label", 5, ("train", "test")),
    "dbpedia": ("fancyzhx/dbpedia_14", "dbpedia_14", "content", "label", 14, ("train", "test")),
    "yahoo":   ("community-datasets/yahoo_answers_topics", "yahoo_answers_topics", "question_title", "topic", 10, ("train", "test")),
    "agnews":  ("fancyzhx/ag_news", "default", "text", "label", 4, ("train", "test")),
    # HF has no direct mirror of the Amazon-5 CL benchmark; this is a
    # schema-compatible substitute (same text/label(0-4) shape as yelp).
    "amazon":  ("SetFit/amazon_reviews_multi_en", "default", "text", "label", 5, ("train", "test")),
    "imdb":    ("stanfordnlp/imdb", "plain_text", "text", "label", 2, ("train", "test")),
    "mnli":    ("nyu-mll/glue", "mnli", ("premise", "hypothesis"), "label", 3, ("train", "validation_matched")),
    "qqp":     ("nyu-mll/glue", "qqp", ("question1", "question2"), "label", 2, ("train", "validation")),
    "rte":     ("nyu-mll/glue", "rte", ("sentence1", "sentence2"), "label", 2, ("train", "validation")),
    "sst2":    ("nyu-mll/glue", "sst2", "sentence", "label", 2, ("train", "validation")),
    "wic":     ("aps/super_glue", "wic", ("sentence1", "sentence2"), "label", 2, ("train", "validation")),
    "cb":      ("aps/super_glue", "cb", ("premise", "hypothesis"), "label", 3, ("train", "validation")),
    "boolq":   ("aps/super_glue", "boolq", ("question", "passage"), "label", 2, ("train", "validation")),
    # copa / multirc have columns that don't fit the single/pair shape above;
    # they get their own encoders in _SPECIAL_ENCODERS/_SPECIAL_DEFS below.
}

_SPECIAL_DEFS = {
    "copa":    ("aps/super_glue", "copa", 2, ("train", "validation")),
    "multirc": ("aps/super_glue", "multirc", 2, ("train", "validation")),
}


def _encode_pair_or_single(ex, tokenizer, text_cols, max_len):
    if isinstance(text_cols, tuple):
        return tokenizer(ex[text_cols[0]], ex[text_cols[1]], truncation=True, max_length=max_len, padding="max_length")
    return tokenizer(ex[text_cols], truncation=True, max_length=max_len, padding="max_length")


def _encode_copa(ex, tokenizer, max_len):
    # Simplified to a binary (premise+question, chosen choice) pair rather
    # than the original "pick one of two choices" formulation.
    context = ex["premise"] + " " + ex["question"]
    choice = ex["choice1"] if ex["label"] == 0 else ex["choice2"]
    return tokenizer(context, choice, truncation=True, max_length=max_len, padding="max_length")


def _encode_multirc(ex, tokenizer, max_len):
    context = ex["paragraph"] + " " + ex["question"]
    return tokenizer(context, ex["answer"], truncation=True, max_length=max_len, padding="max_length")


_SPECIAL_ENCODERS = {"copa": _encode_copa, "multirc": _encode_multirc}

_ALL_TASK_NAMES = list(TASK_DEFS.keys()) + list(_SPECIAL_DEFS.keys())

# Mirrors src/datasets/cifar100.py's class_order_dict A/B/C pattern for
# --taskseq_pattern. "A" is just this file's definition order (not a
# reproduction of any specific paper's published task order); B/C are fixed
# deterministic shuffles, so all three are reproducible across runs.
TASK_ORDER_PATTERNS = {
    "A": list(_ALL_TASK_NAMES),
    "B": random.Random(1).sample(_ALL_TASK_NAMES, len(_ALL_TASK_NAMES)),
    "C": random.Random(2).sample(_ALL_TASK_NAMES, len(_ALL_TASK_NAMES)),
}


def load_lsb_task(name: str, tokenizer, batch_size: int = 32, max_len: int = 256) -> TaskSpec:
    if name in _SPECIAL_ENCODERS:
        repo, config, n_cls, (train_split, eval_split) = _SPECIAL_DEFS[name]
        encode_fn = _SPECIAL_ENCODERS[name]
        encode = lambda ex: {**encode_fn(ex, tokenizer, max_len), "labels": ex["label"]}
    else:
        repo, config, text_cols, label_col, n_cls, (train_split, eval_split) = TASK_DEFS[name]
        encode = lambda ex: {**_encode_pair_or_single(ex, tokenizer, text_cols, max_len), "labels": ex[label_col]}

    train_ds = load_dataset(repo, config, split=train_split).map(encode)
    eval_ds = load_dataset(repo, config, split=eval_split).map(encode)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return TaskSpec(
        name=name,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        eval_loader=DataLoader(eval_ds, batch_size=batch_size, shuffle=False),
        task_type="classification",
        num_labels=n_cls,
    )


def build_lsb_task_sequence(task_order: list[str], tokenizer_name: str = BASE_MODEL_NAME) -> list[TaskSpec]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [load_lsb_task(name, tokenizer) for name in task_order]
