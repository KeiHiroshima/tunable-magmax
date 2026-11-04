"""StdCL / LSB as src/nlp/long_sequence_benchmark.py defines them, against
O-LoRA (Wang et al., Findings of EMNLP 2023) and its repository.

The prompt is checked against a verbatim port of O-LoRA's own construction
(uie_dataset_lora.py's load_*_dataset + uie_collator.py::get_instruction with
the flags its T5 scripts pass), so a drift in either spacing or prefixes
fails here rather than silently changing what the model is trained on.

Tests that read O-LoRA's CL_Benchmark are skipped when it is not present.
"""

import os

import pytest

from src.nlp.long_sequence_benchmark import (
    INSTRUCTIONS,
    OLORA_DATA_DIR,
    TASK_CATEGORY,
    TASK_ORDERS,
    build_prompt,
    load_examples,
    normalize_answer,
    task_order,
)

needs_data = pytest.mark.skipif(
    not os.path.isdir(OLORA_DATA_DIR), reason=f"O-LoRA CL_Benchmark not found at {OLORA_DATA_DIR}"
)


def _olora_prompt(task_category, dataset_name, instruction, labels, sentence):
    """O-LoRA's construction, kept in its original shape."""
    if task_category == "COPA":
        instruction = "{0}" + "\nAnswer:"
    else:
        labels_str = ", ".join(labels)
        instruction = instruction + "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"
    prefix = ""
    prefix += "Task:" + task_category + "\n"          # --add_task_name True
    prefix = prefix + "Dataset:"                      # --add_dataset_name True
    prefix = prefix + dataset_name + "\n" if prefix else dataset_name + "\n"
    instruction = prefix + instruction
    return instruction.format(sentence)


@pytest.mark.parametrize("task", sorted(TASK_CATEGORY))
def test_prompt_matches_olora(task):
    category = TASK_CATEGORY[task]
    labels = ["alpha", "beta gamma"]
    sentence = "Title: t\nText: some {braced} text\n"

    expected = _olora_prompt(category, task, INSTRUCTIONS.get(category), labels, sentence)

    assert build_prompt(task, sentence, labels) == expected


def test_orders_are_the_papers_table_7():
    assert TASK_ORDERS["StdCL"] == {
        "1": ["dbpedia", "amazon", "yahoo", "agnews"],
        "2": ["dbpedia", "amazon", "agnews", "yahoo"],
        "3": ["yahoo", "amazon", "agnews", "dbpedia"],
    }
    lsb = TASK_ORDERS["LSB"]
    assert lsb["4"][:4] == ["MNLI", "CB", "WiC", "COPA"]
    assert lsb["5"][:4] == ["MultiRC", "BoolQA", "WiC", "MNLI"]
    # Order 6 is also O-LoRA's scripts/long.sh.
    assert lsb["6"] == ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB",
                        "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"]
    for order in lsb.values():
        assert len(order) == 15 and set(order) == set(TASK_CATEGORY)


def test_unknown_order_is_rejected_with_the_valid_ones():
    with pytest.raises(ValueError, match=r"\['4', '5', '6'\]"):
        task_order("LSB", "1")


def test_normalize_answer_matches_olora():
    assert normalize_answer("  Science or Technology. ") == "science or technology"
    assert normalize_answer("very  positive!") == "very positive"


@needs_data
@pytest.mark.parametrize("task", sorted(TASK_CATEGORY))
def test_every_task_loads_with_its_label_set(task):
    train, labels = load_examples(task, "train")
    test, _ = load_examples(task, "test")

    assert train and test
    assert {ex["target"] for ex in train} <= set(labels)
    assert all(ex["source"].startswith(f"Task:{TASK_CATEGORY[task]}\nDataset:{task}\n") for ex in train[:5])


@needs_data
def test_training_split_is_1000_per_class_where_the_data_allows():
    """O-LoRA trains on train.json whole: 1000 examples per class."""
    for task in ("dbpedia", "yahoo", "agnews", "amazon", "yelp", "MNLI"):
        train, labels = load_examples(task, "train")
        assert len(train) == 1000 * len(labels), task
