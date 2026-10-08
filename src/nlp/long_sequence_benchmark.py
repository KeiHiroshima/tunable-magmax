"""The continual-learning text classification benchmarks of O-LoRA (Wang et
al., Findings of EMNLP 2023), in the T5 instruction-tuning form that paper
evaluates:

    StdCL  the standard CL benchmark, 4 tasks (Orders 1-3)
    LSB    the long sequence benchmark, 15 tasks (Orders 4-6)

Every task is text-to-text: the model reads an instruction, the task's label
options and the input, and generates the label string. Accuracy is exact
match on that string.

Data is O-LoRA's preprocessed `CL_Benchmark` (github.com/cmnfriend/O-LoRA),
read from $MAGMAX_OLORA_DATA_DIR. Each task directory there holds
`{train,dev,test,labels}.json`, a record being `{"sentence", "label"}` with
the input already rendered as text. Train is used whole (1000 examples per
class, fewer for CB/COPA), as O-LoRA's "full" sampling strategy does; test is
the evaluation split (O-LoRA ships dev.json identical to it).

The prompt reproduces O-LoRA's src/uie_collator.py::get_instruction with
--add_task_name True --add_dataset_name True and --instruction_strategy
single, which every T5 run script of that repository passes:

    Task:{category}\\nDataset:{task}\\n{instruction}Option: {l1, l2, ...} \\n{sentence}\\nAnswer:

COPA is the exception, as in O-LoRA's load_COPA_dataset: its sentence already
spells out the two choices, so it gets neither an instruction nor an Option
line.
"""

import json
import os
import string

from torch.utils.data import DataLoader

from src.task_spec import TaskSpec

OLORA_DATA_DIR = os.environ.get(
    "MAGMAX_OLORA_DATA_DIR", os.path.expanduser("~/O-LoRA/CL_Benchmark")
)

# O-LoRA's max_source_length / max_target_length / generation_max_length.
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 50

EVAL_BATCH_SIZE = 64

# task (the CL_Benchmark directory and the "Dataset:" prefix) -> category
# (its parent directory, the "Task:" prefix, and the instruction it gets).
TASK_CATEGORY = {
    "yelp": "SC",
    "amazon": "SC",
    "IMDB": "SC",
    "SST-2": "SC",
    "dbpedia": "TC",
    "agnews": "TC",
    "yahoo": "TC",
    "MNLI": "NLI",
    "CB": "NLI",
    "RTE": "NLI",
    "QQP": "QQP",
    "WiC": "WiC",
    "COPA": "COPA",
    "BoolQA": "BoolQA",
    "MultiRC": "MultiRC",
}

# O-LoRA's configs/instruction_config.json (paper Table 8). COPA has none.
INSTRUCTIONS = {
    "NLI": 'What is the logical relationship between the "sentence 1" and the "sentence 2"? Choose one from the option.\n',
    "QQP": 'Whether the "first sentence" and the "second sentence" have the same meaning? Choose one from the option.\n',
    "SC": "What is the sentiment of the following paragraph? Choose one from the option.\n",
    "TC": "What is the topic of the following paragraph? Choose one from the option.\n",
    "BoolQA": "According to the following passage, is the question true or false? Choose one from the option.\n",
    "MultiRC": "According to the following passage and question, is the candidate answer true or false? Choose one from the option.\n",
    "WiC": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.\n",
}

# Paper Table 7. --taskseq_pattern picks the order; the dataset fixes which
# orders exist, since the two benchmarks have different task sets.
TASK_ORDERS = {
    "StdCL": {
        "1": ["dbpedia", "amazon", "yahoo", "agnews"],
        "2": ["dbpedia", "amazon", "agnews", "yahoo"],
        "3": ["yahoo", "amazon", "agnews", "dbpedia"],
    },
    "LSB": {
        "4": [
            "MNLI",
            "CB",
            "WiC",
            "COPA",
            "QQP",
            "BoolQA",
            "RTE",
            "IMDB",
            "yelp",
            "amazon",
            "SST-2",
            "dbpedia",
            "agnews",
            "MultiRC",
            "yahoo",
        ],
        "5": [
            "MultiRC",
            "BoolQA",
            "WiC",
            "MNLI",
            "CB",
            "COPA",
            "QQP",
            "RTE",
            "IMDB",
            "SST-2",
            "dbpedia",
            "agnews",
            "yelp",
            "amazon",
            "yahoo",
        ],
        "6": [
            "yelp",
            "amazon",
            "MNLI",
            "CB",
            "COPA",
            "QQP",
            "RTE",
            "IMDB",
            "SST-2",
            "dbpedia",
            "agnews",
            "yahoo",
            "MultiRC",
            "BoolQA",
            "WiC",
        ],
    },
}


def task_order(dataset: str, pattern: str) -> list[str]:
    try:
        orders = TASK_ORDERS[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown NLP dataset {dataset!r}. Known: {sorted(TASK_ORDERS)}"
        ) from None
    if pattern not in orders:
        raise ValueError(
            f"--taskseq_pattern {pattern!r} is not an order of {dataset}; "
            f"choose one of {sorted(orders)}"
        )
    return orders[pattern]


def build_prompt(task: str, sentence: str, labels: list[str]) -> str:
    category = TASK_CATEGORY[task]
    prefix = f"Task:{category}\nDataset:{task}\n"
    if category == "COPA":
        return prefix + sentence + "\nAnswer:"
    options = "Option: " + ", ".join(labels) + " \n"
    return prefix + INSTRUCTIONS[category] + options + sentence + "\nAnswer:"


def normalize_answer(s: str) -> str:
    """O-LoRA's src/compute_metrics.py::normalize_answer: lowercase, strip
    punctuation, collapse whitespace."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(s.split())


class Seq2SeqCollator:
    """Tokenises a batch of {"source", "target"} examples, padding to the
    longest in the batch (O-LoRA pads "longest" too, which matters: padding
    every input to 512 would multiply the cost of the short tasks).

    `targets` passes the raw label strings through for exact-match scoring.
    """

    def __init__(
        self,
        tokenizer,
        max_source_length=MAX_SOURCE_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch):
        sources = [ex["source"] for ex in batch]
        targets = [ex["target"] for ex in batch]
        enc = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        labels = tgt["input_ids"].masked_fill(tgt["attention_mask"] == 0, -100)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "targets": targets,
        }


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_examples(
    task: str, split: str, data_dir: str = OLORA_DATA_DIR
) -> tuple[list[dict], list[str]]:
    """(examples, labels) for one split of one task. Each example is
    {"source": prompt, "target": label string}."""
    task_dir = os.path.join(data_dir, TASK_CATEGORY[task], task)
    labels = _read_json(os.path.join(task_dir, "labels.json"))
    records = _read_json(os.path.join(task_dir, f"{split}.json"))
    examples = [
        {"source": build_prompt(task, r["sentence"], labels), "target": r["label"]}
        for r in records
    ]
    return examples, labels


def load_task(
    task: str, tokenizer, batch_size: int, data_dir: str = OLORA_DATA_DIR
) -> TaskSpec:
    train, labels = load_examples(task, "train", data_dir)
    test, _ = load_examples(task, "test", data_dir)
    collate = Seq2SeqCollator(tokenizer)
    return TaskSpec(
        name=task,
        train_loader=DataLoader(
            train, batch_size=batch_size, shuffle=True, collate_fn=collate
        ),
        eval_loader=DataLoader(
            test, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate
        ),
        task_type="seq2seq",
        num_labels=len(labels),
        labels=labels,
    )


def build_task_sequence(
    dataset: str, pattern: str, tokenizer_name: str, batch_size: int
) -> list[TaskSpec]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [
        load_task(task, tokenizer, batch_size) for task in task_order(dataset, pattern)
    ]
