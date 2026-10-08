"""CITB (Zhang et al., Findings of EMNLP 2023) / Super-NaturalInstructions
task streams, loaded via the `Muennighoff/natural-instructions` HF mirror.

That mirror stores one JSONL file per task (`train/{task}_train.jsonl`,
`test/{task}_test.jsonl`, columns `task_name, id, definition, inputs,
targets`) rather than a single combined split, so a task's data is fetched
by its file URL directly instead of `.filter()`-ing a 100M+ row dataset.
"""

import random

from torch.utils.data import DataLoader

from datasets import load_dataset
from src.nlp.modeling_nlp import BASE_MODEL_NAME
from src.task_spec import TaskSpec

NI_REPO = "Muennighoff/natural-instructions"


# Source: https://github.com/hyintell/CITB/blob/main/data/splits/CIT_splits/cl_dialogue_tasks.txt
INSTR_DIALOG_TASKS = [
    "task1590_diplomacy_text_generation",
    "task1713_convai3_sentence_generation",
    "task565_circa_answer_generation",
    "task1730_personachat_choose_next",
    "task573_air_dialogue_classification",
    "task294_storycommonsense_motiv_text_generation",
    "task1500_dstc3_classification",
    "task1603_smcalflow_sentence_generation",
    "task576_curiosity_dialogs_answer_generation",
    "task639_multi_woz_user_utterance_generation",
    "task1714_convai3_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1729_personachat_generate_next",
    "task574_air_dialogue_sentence_generation",
    "task848_pubmedqa_classification",
    "task611_mutual_multi_turn_dialogue",
    "task766_craigslist_bargains_classification",
    "task1384_deal_or_no_dialog_classification",
    "task1600_smcalflow_sentence_generation",
]  # 19 tasks

# Source: https://github.com/hyintell/CITB/blob/main/data/splits/CIT_splits/cl_38_random_tasks.txt
# minus the 19 INSTR_DIALOG_TASKS above (disjoint from them).
INSTR_DIALOG_PLUS_PLUS_EXTRA_TASKS = [
    "task1549_wiqa_answer_generation_missing_step",
    "task927_yelp_negative_to_positive_style_transfer",
    "task459_matres_static_classification",
    "task379_agnews_topic_classification",
    "task347_hybridqa_incorrect_answer_generation",
    "task1360_numer_sense_multiple_choice_qa_generation",
    "task1151_swap_max_min",
    "task301_record_question_generation",
    "task306_jeopardy_answer_generation_double",
    "task298_storycloze_correct_end_classification",
    "task864_asdiv_singleop_question_answering",
    "task082_babi_t1_single_supporting_fact_question_generation",
    "task1427_country_region_in_world",
    "task1553_cnn_dailymail_summarization",
    "task1203_atomic_classification_xreact",
    "task967_ruletaker_incorrect_fact_generation_based_on_given_paragraph",
    "task598_cuad_answer_generation",
    "task636_extract_and_sort_unique_alphabets_in_a_list",
    "task1607_ethos_text_classification",
]  # 19 tasks

INSTR_DIALOG_PLUS_PLUS_TASKS = (
    INSTR_DIALOG_TASKS + INSTR_DIALOG_PLUS_PLUS_EXTRA_TASKS
)  # 38 tasks

_TASK_LISTS_BY_DATASET = {"CITB19": INSTR_DIALOG_TASKS, "CITB38": INSTR_DIALOG_PLUS_PLUS_TASKS}


def _order_patterns(task_list: list[str]) -> dict[str, list[str]]:
    # "A" is the CITB split file's own order; B/C are fixed deterministic
    # shuffles of it, mirroring src/datasets/cifar100.py's A/B/C pattern for
    # --taskseq_pattern.
    return {
        "A": list(task_list),
        "B": random.Random(1).sample(task_list, len(task_list)),
        "C": random.Random(2).sample(task_list, len(task_list)),
    }


# dataset name -> taskseq_pattern -> ordered task id list
TASK_ORDER_PATTERNS = {name: _order_patterns(tasks) for name, tasks in _TASK_LISTS_BY_DATASET.items()}


def _ni_file_url(split: str, task_name: str) -> str:
    return f"https://huggingface.co/datasets/{NI_REPO}/resolve/main/{split}/{task_name}_{split}.jsonl"


def load_superni_task(
    task_name: str,
    tokenizer,
    batch_size: int = 16,
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> TaskSpec:
    def encode(ex):
        prompt = ex["definition"] + "\n" + ex["inputs"]
        src = tokenizer(
            prompt, truncation=True, max_length=max_src_len, padding="max_length"
        )
        tgt = tokenizer(
            text_target=ex["targets"],
            truncation=True,
            max_length=max_tgt_len,
            padding="max_length",
        )
        src["labels"] = tgt["input_ids"]
        return src

    # `test/{task}_test.jsonl` only exists for the ~119 tasks in Natural
    # Instructions' official held-out test-task split (verified: a 404 for
    # every CITB train-stream task checked). CITB's train-stream tasks ship
    # only a `train/` file here, so the eval set is carved out of it.
    full_ds = load_dataset(
        "json", data_files=_ni_file_url("train", task_name), split="train"
    )
    split_ds = full_ds.train_test_split(test_size=eval_fraction, seed=seed)
    train_ds = split_ds["train"].map(encode)
    eval_ds = split_ds["test"].map(encode)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return TaskSpec(
        name=task_name,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        eval_loader=DataLoader(eval_ds, batch_size=batch_size, shuffle=False),
        task_type="seq2seq",
    )


def build_citb_task_sequence(
    task_ids: list[str], tokenizer_name: str = BASE_MODEL_NAME
) -> list[TaskSpec]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [load_superni_task(t, tokenizer) for t in task_ids]
