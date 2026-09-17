"""The task-incremental fine-tuning loop (src/nlp/finetune_nlp.py).

Before this loop was shared, its control flow had no test at all: the
characterization test for the vision pipeline stubs os.path.exists to always
return True, so it only ever walked the skip path. These tests drive the real
loop with a recording stub in place of training, and pin the invariants I1-I5
documented in finetune_nlp.py.

Nothing here loads a real model: `train_task` never runs, and checkpoints are
one-line JSON stand-ins, so the whole file is millisecond-fast.
"""

import json
import os
from argparse import Namespace

import pytest

from src.nlp import finetune_nlp
from src.paths import finetuned_path


class FakeTask:
    """Stands in for TaskSpec — the loop only ever reads .name/.num_labels."""

    def __init__(self, name, num_labels):
        self.name = name
        self.num_labels = num_labels


class FakeModel:
    def __init__(self, origin):
        self.origin = origin  # which checkpoint this was loaded from
        self.heads_set = []


@pytest.fixture
def loop(tmp_path, monkeypatch):
    """Runs the real loop against fake checkpoint I/O, and records what happened.

    Returns a callable: run(n_tasks, sequential=..., existing=[...], with_head=...)
    -> a record dict.
    """
    zeroshot = tmp_path / "zeroshot.pt"
    ckpt_dir = tmp_path / "run"
    record = {"loaded_from": [], "trained": [], "saved": [], "heads": [], "built_base": 0}

    def fake_torch_save(model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"origin": getattr(model, "origin", "fresh")}, f)
        # Only per-task checkpoints count as loop output; materialising the
        # zero-shot base is setup, and is observed via record["built_base"].
        if os.path.dirname(path) == str(ckpt_dir):
            record["saved"].append(path)

    def fake_torch_load(path, device=None):
        assert os.path.exists(path), f"loop tried to load a missing checkpoint: {path}"
        record["loaded_from"].append(path)
        return FakeModel(origin=path)

    def build_base_model(model_name):
        record["built_base"] += 1
        return FakeModel(origin="base")

    monkeypatch.setattr(finetune_nlp, "torch_save", fake_torch_save)
    monkeypatch.setattr(finetune_nlp, "torch_load", fake_torch_load)
    monkeypatch.setattr(
        finetune_nlp, "get_zeroshot_checkpoint", lambda model_name: str(zeroshot)
    )

    def run(n_tasks=3, *, sequential=True, existing=(), with_head=False):
        tasks = [FakeTask(f"task{i}", num_labels=i + 2) for i in range(n_tasks)]
        for idx in existing:
            fake_torch_save(FakeModel(origin="preexisting"), finetuned_path(str(ckpt_dir), idx))
        record["saved"].clear()  # the pre-seeded files are setup, not loop output
        record["loaded_from"].clear()

        def train_task(model, task, args):
            record["trained"].append(task.name)

        def prepare_model(model, task):
            record["heads"].append((task.name, task.num_labels))

        finetune_nlp.finetune_task_sequence(
            Namespace(model="fake-model", device="cpu", sequential_finetuning=sequential),
            tasks,
            str(ckpt_dir),
            build_base_model=build_base_model,
            train_task=train_task,
            prepare_model=prepare_model if with_head else None,
        )
        record["zeroshot"] = str(zeroshot)
        record["ckpt_dir"] = str(ckpt_dir)
        return record

    return run


# --- I1 / I5 ---------------------------------------------------------------


def test_each_task_writes_its_own_numbered_checkpoint(loop):
    r = loop(n_tasks=3)

    assert r["saved"] == [finetuned_path(r["ckpt_dir"], i) for i in range(3)]


def test_tasks_are_trained_in_order(loop):
    r = loop(n_tasks=4)

    assert r["trained"] == ["task0", "task1", "task2", "task3"]


# --- I2: resumability ------------------------------------------------------


def test_existing_checkpoints_are_not_retrained(loop):
    """Re-issuing the same command after an interrupted run must pick up where
    it left off rather than redoing finished tasks."""
    r = loop(n_tasks=3, existing=[0, 1])

    assert r["trained"] == ["task2"]
    assert r["saved"] == [finetuned_path(r["ckpt_dir"], 2)]


def test_a_fully_finished_run_does_no_work(loop):
    r = loop(n_tasks=3, existing=[0, 1, 2])

    assert r["trained"] == []
    assert r["saved"] == []


# --- I3: the chain survives skipped tasks ----------------------------------


def test_sequential_run_chains_from_the_previous_task(loop):
    r = loop(n_tasks=3, sequential=True)

    assert r["loaded_from"] == [
        r["zeroshot"],
        finetuned_path(r["ckpt_dir"], 0),
        finetuned_path(r["ckpt_dir"], 1),
    ]


def test_resume_chains_across_a_skipped_task(loop):
    """The invariant most easily lost: task 1 was already done, so task 2 must
    continue from *task 1's* checkpoint — not from task 0's, and not from the
    zero-shot base. Dropping the bookkeeping here silently trains the rest of
    the sequence from a stale model, with no error to notice.
    """
    r = loop(n_tasks=3, sequential=True, existing=[1])

    assert r["trained"] == ["task0", "task2"]
    assert r["loaded_from"] == [
        r["zeroshot"],                            # task 0
        finetuned_path(r["ckpt_dir"], 1),         # task 2 resumes from task 1
    ]


def test_resume_after_a_leading_skip(loop):
    r = loop(n_tasks=3, sequential=True, existing=[0])

    assert r["trained"] == ["task1", "task2"]
    assert r["loaded_from"] == [
        finetuned_path(r["ckpt_dir"], 0),
        finetuned_path(r["ckpt_dir"], 1),
    ]


# --- I4 / independent fine-tuning ------------------------------------------


def test_first_task_starts_from_the_zeroshot_checkpoint(loop):
    r = loop(n_tasks=2, sequential=True)

    assert r["loaded_from"][0] == r["zeroshot"]


def test_independent_finetuning_always_restarts_from_the_zeroshot(loop):
    """Without --sequential-finetuning every task is trained from the base, so
    the task vectors are independent rather than cumulative."""
    r = loop(n_tasks=3, sequential=False)

    assert r["loaded_from"] == [r["zeroshot"]] * 3


def test_independent_finetuning_still_skips_finished_tasks(loop):
    r = loop(n_tasks=3, sequential=False, existing=[1])

    assert r["trained"] == ["task0", "task2"]
    assert r["loaded_from"] == [r["zeroshot"]] * 2


# --- zero-shot materialisation ---------------------------------------------


def test_zeroshot_is_built_once_when_missing(loop):
    r = loop(n_tasks=3)

    assert r["built_base"] == 1
    assert os.path.exists(r["zeroshot"])


def test_existing_zeroshot_is_reused_not_rebuilt(loop, tmp_path):
    loop(n_tasks=1)  # materialises it
    r = loop(n_tasks=1)

    # Still 1 from the first run: the second run found the file and skipped the
    # build. Rebuilding would change the base every task vector is taken against.
    assert r["built_base"] == 1


# --- prepare_model hook -----------------------------------------------------


def test_classification_gets_a_fresh_head_sized_for_each_task(loop):
    r = loop(n_tasks=3, with_head=True)

    assert r["heads"] == [("task0", 2), ("task1", 3), ("task2", 4)]


def test_seq2seq_touches_no_head(loop):
    """prepare_model=None is how the seq2seq backend says "the LM head is
    shared"; the loop must not invent a head-swapping step for it."""
    r = loop(n_tasks=3, with_head=False)

    assert r["heads"] == []


def test_skipped_tasks_do_not_get_a_head_swap(loop):
    r = loop(n_tasks=3, existing=[0], with_head=True)

    assert r["heads"] == [("task1", 3), ("task2", 4)]


# --- the backends actually use the shared loop ------------------------------


@pytest.mark.parametrize(
    "module_name,expect_prepare_model",
    [("nlp_classification_backend", True), ("nlp_seq2seq_backend", False)],
)
def test_backends_delegate_to_the_shared_loop(module_name, expect_prepare_model, monkeypatch):
    """Both NLP backends must go through finetune_task_sequence rather than
    growing a private copy of the loop again."""
    import importlib

    backend = importlib.import_module(f"src.backends.{module_name}")
    captured = {}

    def fake_sequence(args, tasks, ckpt_dir, **kwargs):
        captured.update(kwargs)
        captured["ckpt_dir"] = ckpt_dir

    monkeypatch.setattr(backend, "finetune_task_sequence", fake_sequence)
    monkeypatch.setattr(backend, "_build_tasks", lambda args: [])

    args = Namespace(
        model="bert-base-uncased",
        dataset="LSB" if expect_prepare_model else "CITB19",
        epochs=3,
        taskseq_pattern="A",
        seed=3,
        sequential_finetuning=True,
    )
    backend.finetune(args)

    assert captured["ckpt_dir"] == backend._ckpt_dir(args)
    assert callable(captured["build_base_model"])
    assert callable(captured["train_task"])
    assert (captured.get("prepare_model") is not None) is expect_prepare_model
