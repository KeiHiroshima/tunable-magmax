"""How eval_given_dataset turns per-task results into the two reported numbers.

`do_eval` counts correct predictions and examples but used to return only the
ratio, so eval_given_dataset ran the whole evaluation a second time — the same
model over the same data, with the inference loop copied out of do_eval — just
to recover the totals. It now returns the counts and the second pass is gone.

The two numbers it produces are not interchangeable, and the results JSON
records both:

    average_accuracy  mean of the per-task accuracies — every task counts once
    overall_accuracy  correct / examples over everything — every example counts
                      once, so a task the environment draws more heavily from
                      weighs more

These tests pin that distinction and the counting that backs it.
"""

from argparse import Namespace

import pytest
import torch

from src import eval as eval_module
from src.utils import do_eval


class FakeLoader:
    """Yields (images, labels) batches; maybe_dictionarize turns them into the
    dict do_eval expects."""

    def __init__(self, n_correct, n_wrong, batch_size=4):
        self.rows = [(1, 1)] * n_correct + [(1, 0)] * n_wrong
        self.batch_size = batch_size

    def __iter__(self):
        for start in range(0, len(self.rows), self.batch_size):
            chunk = self.rows[start : start + self.batch_size]
            # one logit per class; class 1 wins when the prediction is "1"
            logits = torch.tensor([[0.0, 1.0] if p else [1.0, 0.0] for p, _ in chunk])
            labels = torch.tensor([y for _, y in chunk])
            yield logits, labels

    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size


class IdentityModel(torch.nn.Module):
    """The fake loader already yields logits, so the model passes them through."""

    def forward(self, x):
        return x

    def eval(self):
        return self


# --- do_eval reports counts, not just the ratio -----------------------------


@pytest.mark.parametrize(
    "n_correct,n_wrong", [(10, 0), (0, 10), (7, 3), (1, 999), (33, 67)]
)
def test_do_eval_reports_counts_alongside_the_ratio(n_correct, n_wrong):
    metrics = do_eval(IdentityModel(), FakeLoader(n_correct, n_wrong), device="cpu")

    assert metrics["correct"] == n_correct
    assert metrics["n"] == n_correct + n_wrong
    assert metrics["top1"] == pytest.approx(n_correct / (n_correct + n_wrong))


def test_counts_are_plain_ints():
    """They are summed across tasks and written to JSON, so they should not
    arrive as floats or tensors."""
    metrics = do_eval(IdentityModel(), FakeLoader(7, 3), device="cpu")

    assert isinstance(metrics["correct"], int)
    assert isinstance(metrics["n"], int)
    assert isinstance(metrics["top1"], float)


# --- the two accuracies -----------------------------------------------------


def _eval_with_fake_tasks(monkeypatch, per_task, batch_size=4):
    """Run eval_given_dataset over tasks described as (n_correct, n_wrong),
    with None for a task this environment draws nothing from."""
    loaders = {}

    def fake_dataloader(test_data, **kwargs):
        return loaders[id(test_data)]

    tasks = []
    for spec in per_task:
        if spec is None:
            tasks.append(None)
            continue
        marker = object()
        loaders[id(marker)] = FakeLoader(*spec, batch_size=batch_size)
        tasks.append(marker)

    monkeypatch.setattr(
        eval_module, "get_classification_head", lambda args, name: torch.nn.Identity()
    )
    monkeypatch.setattr(
        eval_module, "ImageClassifier", lambda encoder, head: IdentityModel()
    )
    monkeypatch.setattr(torch.utils.data, "DataLoader", fake_dataloader)

    args = Namespace(model="ViT-B-16", device="cpu", batch_size=batch_size)
    return eval_module.eval_given_dataset(None, tasks, "CIFAR100", args)


def test_returns_per_task_accuracies_then_average_then_overall(monkeypatch):
    result = _eval_with_fake_tasks(monkeypatch, [(8, 2), (6, 4)])

    assert result[:-2] == [pytest.approx(0.8), pytest.approx(0.6)]
    assert result[-2] == pytest.approx(0.7)  # average over tasks
    assert result[-1] == pytest.approx(14 / 20)  # overall over examples


def test_overall_weights_tasks_by_example_count(monkeypatch):
    """The distinction that makes both numbers worth reporting: a large task at
    50% and a small one at 100% average to 75%, but overall is nearer 50%."""
    result = _eval_with_fake_tasks(monkeypatch, [(50, 50), (4, 0)])

    assert result[-2] == pytest.approx(0.75)
    assert result[-1] == pytest.approx(54 / 104)
    assert result[-1] != pytest.approx(result[-2])


def test_tasks_with_no_data_score_zero_but_are_left_out_of_both_means(monkeypatch):
    """A target environment draws from only some tasks; the rest must not drag
    the average down, nor contribute to the overall count."""
    result = _eval_with_fake_tasks(monkeypatch, [(8, 2), None, (6, 4), None])

    assert result[1] == 0.0 and result[3] == 0.0
    assert result[-2] == pytest.approx(0.7)  # mean of 0.8 and 0.6 only
    assert result[-1] == pytest.approx(14 / 20)


def test_overall_equals_the_micro_average_of_the_task_counts(monkeypatch):
    """What the deleted second pass computed. It concatenated every task's data
    and counted again; summing the per-task counts gives the same number.
    """
    per_task = [(17, 8), (3, 1), None, (40, 60)]
    result = _eval_with_fake_tasks(monkeypatch, per_task)

    correct = sum(c for spec in per_task if spec for c in [spec[0]])
    total = sum(c + w for spec in per_task if spec for c, w in [spec])
    assert result[-1] == pytest.approx(correct / total)


def test_evaluation_runs_once_per_task(monkeypatch):
    """The point of the change: one pass over the data, not two."""
    calls = []
    real_do_eval = eval_module.utils.do_eval

    def counting_do_eval(model, dl, device, **kwargs):
        calls.append(id(dl))
        return real_do_eval(model, dl, device, **kwargs)

    monkeypatch.setattr(eval_module.utils, "do_eval", counting_do_eval)
    _eval_with_fake_tasks(monkeypatch, [(8, 2), (6, 4), (1, 1)])

    assert len(calls) == 3
    assert len(set(calls)) == 3
