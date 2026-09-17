"""The learning-rate schedule, and why warmup is a ratio rather than a step count.

Task schedules across this project differ by orders of magnitude: 40 optimizer
steps per task for ImageNet-R-50, 790 for CIFAR-100-5, and within a single LSB
run anything from 80 steps (`cb`) to 203,000 (`yelp`). A warmup expressed in
steps that suits one of those covers another's whole schedule — the learning
rate then ramps linearly from zero, never reaches `--lr`, and never decays. The
old default of 500 steps did exactly that to seven of the eight settings.

`--warmup_ratio` cannot do that for any task size, and rescales by itself when
`--epochs` or `--batch_size` change. The NLP trainer used to clamp warmup to 10%
of the schedule privately, so the flag meant different things on the two sides;
both now go through src.trainer.build_scheduler.
"""

import math
from argparse import Namespace

import pytest
import torch

from src.trainer import build_scheduler

# Training data per task: Table 1 of the paper for the vision settings, measured
# from the downloaded datasets for the NLP ones. Batch sizes are --batch_size
# for vision and hardcoded in the loaders for NLP.
SETTINGS = {
    "CIFAR-100-5": (10_000, 128),
    "CIFAR-100-20": (2_500, 128),
    "CIFAR-100-50": (1_000, 128),
    "ImageNet-R-5": (4_800, 128),
    "ImageNet-R-20": (3_000, 128),
    "ImageNet-R-50": (480, 128),
    "LSB (shortest task: cb)": (250, 32),
    "LSB (longest task: yelp)": (650_000, 32),
    "CITB (shortest task)": (142, 16),
}
EPOCHS = 10
DEFAULT_RATIO = 0.1


def total_steps(setting, epochs=EPOCHS):
    n_train, batch_size = SETTINGS[setting]
    return math.ceil(n_train / batch_size) * epochs


def _schedule(ratio, steps, base_lr=1e-5):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    scheduler = build_scheduler(
        optimizer, Namespace(lr=base_lr, warmup_ratio=ratio), steps
    )
    lrs = []
    for step in range(steps):
        scheduler(step)
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


# --- the default works everywhere -------------------------------------------


@pytest.mark.parametrize("setting", sorted(SETTINGS))
def test_default_ratio_reaches_the_base_rate_and_decays(setting):
    """One value for every setting — the property a step count could not have."""
    steps = total_steps(setting)
    lrs = _schedule(DEFAULT_RATIO, steps)

    assert max(lrs) == pytest.approx(1e-5), "the base learning rate is never reached"
    assert lrs[-1] < max(lrs), "the schedule never decays"


@pytest.mark.parametrize("setting", sorted(SETTINGS))
def test_warmup_is_the_same_fraction_whatever_the_task_length(setting):
    steps = total_steps(setting)
    lrs = _schedule(DEFAULT_RATIO, steps)

    rising = sum(1 for i in range(1, len(lrs)) if lrs[i] > lrs[i - 1]) + 1
    assert rising / steps == pytest.approx(DEFAULT_RATIO, abs=0.02)


def test_default_is_one_tenth():
    from src.args import parse_arguments

    args = parse_arguments(["--model", "ViT-B-16", "--dataset", "CIFAR100"])

    assert args.warmup_ratio == DEFAULT_RATIO


# --- the heterogeneity a step count could not cover -------------------------


def test_one_ratio_serves_both_ends_of_a_single_lsb_run():
    """`cb` and `yelp` are in the same task sequence and differ by 2500x in
    length. Any single step count is either most of `cb`'s schedule or a
    rounding error of `yelp`'s; the ratio is 10% of each.
    """
    short, long = total_steps("LSB (shortest task: cb)"), total_steps(
        "LSB (longest task: yelp)"
    )
    assert long / short > 2000  # the fixture still represents the problem

    for steps in (short, long):
        lrs = _schedule(DEFAULT_RATIO, steps)
        assert max(lrs) == pytest.approx(1e-5)
        assert lrs[-1] < max(lrs)


def test_no_single_step_count_suits_every_setting():
    """Why the flag changed from a step count to a ratio.

    79 steps is 10% of CIFAR-100-5's schedule and was the recommended value for
    it. Across the other settings the same number lands anywhere from nearly the
    whole schedule to a rounding error, so no one value could have been right
    for all of them.
    """
    fractions = {name: 79 / total_steps(name) for name in SETTINGS}

    assert max(fractions.values()) > 0.9, fractions  # swallows a whole schedule
    assert min(fractions.values()) < 0.001, fractions  # effectively no warmup
    # ... whereas the ratio is 10% of every one of them, by construction.


def test_the_ratio_gives_every_setting_the_same_fraction():
    fractions = {
        name: max(1, int(DEFAULT_RATIO * total_steps(name))) / total_steps(name)
        for name in SETTINGS
    }

    assert all(f == pytest.approx(DEFAULT_RATIO, abs=0.02) for f in fractions.values()), (
        fractions
    )


@pytest.mark.parametrize(
    "setting",
    ["CIFAR-100-20", "CIFAR-100-50", "ImageNet-R-5", "ImageNet-R-20", "ImageNet-R-50"],
)
def test_the_old_default_of_500_steps_covered_these_whole_schedules(setting):
    """Recorded so the reason for the change stays visible."""
    steps = total_steps(setting)

    assert steps <= 500


# --- scaling with the other flags -------------------------------------------


@pytest.mark.parametrize("epochs", [1, 3, 10, 30])
def test_warmup_follows_epochs_without_being_retuned(epochs):
    steps = total_steps("CIFAR-100-5", epochs=epochs)
    lrs = _schedule(DEFAULT_RATIO, steps)

    assert max(lrs) == pytest.approx(1e-5)
    assert lrs[-1] < max(lrs)


@pytest.mark.parametrize("batch_size", [32, 64, 128, 256])
def test_warmup_follows_batch_size_without_being_retuned(batch_size):
    steps = math.ceil(10_000 / batch_size) * EPOCHS
    lrs = _schedule(DEFAULT_RATIO, steps)

    assert max(lrs) == pytest.approx(1e-5)
    assert lrs[-1] < max(lrs)


# --- schedule shape ---------------------------------------------------------


def test_warmup_rises_linearly_then_cosine_decays():
    lrs = _schedule(ratio=0.1, steps=100)

    assert lrs[:10] == sorted(lrs[:10])
    assert lrs[9] == pytest.approx(1e-5)  # base rate at the end of warmup
    assert lrs[10:] == sorted(lrs[10:], reverse=True)


def test_a_very_short_schedule_still_warms_up_for_at_least_one_step():
    """int(ratio * steps) floors to 0 for a handful of steps; warmup of 0 would
    divide by zero in the linear ramp."""
    lrs = _schedule(ratio=0.1, steps=5)

    assert len(lrs) == 5
    assert all(lr > 0 for lr in lrs)


@pytest.mark.parametrize("ratio", [0.01, 0.1, 0.25, 0.5])
def test_ratio_sets_the_warmup_length(ratio):
    steps = 1000
    lrs = _schedule(ratio, steps)
    expected_warmup = int(ratio * steps)

    assert lrs[expected_warmup - 1] == pytest.approx(1e-5)
    assert lrs[expected_warmup - 2] < 1e-5


# --- one scheduler, both pipelines ------------------------------------------


def test_both_trainers_use_the_shared_builder():
    """The NLP trainer's private clamp made warmup mean 10% of the schedule
    there and a literal step count in vision."""
    import inspect

    from src import trainer
    from src.nlp import trainer_nlp

    nlp_source = inspect.getsource(trainer_nlp)
    vision_source = inspect.getsource(trainer)

    assert "build_scheduler(" in nlp_source
    assert "cosine_lr(" not in nlp_source, "the NLP trainer builds its own schedule again"
    assert vision_source.count("cosine_lr(") == 1, "only build_scheduler may call cosine_lr"


def test_no_step_count_flag_survives():
    import inspect

    from src import args, trainer
    from src.nlp import trainer_nlp

    for module in (args, trainer, trainer_nlp):
        source = inspect.getsource(module)
        assert "args.warmup_length" not in source, module.__name__
        assert '"--warmup_length"' not in source, module.__name__
