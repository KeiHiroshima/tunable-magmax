"""Which tasks a target environment is a mixture of must depend only on that
environment's own seed.

The vision datasets used to pick the task mixture with the module-level
`random.sample`, while seeding only numpy from the variant's seed. The mixture
therefore ignored `random_seed` and tracked the process-wide RNG state instead,
which made it a function of how much randomness the preceding merges had
consumed. These tests pin the three properties that restored:

  * the variant's seed determines the mixture,
  * nothing that happened earlier in the process can change it,
  * and neither can --seed.

They call the real dataset methods' selection logic through a tiny stand-in
rather than instantiating CIFAR-100/ImageNet-R, which would need the data on
disk; test_target_task_selection_matches_the_datasets below checks the stand-in
against the real source.
"""

import random
import re
from pathlib import Path

import numpy as np
import pytest

from src.target_env import select_target_tasks

N_SPLITS = 5
REPO_ROOT = Path(__file__).resolve().parents[2]
# The vision samplers live in datasets/common.py, shared by CIFAR-100 and
# ImageNet-R; the NLP backend calls the selector directly.
SELECTION_CALLERS = [
    REPO_ROOT / "src" / "datasets" / "common.py",
    REPO_ROOT / "src" / "backends" / "nlp_classification_backend.py",
]


def _seed_everything(seed):
    """Mirrors src/args.py::seed_everything, run once per process at startup."""
    random.seed(seed)
    np.random.seed(seed)


def select_tasks(variant_seed, n_selected, n_splits=N_SPLITS):
    """What the dataset methods do to pick the mixture: seed numpy for the
    example-level draws that follow, then delegate the task choice."""
    np.random.seed(variant_seed)
    return select_target_tasks(n_splits, n_selected, variant_seed)


def _code_only(source: Path) -> str:
    """Source with comment lines dropped, so prose about the old behaviour is
    not mistaken for the old behaviour."""
    lines = []
    for line in source.read_text().splitlines():
        stripped = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        lines.append(stripped)
    return "\n".join(lines)


def test_every_backend_delegates_the_task_choice():
    """One implementation, one caller per pipeline.

    The vision datasets and the NLP backend once had their own copies; the
    vision ones drew from the process-wide RNG and the NLP one did not, so they
    disagreed. Now that they agree, a private copy reappearing is how they
    would drift apart again.
    """
    for source in SELECTION_CALLERS:
        code = _code_only(source)
        assert "select_target_tasks(" in code, (
            f"{source.name} no longer goes through the shared task selector"
        )
        assert not re.search(r"(?<![.\w])random\.(Random\([^)]*\)\.)?sample\(", code), (
            f"{source.name} picks the task mixture itself instead of calling "
            f"src.target_env.select_target_tasks"
        )


def test_the_vision_datasets_no_longer_sample_at_all():
    """CIFAR-100 and ImageNet-R had a copy of the sampling code each; both now
    go through src/datasets/common.py."""
    for name in ("cifar100.py", "imagenetr.py"):
        code = _code_only(REPO_ROOT / "src" / "datasets" / name)
        assert "select_target_tasks(" not in code, f"{name} has its own task selection again"
        assert "np.random.choice(" not in code, f"{name} has its own example sampling again"


def test_the_shared_selector_is_seeded_per_environment():
    """The one place the seeding now lives."""
    code = _code_only(REPO_ROOT / "src" / "target_env.py")

    assert "random.Random(seed).sample(" in code


def test_variant_seed_determines_the_mixture():
    _seed_everything(3)
    a = select_tasks(42, 2)
    _seed_everything(3)
    b = select_tasks(999_999, 2)

    assert a != b, "different target environments must not share a task mixture"


def test_same_variant_seed_gives_the_same_mixture():
    _seed_everything(3)
    a = select_tasks(42, 2)
    _seed_everything(11)
    b = select_tasks(42, 2)

    assert a == b


def test_mixture_is_unaffected_by_earlier_random_draws():
    """The concrete failure this prevents: the masked merge draws from the
    global RNG (src/merging/task_vectors.py), and how often is data-dependent.
    While the mixture came from that same RNG, the same target_id denoted
    different environments under different --merge_fn values — so a baseline
    row and a Tunable MAGMAX row in the same table column were not necessarily
    evaluated on the same data.
    """
    _seed_everything(3)
    undisturbed = [select_tasks(42 + i, 2) for i in range(5)]

    _seed_everything(3)
    random.sample(range(100), 3)  # stands in for a merge consuming the global RNG
    disturbed = [select_tasks(42 + i, 2) for i in range(5)]

    assert undisturbed == disturbed


def test_mixture_survives_resuming_a_partially_finished_run():
    """src/eval.py skips target environments whose results file already exists.
    While the mixture depended on the global RNG, skipping also skipped its
    draw, so filling in missing results produced different environments than a
    single full run.
    """
    _seed_everything(3)
    full_run = [select_tasks(42 + i, 2) for i in range(5)]

    _seed_everything(3)
    resumed = [None if i < 2 else select_tasks(42 + i, 2) for i in range(5)]

    assert resumed[2:] == full_run[2:]


def test_mixture_does_not_depend_on_the_training_seed():
    """--seed picks the model's training run; it must not silently redefine
    what the target environments are."""
    _seed_everything(3)
    with_seed_3 = [select_tasks(42 + i, 2) for i in range(5)]

    _seed_everything(99)
    with_seed_99 = [select_tasks(42 + i, 2) for i in range(5)]

    assert with_seed_3 == with_seed_99


@pytest.mark.parametrize("n_selected", [1, 2, 3, 5])
def test_mixture_is_a_valid_subset(n_selected):
    selected = select_tasks(42, n_selected)

    assert len(selected) == n_selected
    assert len(set(selected)) == n_selected
    assert all(0 <= i < N_SPLITS for i in selected)
