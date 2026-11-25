"""ImageNet-R preparation (docs/imagenetr_plan.md §3.1, §3.2).

- The train/test split is reused, never re-made: without it, loading fails and
  nothing is moved.
- Subsets are built once, beforehand, and atomically; the dataset only reads
  finished ones, so parallel jobs never race to create them.
- A merge-only job checks the very files fine-tuning writes and merging reads.

CPU only, no downloads: the images are empty files (ImageFolder only lists
them).
"""

import json
import os

import pytest

import src.config
import src.paths
from src.datasets.imagenetr import SUBSET_CONFIG_FILE, _ImageNetR

SUBSET = {"domains": ["art", "sketch"], "classes": ["ant", "bee"]}


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


@pytest.fixture
def root(tmp_path):
    """<root>/imagenet-r/{train,test}/<class>/<domain>_<n>.jpg, with one
    domain outside SUBSET so the filtering shows."""
    for split, n in (("train", 3), ("test", 1)):
        for c in ("ant", "bee", "cat"):
            for domain in ("art", "sketch", "toy"):
                for i in range(n):
                    _touch(tmp_path / "imagenet-r" / split / c / f"{domain}_{i}.jpg")
    return str(tmp_path)


def _subset(root, train=True):
    return _ImageNetR.subset_path(root, SUBSET, train)


# --- the split ------------------------------------------------------------------


def test_missing_split_raises_and_moves_nothing(tmp_path):
    # The layout before a split: images directly under imagenet-r/<class>/.
    _touch(tmp_path / "imagenet-r" / "ant" / "art_0.jpg")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    with pytest.raises(RuntimeError, match="split not found"):
        _ImageNetR(str(tmp_path), train=True, subset_config=SUBSET)
    with pytest.raises(RuntimeError, match="split not found"):
        _ImageNetR.build_subset(str(tmp_path), SUBSET, train=True)

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


# --- building the subset ----------------------------------------------------------


def test_build_subset_links_only_the_subset(root):
    path = _ImageNetR.build_subset(root, SUBSET, train=True)

    assert path == _subset(root)
    assert sorted(os.listdir(path)) == ["ant", "bee", SUBSET_CONFIG_FILE]
    for c in ("ant", "bee"):
        names = sorted(os.listdir(os.path.join(path, c)))
        assert names == [f"{d}_{i}.jpg" for d in ("art", "sketch") for i in range(3)]
        assert all(os.path.islink(os.path.join(path, c, n)) for n in names)
    with open(os.path.join(path, SUBSET_CONFIG_FILE)) as f:
        assert json.load(f) == SUBSET
    # Nothing temporary left beside it.
    assert os.listdir(os.path.dirname(path)) == ["train"]


def test_dataset_reads_only_a_finished_subset(root):
    with pytest.raises(RuntimeError, match="prepare_imagenetr"):
        _ImageNetR(root, train=False, subset_config=SUBSET)

    _ImageNetR.build_subset(root, SUBSET, train=False)
    data = _ImageNetR(root, train=False, subset_config=SUBSET).data

    assert data.classes == ["ant", "bee"]
    assert len(data) == 4  # 2 classes x 2 domains x 1 image


def test_an_unfinished_subset_is_rejected_then_rebuilt(root):
    # What older code left when interrupted: links, but no config.json.
    stale = _subset(root)
    os.makedirs(os.path.join(stale, "ant"))
    os.symlink("/nonexistent", os.path.join(stale, "ant", "art_0.jpg"))

    with pytest.raises(RuntimeError, match="missing or unfinished"):
        _ImageNetR(root, train=True, subset_config=SUBSET)

    _ImageNetR.build_subset(root, SUBSET, train=True)
    data = _ImageNetR(root, train=True, subset_config=SUBSET).data
    assert len(data) == 12
    assert all(os.path.exists(p) for p, _ in data.samples)


def test_an_interrupted_build_leaves_nothing_that_looks_finished(root, monkeypatch):
    real_symlink = os.symlink
    calls = []

    def failing_symlink(src, dst):
        calls.append(dst)
        if len(calls) == 3:
            raise OSError("interrupted")
        real_symlink(src, dst)

    monkeypatch.setattr(os, "symlink", failing_symlink)
    with pytest.raises(OSError, match="interrupted"):
        _ImageNetR.build_subset(root, SUBSET, train=True)
    monkeypatch.setattr(os, "symlink", real_symlink)

    assert not os.path.exists(_subset(root))
    # The next build starts over and cleans up the leftover.
    _ImageNetR.build_subset(root, SUBSET, train=True)
    assert _ImageNetR.is_subset_complete(_subset(root))
    assert os.listdir(os.path.dirname(_subset(root))) == ["train"]


def test_a_finished_subset_is_reused_as_it_is(root):
    path = _ImageNetR.build_subset(root, SUBSET, train=True)
    marker = os.path.join(path, "marker")
    _touch(marker)

    assert _ImageNetR.build_subset(root, SUBSET, train=True) == path
    assert os.path.exists(marker)


# --- merge-only jobs: checking the checkpoints ------------------------------------


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    base = tmp_path / "base"
    monkeypatch.setattr(src.paths, "BASE_DIR", str(base))
    monkeypatch.setattr(src.config, "BASE_DIR", str(base))
    _touch(src.config.get_zeroshot_checkpoint("ViT-B-16"))
    return base


@pytest.mark.parametrize(
    "variant,scope,name",
    [
        ("full", "ImageNetR-5", "finetuned"),
        ("lora", "ImageNetR-5-lora", "adapter"),
        ("lora-reuse", "ImageNetR-5-lora-reuse", "adapter"),
    ],
)
def test_check_looks_where_finetuning_writes(base_dir, variant, scope, name):
    from src.check_checkpoints import missing_checkpoints, run_args

    args = run_args(variant, "ImageNetR", 5, 3, 10)
    run_dir = (base_dir / "checkpoints" / "ViT-B-16" / "sequential_finetuning"
               / "class_incremental" / scope / "ft-pattern_A-epochs-10-seed:3")
    expected = [str(run_dir / f"{name}_{i}.pt") for i in range(5)]

    assert missing_checkpoints(args) == expected

    for p in expected:
        _touch(p)
    assert missing_checkpoints(args) == []

    os.remove(expected[2])
    assert missing_checkpoints(args) == [expected[2]]


def test_check_main_fails_on_a_missing_split(base_dir, capsys):
    from src.check_checkpoints import main

    assert main(["full", "ImageNetR", "5", "3", "10"]) == 1
    assert "5 checkpoint(s) missing" in capsys.readouterr().out
    assert main(["no-such-variant", "ImageNetR", "5", "3", "10"]) == 2
