"""Prepare ImageNet-R once, before any experiment runs on it.

    uv run python -m src.datasets.prepare_imagenetr

1. Checks that the existing train/test split is at
   $MAGMAX_DATA_DIR/imagenet-r/{train,test} (it is reused, never re-made).
2. Creates the train and test subsets of symlinks that the dataset reads
   (imagenet-r_subsets/<md5>/). The experiment jobs run in parallel and only
   read them, so they never race to create them (src/datasets/imagenetr.py).
3. Prints the class and image counts, and per T the smallest task's train and
   test sizes: a task with fewer train images than --num_train_data_each_task
   (500) or too few test images is sampled short, with a warning, during
   merging.
4. Checks that the checkpoints to be reused are all there (by default full
   fine-tuning, T in {5, 20, 50}, seeds 3/4/5; src/check_checkpoints.py).

Exits 1 if anything is missing. See docs/imagenetr_plan.md §2 for what has to
be in place on the machine first.
"""

import argparse
import sys

import numpy as np

from src.check_checkpoints import VARIANTS, missing_checkpoints, run_args
from src.config import BASE_DIR, DATA_DIR
from src.datasets.common import get_subset_indices_with_classes, get_task_classes
from src.datasets.imagenetr import ImageNetR, _ImageNetR


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_dir", default=DATA_DIR,
                        help="where imagenet-r/ lives (default: $MAGMAX_DATA_DIR)")
    parser.add_argument("--n_splits_list", default="5 20 50")
    parser.add_argument("--seeds", default="3 4 5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--check_variants", default="full",
                        help="whose checkpoints to check, space-separated "
                             f"({', '.join(VARIANTS)}); empty to skip")
    parser.add_argument("--num_train_data_each_task", type=int, default=500)
    return parser.parse_args(argv)


def _class_counts(dataset):
    return np.bincount(np.array(dataset.targets), minlength=len(dataset.classes))


def main(argv=None):
    opts = _parse(argv)
    ok = True

    print(f"data dir: {opts.data_dir}")
    try:
        _ImageNetR.require_split(opts.data_dir)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    splits = {}
    for train in (True, False):
        name = "train" if train else "test"
        path = _ImageNetR.build_subset(opts.data_dir, train=train)
        splits[name] = _ImageNetR(opts.data_dir, train=train).data
        print(f"{name} subset: {path}")

    print("\n== data ==")
    counts = {name: _class_counts(ds) for name, ds in splits.items()}
    for name, c in counts.items():
        print(f"{name}: {len(splits[name].classes)} classes, {c.sum()} images, "
              f"per class min {c.min()} / median {int(np.median(c))} / max {c.max()}")
        if len(splits[name].classes) != len(_ImageNetR.CLASSES):
            print(f"ERROR: expected {len(_ImageNetR.CLASSES)} classes")
            ok = False
        if (c == 0).any():
            print(f"WARNING: {int((c == 0).sum())} empty classes in {name}")

    order = ImageNetR.default_class_order
    for t in (int(x) for x in opts.n_splits_list.split()):
        sizes = {
            name: [len(get_subset_indices_with_classes(splits[name], get_task_classes(order, t, i)))
                   for i in range(t)]
            for name in splits
        }
        short = sum(n < opts.num_train_data_each_task for n in sizes["train"])
        print(f"T={t}: per task train min {min(sizes['train'])}, test min {min(sizes['test'])} "
              f"/ median {int(np.median(sizes['test']))}; "
              f"{short} task(s) with fewer than {opts.num_train_data_each_task} train images")

    variants = opts.check_variants.split()
    if variants:
        print(f"\n== checkpoints to reuse ({BASE_DIR}) ==")
    for variant in variants:
        for t in (int(x) for x in opts.n_splits_list.split()):
            for seed in (int(s) for s in opts.seeds.split()):
                args = run_args(variant, "ImageNetR", t, seed, opts.epochs)
                missing = missing_checkpoints(args)
                status = "ok" if not missing else f"{len(missing)} MISSING"
                print(f"{variant} ImageNetR-{t} seed {seed}: {status}")
                for p in missing[:3]:
                    print(f"  {p}")
                if len(missing) > 3:
                    print(f"  ... and {len(missing) - 3} more")
                ok = ok and not missing

    print("\n" + ("ready" if ok else "NOT ready: fix the above before launching"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
