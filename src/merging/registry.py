"""The single source of truth for `--merge_fn`.

Before this table the same knowledge was spelled out in four places — the
vision backend's own `merge_fn_dict`, an NLP-only `MERGE_FNS`, a chain of
`merging_f.__name__` string comparisons in src/eval.py, and the label table in
postprocessing/utils.py — so adding or changing a merge method meant editing
four files, and the four had already drifted apart (NLP's "average" was really
a sum, because only the vision copy divided by the task count).

Everything a caller needs now hangs off one MergeSpec:

    spec = get_merge_spec(args.merge_fn)
    merged = apply_merge(spec, task_vectors)

Output naming: results are written under the `--merge_fn` value itself
(`.../magmax/`, `.../average/`). The old layout used the *Python function*
name instead, which produced `merge_max_abs` for `magmax` and — least
obviously — `sum` for `average`. Result files written before this change keep
the old directory names; see the note in postprocessing/utils.py.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from src.merging.task_vectors import (
    merge_max_abs,
    merge_max_abs_masked_with_targetdata,
    merge_rnd_mix,
    ties,
)


@dataclass(frozen=True)
class MergeSpec:
    """One `--merge_fn` value and everything the pipeline needs to know about it.

    name        the CLI value, and the directory results are written under.
    fn          list[TaskVector] -> TaskVector. None means "no merging happens":
                the caller supplies the vector itself (see apply_merge).
    label       display name used by postprocessing/ for the paper's tables.
    averaged    divide the merged vector by the number of task vectors. Kept as
                a flag rather than folded into `fn` because `fn` is the builtin
                `sum`, which several callers rely on being exactly that.
    needs_target_data  the merge depends on target-environment data, so it
                cannot be called with just (task_vectors).
    nlp_supported  whether the NLP backends can run it.
    """

    name: str
    fn: Optional[Callable]
    label: str
    averaged: bool = False
    needs_target_data: bool = False
    nlp_supported: bool = True


# Order matters: postprocessing/utils.py assigns plot colors by position.
MERGE_SPECS: dict[str, MergeSpec] = {
    "finetune": MergeSpec("finetune", None, "Baseline"),
    "random_mix": MergeSpec("random_mix", merge_rnd_mix, "Random Mix"),
    "average": MergeSpec("average", sum, "Average", averaged=True),
    "ties": MergeSpec("ties", ties, "TIES-Merging"),
    "magmax": MergeSpec("magmax", merge_max_abs, "MAGMAX"),
    "masked_magmax_with_targetdata": MergeSpec(
        "masked_magmax_with_targetdata",
        merge_max_abs_masked_with_targetdata,
        "Tunable MAGMAX",
        needs_target_data=True,
        # Needs a whole target-environment-construction flow (which tasks, in
        # what ratio, sampled eval data) rather than a plain
        # task_vectors -> TaskVector call. The NLP backends build that flow
        # themselves; see nlp_classification_backend::_merge_and_evaluate_masked
        # and src/nlp/target_data.py.
        nlp_supported=False,
    ),
}


def get_merge_spec(merge_fn_name: str) -> MergeSpec:
    try:
        return MERGE_SPECS[merge_fn_name]
    except KeyError:
        raise ValueError(
            f"Unknown merge_fn {merge_fn_name!r}. Available: {sorted(MERGE_SPECS)}"
        ) from None


def apply_merge(spec: MergeSpec, task_vectors: list, **kwargs):
    """Merge `task_vectors` according to `spec`.

    This is the only place the post-merge scaling lives, so both backends get
    it — the bug this replaces was the vision path dividing "average" by the
    task count while the NLP path silently returned the raw sum.
    """
    if spec.fn is None:
        # "finetune" = the CL baseline: no merging at all, just keep the model
        # as it came out of the last task.
        return task_vectors[-1]

    merged = spec.fn(task_vectors, **kwargs)
    if spec.averaged:
        merged = merged / len(task_vectors)
    return merged


SUPPORTED_MERGE_FN_NAMES = {
    name for name, spec in MERGE_SPECS.items() if spec.nlp_supported
}


def merge_task_vectors(merge_fn_name: str, task_vectors: list):
    """Merge for the NLP backends, which only ever need the (task_vectors) form."""
    spec = get_merge_spec(merge_fn_name)
    if not spec.nlp_supported:
        raise NotImplementedError(
            f"merge_fn={merge_fn_name!r} not supported for NLP backends "
            f"(supported: {sorted(SUPPORTED_MERGE_FN_NAMES)})."
        )
    return apply_merge(spec, task_vectors)
