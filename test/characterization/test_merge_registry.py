"""The merge_fn dispatch surface.

This used to live in four places — the vision backend's `merge_fn_dict`, an
NLP-only `MERGE_FNS`, `__name__` string comparisons in src/eval.py, and the
label table in postprocessing/utils.py — and those copies had drifted. These
tests pin the single table that replaced them, and in particular the two
things the old duplication got wrong:

  * "average" must be an average on *both* backends (it was a raw sum on NLP);
  * a merge method's on-disk name must be its `--merge_fn` value, not whatever
    the underlying Python function happens to be called.
"""

import pytest

from conftest import build_synthetic_task_vectors
from src.merging.registry import (
    MERGE_SPECS,
    SUPPORTED_MERGE_FN_NAMES,
    apply_merge,
    get_merge_spec,
    merge_task_vectors,
)

# Every name scripts/*.sh and README.md document as a valid --merge_fn value.
CLI_DOCUMENTED_MERGE_FNS = {
    "finetune",
    "magmax",
    "ties",
    "average",
    "random_mix",
    "masked_magmax_with_targetdata",
}


def test_registry_matches_the_documented_cli_surface():
    assert set(MERGE_SPECS) == CLI_DOCUMENTED_MERGE_FNS


def test_spec_name_matches_its_key():
    """`spec.name` is what results are written under, so a key/name mismatch
    would silently scatter results across two directory names."""
    for key, spec in MERGE_SPECS.items():
        assert spec.name == key


def test_on_disk_names_are_the_cli_names():
    """No non-obvious indirection between what the user types and where the
    results land. The previous layout derived the directory from the Python
    function, which mapped magmax -> "merge_max_abs" and, least obviously,
    average -> "sum"."""
    assert {spec.name for spec in MERGE_SPECS.values()} == CLI_DOCUMENTED_MERGE_FNS


def test_only_the_target_data_merge_is_vision_only():
    assert SUPPORTED_MERGE_FN_NAMES == CLI_DOCUMENTED_MERGE_FNS - {
        "masked_magmax_with_targetdata"
    }
    assert MERGE_SPECS["masked_magmax_with_targetdata"].needs_target_data
    assert not any(
        spec.needs_target_data
        for name, spec in MERGE_SPECS.items()
        if name != "masked_magmax_with_targetdata"
    )


def test_finetune_resolves_to_the_last_task_vector():
    task_vectors = build_synthetic_task_vectors()
    spec = get_merge_spec("finetune")

    assert spec.fn is None  # "no merging happens" is expressed by fn=None
    assert apply_merge(spec, task_vectors) is task_vectors[-1]
    assert merge_task_vectors("finetune", task_vectors) is task_vectors[-1]


def test_average_is_the_mean_on_both_backends():
    """The defect the unified registry fixes.

    Both backends used to map "average" to the bare builtin `sum`; only the
    vision path then divided by the task count (in src/eval.py), so the NLP
    path's "average" was len(task_vectors)x too large. The division now lives
    in apply_merge, which both backends go through.
    """
    task_vectors = build_synthetic_task_vectors()
    key = "layer1.weight"
    expected_mean = sum(tv.vector[key] for tv in task_vectors) / len(task_vectors)

    via_apply_merge = apply_merge(get_merge_spec("average"), task_vectors)
    via_nlp_entry_point = merge_task_vectors("average", task_vectors)

    assert via_apply_merge.vector[key].allclose(expected_mean)
    assert via_nlp_entry_point.vector[key].allclose(expected_mean)


# random_mix is excluded: it draws from the global RNG, so two calls differ by
# construction. That apply_merge leaves it alone is covered by the golden test.
@pytest.mark.parametrize("name", ["magmax", "ties"])
def test_non_averaged_merges_are_not_rescaled(name):
    """Only "average" carries the post-merge division; a stray `averaged=True`
    would quietly rescale a baseline."""
    spec = get_merge_spec(name)
    key = "layer1.weight"

    assert not spec.averaged
    direct = spec.fn(build_synthetic_task_vectors())
    via_apply_merge = apply_merge(spec, build_synthetic_task_vectors())

    assert via_apply_merge.vector[key].allclose(direct.vector[key])


def test_random_mix_is_not_rescaled():
    assert not get_merge_spec("random_mix").averaged


def test_unknown_merge_fn_raises_with_the_available_names():
    with pytest.raises(ValueError, match="Unknown merge_fn"):
        get_merge_spec("no_such_merge_fn")


def test_nlp_backends_reject_the_vision_only_merge():
    with pytest.raises(NotImplementedError):
        merge_task_vectors("masked_magmax_with_targetdata", build_synthetic_task_vectors())


def test_removed_merge_fns_are_gone_from_every_surface():
    """select_one_task_vector and the "hpo" preference-vector search were
    deleted as unused by the paper's experiments. Asserted here so a future
    refactor cannot quietly resurrect a half-wired entry."""
    from src.merging import task_vectors

    assert not hasattr(task_vectors, "select_one_task_vector")
    assert "select_one_task_vector" not in MERGE_SPECS
    # the no-op finetune() that existed only to carry a __name__ is gone too
    assert not hasattr(task_vectors, "finetune")


def _read_competitor_dict() -> dict:
    """Extract COMPETITOR_ALL_DICT from postprocessing/utils.py *without*
    importing it — that module pulls in the whole plotting stack (seaborn,
    pandas), which is not part of this project's declared dependencies."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "postprocessing" / "utils.py"
    ).read_text()
    for node in ast.parse(source).body:
        targets = getattr(node, "targets", [])
        if targets and getattr(targets[0], "id", None) == "COMPETITOR_ALL_DICT":
            return ast.literal_eval(node.value)
    raise AssertionError("COMPETITOR_ALL_DICT not found in postprocessing/utils.py")


def test_postprocessing_labels_stay_in_sync_with_the_registry():
    """postprocessing/ deliberately does NOT import src/ — it would drag torch
    and open_clip into the analysis notebooks. The two are kept consistent by
    this test instead of by a code dependency.
    """
    COMPETITOR_ALL_DICT = _read_competitor_dict()

    # The target-data merge appears once per similarity metric, as
    # "<merge_fn>-<metric>"; every other merge_fn appears under its bare name.
    plotted = {key.split("-")[0] for key in COMPETITOR_ALL_DICT}
    assert plotted == set(MERGE_SPECS), (
        f"postprocessing and the registry disagree: "
        f"only in postprocessing={plotted - set(MERGE_SPECS)}, "
        f"only in registry={set(MERGE_SPECS) - plotted}"
    )

    for name, spec in MERGE_SPECS.items():
        if spec.needs_target_data:
            continue
        assert COMPETITOR_ALL_DICT[name] == spec.label
