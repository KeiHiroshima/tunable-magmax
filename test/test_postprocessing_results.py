"""postprocessing/results.py: the backend-agnostic result loader added so
build_comparison_table.py can read both vision's --results_db layout and
NLP's beside-the-checkpoints layout through one aggregation function
(load_overall_accuracy) instead of a second copy of fetch_stats.

Also pins that results.py has no matplotlib/seaborn import, and that the
legacy vision-only aggregators re-exported from postprocessing/utils.py are
unaffected by the split.
"""

import json
from pathlib import Path

import pytest

from postprocessing.build_comparison_table import _default_competitors, build_comparison_table
from postprocessing.results import (
    load_overall_accuracy,
    nlp_result_path,
    parse_competitor_key,
    resolve_dir_path,
    vision_result_path,
)


def test_default_nlp_competitors_include_the_bare_proposed_method_key():
    """LSB's competitor key has no "-<similarity_metric>" suffix (see
    nlp_result_path's docstring), so a naive "drop every '-<metric>' key"
    default would drop the proposed method entirely for --backend nlp."""
    competitors = _default_competitors("nlp")

    assert "masked_magmax_with_targetdata" in competitors
    assert all("-" not in key for key in competitors)


def test_default_vision_competitors_include_every_similarity_metric():
    competitors = _default_competitors("vision")

    assert "masked_magmax_with_targetdata-ot_embedded" in competitors
    assert "masked_magmax_with_targetdata-labels" in competitors


def _write_result(path: Path, overall_accuracy: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"overall_accuracy": overall_accuracy}))


# --- vision_result_path: must resolve exactly like the legacy helpers ------


def test_vision_result_path_matches_the_legacy_helpers(tmp_path):
    dir_path_shared = str(tmp_path)
    key = "masked_magmax_with_targetdata-ot_embedded"

    merge_fn, similarity_metric, _, metric_name_path = parse_competitor_key(key)
    dir_path = resolve_dir_path(dir_path_shared, key, None)
    expected = (
        f"{dir_path}/{merge_fn}/{metric_name_path}"
        f"{merge_fn}_lambda0.5_{similarity_metric}target3_seed7.json"
    )

    assert vision_result_path(dir_path_shared, 0.5)(key, 3, 7) == expected


def test_vision_result_path_plain_merge_fn_has_no_metric_segment(tmp_path):
    path = vision_result_path(str(tmp_path), 0.5)("magmax", 1, 3)
    assert path == f"{tmp_path}/magmax/magmax_lambda0.5_target1_seed3.json"


# --- nlp_result_path: seed changes the directory, not just the filename ----


def test_nlp_result_path_bakes_seed_into_the_checkpoint_directory(tmp_path):
    path_fn = nlp_result_path(
        model="bert-base-uncased",
        dataset="LSB",
        epochs=3,
        taskseq_pattern="A",
        base_dir=str(tmp_path),
    )

    path_seed3 = path_fn("average", 1, 3)
    path_seed4 = path_fn("average", 1, 4)

    assert path_seed3 != path_seed4
    assert "seed:3" in path_seed3 and path_seed3.endswith("average/target1_seed3.json")
    assert "seed:4" in path_seed4 and path_seed4.endswith("average/target1_seed4.json")


# --- load_overall_accuracy / build_comparison_table -------------------------


def test_load_overall_accuracy_is_tidy_long_form(tmp_path):
    path_fn = lambda key, target_id, seed: str(tmp_path / f"{key}_{target_id}_{seed}.json")  # noqa: E731
    for seed in (3, 4):
        _write_result(tmp_path / f"magmax_1_{seed}.json", overall_accuracy=0.5 + 0.1 * seed)

    df = load_overall_accuracy(path_fn, {"magmax": "MAGMAX"}, target_ids=[1], seed_list=[3, 4])

    assert sorted(df["seed"]) == [3, 4]
    assert set(df["competitor"]) == {"MAGMAX"}
    assert df.set_index("seed").loc[3, "overall_accuracy"] == pytest.approx(0.8)
    assert df.set_index("seed").loc[4, "overall_accuracy"] == pytest.approx(0.9)


def test_build_comparison_table_averages_variants_then_seeds(tmp_path):
    """Two target_ids (variants of the same 2-task-balanced config), two
    seeds: each seed's number should be the mean over the two variants, and
    the table cell should be the mean (+/- std) of that over seeds."""
    path_fn = lambda key, target_id, seed: str(  # noqa: E731
        tmp_path / f"{key}_{target_id}_{seed}.json"
    )
    accuracies = {(1, 3): 0.6, (2, 3): 0.8, (1, 4): 0.7, (2, 4): 0.9}
    for (target_id, seed), acc in accuracies.items():
        _write_result(tmp_path / f"magmax_{target_id}_{seed}.json", acc)

    dataset_configs = [
        {
            "num_task_to_be_fetched": 2,
            "ratio_task_to_be_fetched": [0.5, 0.5],
            "variants": [{"target_id": 1, "random_seed": 1}, {"target_id": 2, "random_seed": 2}],
        },
        {"num_task_to_be_fetched": -1, "ratio_task_to_be_fetched": [-1], "variants": [{"target_id": 99, "random_seed": 1}]},
    ]

    table = build_comparison_table(path_fn, {"magmax": "MAGMAX"}, dataset_configs, seeds=[3, 4])

    # seed 3: mean(0.6, 0.8) = 0.7; seed 4: mean(0.7, 0.9) = 0.8 -> mean 0.75, std over 2 seeds.
    [column] = table.columns  # the "-1 / all tasks" entry must be skipped
    assert column == "2-task (0.5, 0.5)"
    assert table.loc["MAGMAX", column] == "0.750 ± 0.071"
