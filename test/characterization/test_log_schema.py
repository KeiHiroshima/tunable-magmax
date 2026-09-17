"""Pins the results-JSON contract.

postprocessing/ and the paper's figures read these files, so the key names and
the file layout are a public interface even though nothing type-checks them.
Phase 5 moves the target-environment plumbing around; these tests make sure the
output files keep the same shape.
"""

import json
from argparse import Namespace

import pytest

from src.eval import _build_merged_encoder, _save_target_eval_results
from src.merging.registry import get_merge_spec
from src.merging.task_vector import TaskVector


def _args(**overrides):
    args = Namespace(
        model="ViT-B-16",
        dataset="CIFAR100",
        merge_fn="magmax",
        similarity_metric="labels",
        seed=3,
        coeff=0.5,
        target_id=0,
        device="cpu",
        results_db=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def real_task_vectors(tiny_checkpoints):
    pretrained_path, finetuned_paths = tiny_checkpoints
    return pretrained_path, [TaskVector(pretrained_path, p) for p in finetuned_paths]


def test_merge_metadata_keys_for_a_plain_merge_fn(real_task_vectors):
    pretrained_path, task_vectors = real_task_vectors
    _, log_data = _build_merged_encoder(
        task_vectors, get_merge_spec("magmax"), _args(), pretrained_path, 0.5
    )

    # "merging_function" is the --merge_fn value, not the Python function name
    # it used to be ("merge_max_abs").
    assert log_data == {
        "merging_function": "magmax",
        "scaling_coefficient": 0.5,
    }


def test_merge_metadata_keys_for_the_finetune_baseline(real_task_vectors):
    """The "finetune" baseline (no merging: keep the last task's model) logs
    under its plain CLI name. It used to write "finetune_only" here, which
    matched neither the CLI value nor the results directory."""
    pretrained_path, task_vectors = real_task_vectors
    spec = get_merge_spec("finetune")

    _, log_data = _build_merged_encoder(
        task_vectors, spec, _args(merge_fn="finetune"), pretrained_path, 0.5
    )

    assert log_data == {
        "merging_function": "finetune",
        "scaling_coefficient": 0.5,
    }


def test_average_is_divided_by_the_task_count(real_task_vectors):
    """The division now comes from the registry (spec.averaged) rather than an
    `args.merge_fn == "average"` check inside _build_merged_encoder, so the NLP
    backend gets it too. See test_merge_registry.py."""
    from src.utils import torch_load

    pretrained_path, task_vectors = real_task_vectors
    key = "layer1.weight"

    merged, _ = _build_merged_encoder(
        task_vectors,
        get_merge_spec("average"),
        _args(merge_fn="average"),
        pretrained_path,
        1.0,  # scaling_coef=1.0 so the delta is applied in full
    )

    pretrained = torch_load(pretrained_path).state_dict()[key]
    summed_delta = sum(tv.vector[key] for tv in task_vectors)
    merged_weights = merged.state_dict()[key]

    assert merged_weights.allclose(pretrained + summed_delta / len(task_vectors))
    # ... and emphatically not the un-divided sum, which is what the NLP path
    # produced before the registry was unified.
    assert not merged_weights.allclose(pretrained + summed_delta)


def test_results_file_location_and_name(tmp_path):
    args = _args(results_db=str(tmp_path))
    _save_target_eval_results(
        {"overall_accuracy": 0.5}, get_merge_spec("magmax"), args, "labels/", "result.json"
    )

    written = tmp_path / "magmax" / "labels" / "result.json"
    assert written.exists()
    assert json.loads(written.read_text()) == {"overall_accuracy": 0.5}


def test_results_file_without_similarity_subdirectory(tmp_path):
    args = _args(results_db=str(tmp_path))
    _save_target_eval_results(
        {"overall_accuracy": 0.5}, get_merge_spec("magmax"), args, "", "r.json"
    )

    assert (tmp_path / "magmax" / "r.json").exists()


def test_preexisting_file_content_wins_over_the_new_log_data(tmp_path):
    """Surprising but load-bearing: `log_data.update(existing)` means values
    already in the file are NOT overwritten. That is how num_unaligned /
    num_params_all — written earlier by the merge function itself — survive
    the evaluation write. An "obvious cleanup" to existing.update(log_data)
    would silently discard them.
    """
    args = _args(results_db=str(tmp_path))
    out_path = tmp_path / "magmax" / "r.json"
    out_path.parent.mkdir(parents=True)
    out_path.write_text(json.dumps({"num_unaligned": {"task_1": 7}, "overall_accuracy": 0.1}))

    _save_target_eval_results(
        {"overall_accuracy": 0.9}, get_merge_spec("magmax"), args, "", "r.json"
    )

    result = json.loads(out_path.read_text())
    assert result["num_unaligned"] == {"task_1": 7}  # preserved
    assert result["overall_accuracy"] == 0.1  # existing wins, new value dropped


def test_merge_and_eval_write_to_the_same_results_file(tmp_path, monkeypatch):
    """The proposed method's results are assembled by two writers.

    merge_max_abs_masked_with_targetdata records num_unaligned/num_params_all
    while merging; src/eval.py then adds the accuracies to the same file and
    keeps what is already there. If the two spell the path differently the
    merge diagnostics end up in a directory of their own and never reach the
    results — which is what happened when the layout moved from Python
    function names to --merge_fn values and only one writer was updated.
    """
    import json as _json
    from pathlib import Path

    from src.merging import task_vectors

    args = _args(
        results_db=str(tmp_path),
        merge_fn="masked_magmax_with_targetdata",
        similarity_metric="labels",
    )
    spec = get_merge_spec("masked_magmax_with_targetdata")

    # what the merge function writes
    merge_dir = Path(args.results_db) / args.merge_fn / args.similarity_metric
    merge_dir.mkdir(parents=True)
    merge_file = (
        merge_dir
        / f"{args.merge_fn}_lambda{args.coeff}_{args.similarity_metric}"
        f"_target{args.target_id}_seed{args.seed}.json"
    )
    merge_file.write_text(_json.dumps({"num_unaligned": {"task_1": 5}, "num_params_all": 99}))

    # what eval.py writes afterwards, via the shared spec name
    file_name = (
        f"{spec.name}_lambda{args.coeff}_{args.similarity_metric}_"
        f"target{args.target_id}_seed{args.seed}.json"
    )
    _save_target_eval_results(
        {"overall_accuracy": 0.9}, spec, args, f"{args.similarity_metric}/", file_name
    )

    written = _json.loads(merge_file.read_text())
    assert written["num_unaligned"] == {"task_1": 5}, "the merge diagnostics were lost"
    assert written["num_params_all"] == 99
    assert "overall_accuracy" in written, "the two writers disagree on the path"
    assert [p.name for p in Path(args.results_db).iterdir()] == [args.merge_fn]


def test_merge_function_derives_its_log_path_from_the_cli_name():
    """Guards the spelling itself: the function used to hardcode its own name."""
    import inspect

    from src.merging import task_vectors

    source = inspect.getsource(task_vectors.merge_max_abs_masked_with_targetdata)

    assert 'Path(args.results_db) / args.merge_fn' in source
    assert '"merge_max_abs_masked_with_targetdata"' not in source
