"""Small shared lookup for merge_fn names -> functions, used by the NLP
backends only (src/backends/vision_backend.py keeps its own copy of this
dict exactly as it was in the original merge_for_targetdata.py, unchanged).

Deliberately excludes two entries from the original merge_fn_dict:
- "masked_magmax_with_targetdata": doesn't go through this single-call
  registry at all for NLP — it needs a whole target-environment-construction
  flow (which tasks, in what ratio, sampled eval data), not just a
  task_vectors -> TaskVector function. See
  src/backends/nlp_classification_backend.py::_merge_and_evaluate_masked and
  src/nlp/target_data.py.
- "select_one_task_vector": needs vision-specific args (a classification
  head + target_dataset_meta) not available outside the vision pipeline.
  Candidate for a follow-up if needed.
"""

from src.merging.task_vectors import merge_max_abs, merge_rnd_mix, ties

MERGE_FNS = {
    "magmax": merge_max_abs,
    "ties": ties,
    "random_mix": merge_rnd_mix,
    "average": sum,
}
# "finetune" (= use the last task's vector, no merging) is handled as a
# special case by callers, matching src/eval.py::_build_merged_encoder.
SUPPORTED_MERGE_FN_NAMES = set(MERGE_FNS) | {"finetune"}


def merge_task_vectors(merge_fn_name: str, task_vectors: list):
    if merge_fn_name == "finetune":
        return task_vectors[-1]
    if merge_fn_name not in MERGE_FNS:
        raise NotImplementedError(
            f"merge_fn={merge_fn_name!r} not supported for NLP backends "
            f"(supported: {sorted(SUPPORTED_MERGE_FN_NAMES)})."
        )
    return MERGE_FNS[merge_fn_name](task_vectors)
