"""Small shared lookup for merge_fn names -> functions, used by the NLP
backends only (src/backends/vision_backend.py keeps its own copy of this
dict exactly as it was in the original merge_for_targetdata.py, unchanged).

Deliberately excludes two entries from the original merge_fn_dict:
- "masked_magmax_with_targetdata": needs a target-data config (analogous to
  configs/target_data_config*.json) that doesn't exist for NLP yet.
- "select_one_task_vector": needs vision-specific args (a classification
  head + target_dataset_meta) not available outside the vision pipeline.
Both are candidates for a follow-up once NLP target-data support exists.
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
