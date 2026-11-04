#!/usr/bin/env bash
# Run the method comparison (scripts/vision/merge_comparison.sh: every merge_fn,
# and every similarity_metric for the proposed method) for every
# (n_splits, seed) combination of one dataset.
#
# A failing combination does not stop the sweep — the failures are listed at
# the end and the exit status is non-zero. Result JSONs that already exist are
# skipped per target inside merge_for_targetdata.py, so re-running resumes the
# sweep where it stopped. A combination whose checkpoints do not exist yet
# (scripts/vision/finetune_all.sh has not reached it) fails and is listed.
#
# Everything merge_comparison.sh and merge.sh read (dataset, merge_fns,
# similarity_metrics, target_config, ...) can be overridden from the
# environment and is passed through, e.g.
#   dataset=ImageNetR bash scripts/vision/merge_comparison_all.sh
#   n_splits_list="20" seeds="4 5" merge_fns="magmax" bash scripts/vision/merge_comparison_all.sh

n_splits_list=${n_splits_list:-"5 20 50"}
seeds=${seeds:-"3 4 5"}

script_dir=$(dirname "$0")
failed=()

for n_splits in ${n_splits_list}; do
    for seed in ${seeds}; do
        if ! n_splits=${n_splits} seed=${seed} bash "${script_dir}/merge_comparison.sh"; then
            failed+=("n_splits=${n_splits} seed=${seed}")
        fi
    done
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All combinations finished."
else
    echo "Failed combinations (the per-method failures are listed above each summary line):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
