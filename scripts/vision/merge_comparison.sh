#!/usr/bin/env bash
# Run every merge_fn (and, for the proposed method, every similarity_metric)
# for one fine-tuned run, by calling scripts/vision/merge.sh once per method.
#
# Baselines ignore the similarity metric, so each runs once; the proposed
# method runs once per entry in similarity_metrics. A failing method does not
# stop the sweep — the failures are listed at the end and the exit status is
# non-zero. Methods whose result JSONs already exist are skipped per target
# inside merge_for_targetdata.py, so re-running resumes the sweep.
#
# Everything merge.sh reads (dataset, n_splits, seed, target_config, ...) can
# be overridden from the environment and is passed through, e.g.
#   dataset=ImageNetR n_splits=20 seed=4 bash scripts/vision/merge_comparison.sh
#   target_config=target_data_config_split20 n_splits=20 \
#       merge_fns="magmax masked_magmax_with_targetdata" \
#       similarity_metrics="labels ot_embedded" bash scripts/vision/merge_comparison.sh

merge_fns=${merge_fns:-"finetune random_mix average ties magmax masked_magmax_with_targetdata"}
similarity_metrics=${similarity_metrics:-"labels ot_embedded"} # cosine_embedded mmd_embedded

script_dir=$(dirname "$0")
failed=()

run() {
    if ! merge_fn=$1 similarity_metric=$2 bash "${script_dir}/merge.sh"; then
        failed+=("$1${2:+ ($2)}")
    fi
}

for merge_fn in ${merge_fns}; do
    if [ "${merge_fn}" = "masked_magmax_with_targetdata" ]; then
        for similarity_metric in ${similarity_metrics}; do
            run ${merge_fn} ${similarity_metric}
        done
    else
        run ${merge_fn} ""
    fi
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All methods finished."
else
    echo "Failed (see the corresponding .out files under outs/):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
