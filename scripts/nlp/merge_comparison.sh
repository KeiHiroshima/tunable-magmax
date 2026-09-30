#!/usr/bin/env bash
# Run every merge_fn for one fine-tuned StdCL/LSB run, by calling scripts/nlp/merge.sh
# once per method.
#
# Unlike scripts/vision/merge_comparison.sh there is no similarity_metric loop: the NLP
# backend's masked_magmax_with_targetdata takes the target environment's own
# mixing ratio as its preference vector (src/nlp/target_data.py) and ignores
# --similarity_metric.
#
# A failing method does not stop the sweep — the failures are listed at the end
# and the exit status is non-zero. Target environments whose result JSONs are
# already finished are skipped inside merge_for_targetdata.py, so re-running
# resumes the sweep.
#
# Everything merge.sh reads (model, dataset, task_seq, seed, target_config, ...)
# can be overridden from the environment and is passed through, e.g.
#   model=t5-large dataset=LSB task_seq=4 seed=4 bash scripts/nlp/merge_comparison.sh
#   merge_fns="magmax masked_magmax_with_targetdata" bash scripts/nlp/merge_comparison.sh

merge_fns=${merge_fns:-"finetune random_mix average ties magmax masked_magmax_with_targetdata"}

script_dir=$(dirname "$0")
failed=()

for merge_fn in ${merge_fns}; do
    if ! merge_fn=${merge_fn} bash "${script_dir}/merge.sh"; then
        failed+=("${merge_fn}")
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
