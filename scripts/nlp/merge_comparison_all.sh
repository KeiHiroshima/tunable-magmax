#!/usr/bin/env bash
# Run the method comparison (scripts/nlp/merge_comparison.sh: every merge_fn)
# for every seed. LSB has a single task sequence (no n_splits), so seed is the
# only axis swept here — unlike scripts/vision/merge_comparison_all.sh.
#
# A failing seed does not stop the sweep — the failures are listed at the end
# and the exit status is non-zero. Finished target environments are skipped
# inside merge_for_targetdata.py, so re-running resumes the sweep. A seed whose
# checkpoints do not exist yet (scripts/nlp/finetune_all.sh has not reached it)
# fails and is listed.
#
# Everything merge_comparison.sh and merge.sh read (merge_fns, epochs,
# target_config, ...) can be overridden from the environment, e.g.
#   epochs=10 seeds="4 5" merge_fns="magmax masked_magmax_with_targetdata" \
#       bash scripts/nlp/merge_comparison_all.sh

seeds=${seeds:-"3 4 5"}

script_dir=$(dirname "$0")
failed=()

for seed in ${seeds}; do
    if ! seed=${seed} bash "${script_dir}/merge_comparison.sh"; then
        failed+=("seed=${seed}")
    fi
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All seeds finished."
else
    echo "Failed seeds (the per-method failures are listed above each summary line):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
