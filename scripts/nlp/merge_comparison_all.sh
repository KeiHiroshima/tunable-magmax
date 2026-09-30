#!/usr/bin/env bash
# Run the method comparison (scripts/nlp/merge_comparison.sh: every merge_fn)
# for every order x seed of one benchmark (O-LoRA's Orders 1-3 for StdCL,
# 4-6 for LSB).
#
# A failing run does not stop the sweep — the failures are listed at the end
# and the exit status is non-zero. Finished target environments are skipped
# inside merge_for_targetdata.py, so re-running resumes the sweep. A run whose
# checkpoints do not exist yet (scripts/nlp/finetune_all.sh has not reached it)
# fails and is listed.
#
# Everything merge_comparison.sh and merge.sh read (model, merge_fns,
# target_config, ...) can be overridden from the environment, e.g.
#   model=t5-large dataset=LSB seeds="4 5" merge_fns="magmax masked_magmax_with_targetdata" \
#       bash scripts/nlp/merge_comparison_all.sh

dataset=${dataset:-StdCL}
case "${dataset}" in
    StdCL) default_orders="1 2 3" ;;
    LSB)   default_orders="4 5 6" ;;
    *)     echo "unknown dataset: ${dataset}" >&2; exit 1 ;;
esac
orders=${orders:-${default_orders}}
seeds=${seeds:-"3 4 5"}

script_dir=$(dirname "$0")
failed=()

for task_seq in ${orders}; do
    for seed in ${seeds}; do
        if ! dataset=${dataset} task_seq=${task_seq} seed=${seed} bash "${script_dir}/merge_comparison.sh"; then
            failed+=("order=${task_seq} seed=${seed}")
        fi
    done
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All runs finished."
else
    echo "Failed runs (see the corresponding .out files under outs/):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
