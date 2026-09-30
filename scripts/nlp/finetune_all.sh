#!/usr/bin/env bash
# Fine-tune every order x seed of one benchmark, by calling
# scripts/nlp/finetune.sh once per pair. The orders are the benchmark's own
# (O-LoRA's Orders 1-3 for StdCL, 4-6 for LSB).
#
# A failing run does not stop the sweep — the failures are listed at the end
# and the exit status is non-zero. Tasks whose checkpoints already exist are
# skipped inside finetune_splitted.py, so re-running resumes the sweep.
#
# Everything finetune.sh reads (model, epochs, gpu_id, ...) can be
# overridden from the environment and is passed through, e.g.
#   model=t5-large dataset=LSB seeds="4 5" gpu_id=1 bash scripts/nlp/finetune_all.sh

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
        if ! dataset=${dataset} task_seq=${task_seq} seed=${seed} bash "${script_dir}/finetune.sh"; then
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
