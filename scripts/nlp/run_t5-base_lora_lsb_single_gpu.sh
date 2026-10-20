#!/usr/bin/env bash
# Run the t5-base x LSB experiment with LoRA adapters (--finetune_mode lora,
# the t5-large setting: r=8, alpha=32, q and v, lr 1e-3) on a machine with a
# single GPU: fine-tune every order x seed (scripts/nlp/finetune_all.sh), then
# run the method comparison on them (scripts/nlp/merge_comparison_all.sh).
#
# Unlike scripts/nlp/run_t5-base_lsb.sh, which runs the orders in parallel with
# one GPU each, the orders here run one after another on gpu_id, and the one
# running process gets every core (threads, below). Three processes on one
# 16GB GPU would leave little room for evaluation (~5GB each while training),
# and parallel processes would compete for the cores the masked merge needs.
# Each order writes its own log under logs/nlp_sweep/.
#
# A failing order does not stop the others — the failures are listed at the
# end and the exit status is non-zero. Finished tasks / target environments are
# skipped inside the Python scripts, so re-running resumes the sweep.
#
# Everything the called scripts read (seeds, merge_fns, ...) can be overridden
# from the environment, e.g.
#   orders="4" seeds="3" bash scripts/nlp/run_t5-base_lora_lsb_single_gpu.sh
# To keep it running after logout:
#   nohup bash scripts/nlp/run_t5-base_lora_lsb_single_gpu.sh > /dev/null 2>&1 &

export model=t5-base
export dataset=LSB
export finetune_mode=lora
export gpu_id=${gpu_id:-0}
# Only one process runs at a time, so give it every core this shell may use.
# This deliberately ignores an inherited OMP_NUM_THREADS (a login profile may
# set one, and nproc itself honours it), so set threads to use fewer cores.
threads=${threads:-$(env -u OMP_NUM_THREADS nproc)}
export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads}
orders=${orders:-"4 5 6"}

script_dir=$(dirname "$0")
log_dir=logs/nlp_sweep
mkdir -p ${log_dir}

failed=()
for task_seq in ${orders}; do
    log=${log_dir}/${model}_${dataset}-${finetune_mode}_order${task_seq}.log
    echo "order ${task_seq} -> GPU ${gpu_id} (log: ${log})"
    (
        export orders=${task_seq}
        # Merge even if some seeds failed to train: the seeds that did finish
        # still get merged, and the failed ones are listed by both scripts.
        bash "${script_dir}/finetune_all.sh"; ft_status=$?
        bash "${script_dir}/merge_comparison_all.sh"; merge_status=$?
        [ ${ft_status} -eq 0 ] && [ ${merge_status} -eq 0 ]
    ) >> ${log} 2>&1 || failed+=("order=${task_seq}")
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All orders finished."
else
    echo "Failed orders (see ${log_dir}/${model}_${dataset}-${finetune_mode}_order*.log):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
