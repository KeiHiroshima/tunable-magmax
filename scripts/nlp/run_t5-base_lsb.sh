#!/usr/bin/env bash
# Run the missing t5-base x LSB experiments: fine-tune every order x seed
# (scripts/nlp/finetune_all.sh), then run the method comparison on them
# (scripts/nlp/merge_comparison_all.sh). This is the same sweep that produced
# logs/nlp_sweep/{t5-base_StdCL,t5-large_LSB,t5-large_StdCL}.log.
#
# The orders (4 5 6) run in parallel, one GPU each: order i goes to the i-th
# entry of gpu_ids. Each order writes its own log under logs/nlp_sweep/.
# A failing order does not stop the others — the failures are listed at the
# end and the exit status is non-zero. Finished tasks / target environments are
# skipped inside the Python scripts, so re-running resumes the sweep.
#
# Everything the called scripts read (seeds, merge_fns, ...) can be overridden
# from the environment, e.g.
#   orders="4" gpu_ids="1" seeds="3" bash scripts/nlp/run_t5-base_lsb.sh
# To keep it running after logout:
#   nohup bash scripts/nlp/run_t5-base_lsb.sh > /dev/null 2>&1 &

export model=t5-base
export dataset=LSB
orders=${orders:-"4 5 6"}
gpu_ids=${gpu_ids:-"0 1 2"}

read -ra order_arr <<< "${orders}"
read -ra gpu_arr <<< "${gpu_ids}"
if [ ${#order_arr[@]} -gt ${#gpu_arr[@]} ]; then
    echo "need one GPU per order: orders=(${orders}) gpu_ids=(${gpu_ids})" >&2
    exit 1
fi

script_dir=$(dirname "$0")
log_dir=logs/nlp_sweep
mkdir -p ${log_dir}

pids=()
for i in "${!order_arr[@]}"; do
    task_seq=${order_arr[$i]}
    gpu_id=${gpu_arr[$i]}
    log=${log_dir}/${model}_${dataset}_order${task_seq}.log
    echo "order ${task_seq} -> GPU ${gpu_id} (log: ${log})"
    (
        export orders=${task_seq} gpu_id=${gpu_id}
        # Merge even if some seeds failed to train: the seeds that did finish
        # still get merged, and the failed ones are listed by both scripts.
        bash "${script_dir}/finetune_all.sh"; ft_status=$?
        bash "${script_dir}/merge_comparison_all.sh"; merge_status=$?
        [ ${ft_status} -eq 0 ] && [ ${merge_status} -eq 0 ]
    ) >> ${log} 2>&1 &
    pids+=($!)
done

failed=()
for i in "${!pids[@]}"; do
    if ! wait ${pids[$i]}; then
        failed+=("order=${order_arr[$i]}")
    fi
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All orders finished."
else
    echo "Failed orders (see ${log_dir}/${model}_${dataset}_order*.log):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
