#!/usr/bin/env bash
# Run the vision LoRA experiment end to end: fine-tune every n_splits x seed
# with --finetune_mode lora (scripts/vision/finetune_all.sh), then run the
# method comparison on them (scripts/vision/merge_comparison_all.sh). Every
# other setting is the full-fine-tuning run's (logs/.../reproducing).
#
# Fine-tuning uses every visible GPU (DataParallel), so the runs go one after
# another. Merging uses one GPU per process, so each n_splits merges in
# parallel on its own GPU: the i-th entry of n_splits_list goes to the i-th
# entry of gpu_ids. Logs go to logs/vision_lora_sweep/.
#
# A failing step does not stop the others — failures are listed at the end and
# the exit status is non-zero. Finished splits / target environments are
# skipped inside the Python scripts, so re-running resumes the sweep.
#
# Everything the called scripts read can be overridden from the environment, e.g.
#   n_splits_list="5" gpu_ids="0" seeds="3" bash scripts/vision/run_lora_sweep.sh
# Only one sweep may run at a time: two would train the same splits at once.

export finetune_mode=lora
# The merges run in parallel and are CPU-heavy (the masked merge, OT). Left
# alone, each process starts one thread per core, and the oversubscribed
# machine (3 processes x 24 threads on 24 cores) ran the proposed method's
# merges up to ~8x slower than a single full-run merge did. Cap the threads
# per process instead; the results do not depend on the thread count.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
n_splits_list=${n_splits_list:-"5 20 50"}
gpu_ids=${gpu_ids:-"0 1 2"}

read -ra n_splits_arr <<< "${n_splits_list}"
read -ra gpu_arr <<< "${gpu_ids}"
if [ ${#n_splits_arr[@]} -gt ${#gpu_arr[@]} ]; then
    echo "need one GPU per n_splits: n_splits_list=(${n_splits_list}) gpu_ids=(${gpu_ids})" >&2
    exit 1
fi

script_dir=$(dirname "$0")
log_dir=logs/vision_lora_sweep
mkdir -p ${log_dir}
failed=()

echo "fine-tuning n_splits=(${n_splits_list}) (log: ${log_dir}/finetune.log)"
if ! n_splits_list="${n_splits_list}" bash "${script_dir}/finetune_all.sh" >> ${log_dir}/finetune.log 2>&1; then
    failed+=("fine-tuning")
fi

pids=()
for i in "${!n_splits_arr[@]}"; do
    n_splits=${n_splits_arr[$i]}
    gpu_id=${gpu_arr[$i]}
    log=${log_dir}/merge_CIFAR100-${n_splits}.log
    echo "merging n_splits=${n_splits} -> GPU ${gpu_id} (log: ${log})"
    n_splits_list=${n_splits} gpu_id=${gpu_id} \
        bash "${script_dir}/merge_comparison_all.sh" >> ${log} 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    if ! wait ${pids[$i]}; then
        failed+=("merging n_splits=${n_splits_arr[$i]}")
    fi
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All steps finished."
else
    echo "Failed (see ${log_dir}/):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
