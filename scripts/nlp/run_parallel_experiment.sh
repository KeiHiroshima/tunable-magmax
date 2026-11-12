#!/usr/bin/env bash
# Fine-tune (optionally) and merge StdCL/LSB for every (model, variant,
# dataset, order, seed), one job per GPU at a time. The experiments run with
# it, and their fixed settings, are in scripts/nlp/experiments/; launch one
# with scripts/nlp/launch_experiment.sh.
#
#   variant    what is trained                                checkpoints / results
#   full       every parameter (shared embeddings frozen)     {dataset}
#   lora       a LoRA adapter per task, merged in (O-LoRA)    {dataset}-lora
#   lora-reuse one LoRA adapter trained across every task     {dataset}-lora-reuse
#
# A job is "<model> <variant> <dataset> <order> <seed>". With finetune=1 it
# first runs scripts/nlp/finetune.sh (tasks whose checkpoints exist are
# skipped); then scripts/nlp/merge_comparison.sh with merge_fns (target
# environments whose result JSON exists are skipped). finetune=0 only merges,
# for checkpoints trained earlier, and so never starts a wandb run.
#
# Resources: worker i owns GPU gpu_ids[i] and the CPU cores
# [first_core + i*cores_per_worker, first_core + (i+1)*cores_per_worker),
# pinned with taskset, so the cores below first_core are left to others.
# Before each job the worker checks that no process is on its GPU, and waits
# (checking every gpu_wait_seconds) while one is, so it never shares a GPU
# with someone else's run. Workers take jobs from a shared queue, longest
# first (LSB before StdCL, t5-base before t5-large).
#
# A failing job does not stop the others; failures are listed at the end and
# the exit status is non-zero. Re-running resumes where it stopped.
#
# Everything can be overridden from the environment, e.g.
#   variants="t5-large:lora-reuse" datasets=StdCL orders_StdCL=1 seeds=3 \
#       gpu_ids=1 bash scripts/nlp/run_parallel_experiment.sh
#   dry_run=1 bash scripts/nlp/run_parallel_experiment.sh   # print the plan only

set -u

# model:variant pairs.
variants=${variants:-"t5-base:full t5-large:lora"}
datasets=${datasets:-"LSB StdCL"}   # longest first
orders_LSB=${orders_LSB:-"4 5 6"}
orders_StdCL=${orders_StdCL:-"1 2 3"}
seeds=${seeds:-"3 4 5"}
gpu_ids=${gpu_ids:-"1 2"}
first_core=${first_core:-8}
cores_per_worker=${cores_per_worker:-8}
gpu_wait_seconds=${gpu_wait_seconds:-600}
finetune=${finetune:-1}
export merge_fns=${merge_fns:-"finetune ties magmax masked_magmax_with_targetdata weighted_ties_with_targetdata"}
export target_config=${target_config:-target_data_config_lsb}
export epochs=${epochs:-1}
log_dir=${log_dir:-logs/nlp_parallel_experiment}
dry_run=${dry_run:-0}
# Names this run's queue and status files, so the phases of one experiment
# (scripts/nlp/launch_experiment.sh) can share a log_dir.
phase=${phase:-}

if [ -z "${MAGMAX_BASE_DIR:-}" ]; then
    echo "set MAGMAX_BASE_DIR (where checkpoints are; see src/config.py)" >&2
    exit 1
fi
read -ra gpu_arr <<< "${gpu_ids}"
n_workers=${#gpu_arr[@]}
last_core=$((first_core + n_workers * cores_per_worker - 1))
if [ "${last_core}" -ge "$(nproc)" ]; then
    echo "cores ${first_core}-${last_core} exceed the $(nproc) cores available" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${log_dir}"
queue=${log_dir}/queue${phase:+.${phase}}
status=${log_dir}/status${phase:+.${phase}}

: > "${queue}"
for dataset in ${datasets}; do
    case ${dataset} in
        LSB)   orders=${orders_LSB} ;;
        StdCL) orders=${orders_StdCL} ;;
        *)     echo "unknown dataset ${dataset}" >&2; exit 1 ;;
    esac
    for pair in ${variants}; do
        model=${pair%%:*} variant=${pair#*:}
        case ${variant} in full|lora|lora-reuse) ;; *) echo "unknown variant ${variant}" >&2; exit 1 ;; esac
        for order in ${orders}; do
            for seed in ${seeds}; do
                echo "${model} ${variant} ${dataset} ${order} ${seed}" >> "${queue}"
            done
        done
    done
done

echo "======================================================================================"
echo "${phase:+phase ${phase}: }MAGMAX_BASE_DIR=${MAGMAX_BASE_DIR}"
echo "$(wc -l < "${queue}") jobs on GPUs (${gpu_ids}), cores ${first_core}-${last_core} (${cores_per_worker} per GPU); logs: ${log_dir}/"
echo "finetune=${finetune} merge_fns=(${merge_fns}) target_config=${target_config} epochs=${epochs}"
sed 's/^/  /' "${queue}"
echo "======================================================================================"
if [ "${dry_run}" = "1" ]; then
    exit 0
fi
: > "${status}"

# Pop the first line of the queue, under a lock shared by every worker.
pop_job() {
    (
        flock 9
        line=$(head -n 1 "${queue}")
        [ -n "${line}" ] && sed -i '1d' "${queue}"
        echo "${line}"
    ) 9> "${queue}.lock"
}

# Number of processes on a GPU, whoever runs them (short options only:
# test/characterization/test_cli_surface.py checks every long option a script uses).
gpu_process_count() {
    nvidia-smi -i "$1" -q -d PIDS | grep -c "Process ID"
}

run_job() {
    local model=$1 variant=$2 dataset=$3 order=$4 seed=$5
    local finetune_mode lora_chain
    case ${variant} in
        full)       finetune_mode=full; lora_chain=merge ;;
        lora)       finetune_mode=lora; lora_chain=merge ;;
        lora-reuse) finetune_mode=lora; lora_chain=reuse ;;
    esac
    if [ "${finetune}" = "1" ]; then
        model=${model} dataset=${dataset} task_seq=${order} seed=${seed} gpu_id=0 \
            finetune_mode=${finetune_mode} lora_chain=${lora_chain} \
            bash "${script_dir}/finetune.sh" || return 1
    fi
    model=${model} dataset=${dataset} task_seq=${order} seed=${seed} gpu_id=0 \
        finetune_mode=${finetune_mode} lora_chain=${lora_chain} merge_space=full \
        bash "${script_dir}/merge_comparison.sh" || return 1
}

worker() {
    local i=$1
    local gpu=${gpu_arr[$i]}
    local first=$((first_core + i * cores_per_worker))
    local cores="${first}-$((first + cores_per_worker - 1))"
    while true; do
        local job
        job=$(pop_job)
        [ -z "${job}" ] && break
        while [ "$(gpu_process_count "${gpu}")" -gt 0 ]; do
            echo "$(date '+%F %T') [GPU ${gpu}] busy with another process; waiting ${gpu_wait_seconds}s before ${job}"
            sleep "${gpu_wait_seconds}"
        done
        read -r model variant dataset order seed <<< "${job}"
        local log=${log_dir}/${model}_${variant}_${dataset}_order${order}_seed${seed}.log
        echo "$(date '+%F %T') [GPU ${gpu}, cores ${cores}] start ${job} (log: ${log})"
        # One GPU, visible as cuda:0; every thread pool sized to this worker's
        # cores. /usr/bin/time -v reports the job's peak host memory at the
        # end of its log.
        if CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=${cores_per_worker} \
            MKL_NUM_THREADS=${cores_per_worker} \
            taskset -c "${cores}" /usr/bin/time -v bash -c "$(declare -f run_job); script_dir=${script_dir}; finetune=${finetune}; run_job ${job}" \
            >> "${log}" 2>&1; then
            result=ok
        else
            result=FAILED
        fi
        echo "$(date '+%F %T') [GPU ${gpu}] ${result} ${job}"
        echo "${result} ${job}" >> "${status}"
    done
}

pids=()
for i in "${!gpu_arr[@]}"; do
    worker "${i}" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

echo "======================================================================================"
if grep -q '^FAILED' "${status}"; then
    echo "Failed jobs${phase:+ in phase ${phase}} (see ${log_dir}/<model>_<variant>_<dataset>_order<o>_seed<s>.log):"
    grep '^FAILED' "${status}" | sed 's/^FAILED /  /'
    exit 1
fi
echo "All $(wc -l < "${status}") jobs finished."
