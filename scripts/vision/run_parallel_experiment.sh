#!/usr/bin/env bash
# The --lora_chain reuse experiment (docs/lora_reuse_plan.md §5): fine-tune and
# merge CIFAR-100 for every (variant, n_splits, seed), three jobs at a time, one
# per GPU.
#
#   variant    what is trained                             results (dir_name)
#   full       every image-tower parameter                 reproducing
#   lora-reuse one LoRA adapter across every split         reproducing-lora-reuse
#   lora       a LoRA adapter per split, merged in (O-LoRA) reproducing-lora
#
# The main comparison is full vs lora-reuse; lora vs lora-reuse is the ablation
# over how the LoRA chain is built. Every job fine-tunes (scripts/vision/
# finetune.sh) and then merges (scripts/vision/merge_comparison.sh) with
# merge_fns, by default the no-merge baseline, MAGMAX and Tunable MAGMAX.
#
# Parallelism: worker i owns GPU i and CPU cores [i*cores_per_worker,
# (i+1)*cores_per_worker), pinned with taskset, so its DataLoader workers and
# the CPU-heavy masked merge never compete with another worker's. Fine-tuning
# therefore runs on one GPU per job (no DataParallel). Workers take jobs from
# a shared queue, longest first, so the GPUs stay busy until the queue is
# empty.
#
# A failing job does not stop the others; failures are listed at the end and
# the exit status is non-zero. Finished splits and result JSONs are skipped
# inside the Python scripts, so re-running resumes where it stopped.
#
# MAGMAX_BASE_DIR must be set: it is where checkpoints go (src/config.py), and
# it must hold checkpoints/ViT-B-16/zeroshot.pt (and the open_clip cache).
#
# Everything can be overridden from the environment, e.g.
#   variants="lora-reuse" n_splits_list="5" seeds="3" bash scripts/vision/run_lora_reuse_experiment.sh
#   dry_run=1 bash scripts/vision/run_lora_reuse_experiment.sh   # print the plan only

set -u

variants=${variants:-"full lora-reuse lora"}
n_splits_list=${n_splits_list:-"50 20 5"}  # longest first: T=50 has the most per-split overhead
seeds=${seeds:-"3 4 5"}
gpu_ids=${gpu_ids:-"0 1 2"}
cores_per_worker=${cores_per_worker:-8}
export merge_fns=${merge_fns:-"finetune magmax masked_magmax_with_targetdata"}
export similarity_metrics=${similarity_metrics:-"labels ot_embedded"}
export dataset=${dataset:-CIFAR100}
export epochs=${epochs:-10}
log_dir=${log_dir:-logs/vision_lora_reuse_experiment}
# Prefixes of the results (merge.sh) and fine-tuning log (finetune.sh)
# directories; the variant's suffix is appended. Change them for a trial run:
# the results directory does not encode epochs, so a trial's result JSONs left
# under "reproducing-*" would make the real run skip those evaluations.
results_prefix=${results_prefix:-reproducing}
ft_log_prefix=${ft_log_prefix:-DEFAULT_NAME}
dry_run=${dry_run:-0}

if [ -z "${MAGMAX_BASE_DIR:-}" ]; then
    echo "set MAGMAX_BASE_DIR (where checkpoints go; see src/config.py)" >&2
    exit 1
fi
if [ ! -e "${MAGMAX_BASE_DIR}/checkpoints/ViT-B-16/zeroshot.pt" ]; then
    echo "missing ${MAGMAX_BASE_DIR}/checkpoints/ViT-B-16/zeroshot.pt" >&2
    exit 1
fi
read -ra gpu_arr <<< "${gpu_ids}"
n_workers=${#gpu_arr[@]}
if [ $((n_workers * cores_per_worker)) -gt "$(nproc)" ]; then
    echo "${n_workers} workers x ${cores_per_worker} cores exceeds $(nproc) cores" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${log_dir}"
queue=${log_dir}/queue
status=${log_dir}/status

# Longest jobs first (full fine-tuning, then large n_splits), so no GPU is
# left with one long job after the others have run dry.
: > "${queue}"
for variant in ${variants}; do
    case ${variant} in full|lora-reuse|lora) ;; *) echo "unknown variant ${variant}" >&2; exit 1 ;; esac
done
for variant in full lora-reuse lora; do
    [[ " ${variants} " == *" ${variant} "* ]] || continue
    for n_splits in ${n_splits_list}; do
        for seed in ${seeds}; do
            echo "${variant} ${n_splits} ${seed}" >> "${queue}"
        done
    done
done

echo "======================================================================================"
echo "MAGMAX_BASE_DIR=${MAGMAX_BASE_DIR}"
echo "$(wc -l < "${queue}") jobs on GPUs (${gpu_ids}), ${cores_per_worker} cores each; logs: ${log_dir}/"
echo "merge_fns=(${merge_fns}) similarity_metrics=(${similarity_metrics}) epochs=${epochs}"
echo "results: logs/.../${results_prefix}[-lora|-lora-reuse]/, fine-tuning logs: .../${ft_log_prefix}[-lora|-lora-reuse]/"
cat "${queue}" | sed 's/^/  /'
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

run_job() {
    local variant=$1 n_splits=$2 seed=$3
    local finetune_mode lora_chain suffix
    case ${variant} in
        full)       finetune_mode=full; lora_chain=merge; suffix="" ;;
        lora)       finetune_mode=lora; lora_chain=merge; suffix=-lora ;;
        lora-reuse) finetune_mode=lora; lora_chain=reuse; suffix=-lora-reuse ;;
    esac
    # The same names finetune.sh / merge.sh pick by default for each variant.
    finetune_mode=${finetune_mode} lora_chain=${lora_chain} n_splits=${n_splits} seed=${seed} \
        dir_name=${ft_log_prefix}${suffix} bash "${script_dir}/finetune.sh" || return 1
    finetune_mode=${finetune_mode} lora_chain=${lora_chain} n_splits=${n_splits} seed=${seed} \
        dir_name=${results_prefix}${suffix} gpu_id=0 bash "${script_dir}/merge_comparison.sh" || return 1
}

worker() {
    local i=$1
    local gpu=${gpu_arr[$i]}
    local first=$((i * cores_per_worker))
    local cores="${first}-$((first + cores_per_worker - 1))"
    while true; do
        local job
        job=$(pop_job)
        [ -z "${job}" ] && break
        read -r variant n_splits seed <<< "${job}"
        local log=${log_dir}/${variant}_${dataset}-${n_splits}_seed${seed}.log
        echo "$(date '+%F %T') [GPU ${gpu}, cores ${cores}] start ${job} (log: ${log})"
        # One GPU, visible as cuda:0; every thread pool sized to this worker's cores.
        if CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=${cores_per_worker} \
            MKL_NUM_THREADS=${cores_per_worker} \
            taskset -c "${cores}" bash -c "$(declare -f run_job); script_dir=${script_dir}; results_prefix=${results_prefix}; ft_log_prefix=${ft_log_prefix}; run_job ${job}" \
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
    echo "Failed jobs (see ${log_dir}/<variant>_${dataset}-<n_splits>_seed<seed>.log):"
    grep '^FAILED' "${status}" | sed 's/^FAILED /  /'
    exit 1
fi
echo "All $(wc -l < "${status}") jobs finished."
