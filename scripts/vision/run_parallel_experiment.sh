#!/usr/bin/env bash
# Fine-tune and merge CIFAR-100 for every (variant, n_splits, seed), three jobs
# at a time, one per GPU. The experiments run with it, and their fixed
# settings, are in scripts/vision/experiments/; launch one with
# scripts/vision/launch_experiment.sh.
#
#   variant    what is trained                             results (dir_name)
#   full       every image-tower parameter                 reproducing
#   lora-reuse one LoRA adapter across every split         reproducing-lora-reuse
#   lora       a LoRA adapter per split, merged in (O-LoRA) reproducing-lora
#
# Every job fine-tunes (scripts/vision/finetune.sh) and then merges
# (scripts/vision/merge_comparison.sh) with merge_fns. Finished splits are
# skipped, so on checkpoints that already exist a job only merges.
#
# Parallelism: worker i owns GPU i and CPU cores [i*cores_per_worker,
# (i+1)*cores_per_worker), pinned with taskset, so its DataLoader workers and
# the CPU-heavy merges never compete with another worker's. Fine-tuning
# therefore runs on one GPU per job (no DataParallel). Workers take jobs from
# a shared queue, longest first, so the GPUs stay busy until the queue is
# empty.
#
# Memory: a job whose line starts with one of heavy_jobs (e.g. "full 50": TIES
# on full fine-tuning at T=50 stacks fifty full-size task vectors, ~140 GB)
# goes to a queue of its own that only the first heavy_workers workers take,
# before they join the others on the rest. At most heavy_workers such jobs
# therefore run at once.
#
# A failing job does not stop the others; failures are listed at the end and
# the exit status is non-zero. Finished splits and result JSONs are skipped
# inside the Python scripts, so re-running resumes where it stopped.
#
# MAGMAX_BASE_DIR must be set: it is where checkpoints go (src/config.py), and
# it must hold checkpoints/ViT-B-16/zeroshot.pt (and the open_clip cache).
#
# Everything can be overridden from the environment, e.g.
#   variants="lora-reuse" n_splits_list="5" seeds="3" bash scripts/vision/run_parallel_experiment.sh
#   dry_run=1 bash scripts/vision/run_parallel_experiment.sh   # print the plan only

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
heavy_jobs=${heavy_jobs:-}          # e.g. "full 50": job-line prefixes, separated by commas
heavy_workers=${heavy_workers:-1}
# Names this run's queue and status files, so the phases of one experiment
# (scripts/vision/launch_experiment.sh) can share a log_dir.
phase=${phase:-}

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
queue=${log_dir}/queue${phase:+.${phase}}
heavy_queue=${queue}.heavy
status=${log_dir}/status${phase:+.${phase}}

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

# Move the heavy jobs to their own queue, keeping their order.
: > "${heavy_queue}"
if [ -n "${heavy_jobs}" ]; then
    if [ "${heavy_workers}" -lt 1 ] || [ "${heavy_workers}" -gt "${n_workers}" ]; then
        echo "heavy_workers must be between 1 and ${n_workers}" >&2
        exit 1
    fi
    IFS=, read -ra heavy_prefixes <<< "${heavy_jobs}"
    : > "${queue}.light"
    while read -r line; do
        dest=${queue}.light
        for prefix in "${heavy_prefixes[@]}"; do
            [[ "${line} " == "${prefix} "* ]] && dest=${heavy_queue}
        done
        echo "${line}" >> "${dest}"
    done < "${queue}"
    mv "${queue}.light" "${queue}"
fi

echo "======================================================================================"
echo "${phase:+phase ${phase}: }MAGMAX_BASE_DIR=${MAGMAX_BASE_DIR}"
echo "$(cat "${queue}" "${heavy_queue}" | wc -l) jobs on GPUs (${gpu_ids}), ${cores_per_worker} cores each; logs: ${log_dir}/"
echo "merge_fns=(${merge_fns}) similarity_metrics=(${similarity_metrics}) epochs=${epochs}"
echo "results: logs/.../${results_prefix}[-lora|-lora-reuse]/, fine-tuning logs: .../${ft_log_prefix}[-lora|-lora-reuse]/"
sed 's/^/  /' "${queue}"
if [ -s "${heavy_queue}" ]; then
    echo "heavy (${heavy_jobs}), run by the first ${heavy_workers} worker(s) only:"
    sed 's/^/  /' "${heavy_queue}"
fi
echo "======================================================================================"
if [ "${dry_run}" = "1" ]; then
    exit 0
fi
: > "${status}"

# Pop the first line of a queue file, under a lock shared by every worker.
pop_job() {
    local file=$1
    (
        flock 9
        line=$(head -n 1 "${file}")
        [ -n "${line}" ] && sed -i '1d' "${file}"
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
        local job=""
        [ "${i}" -lt "${heavy_workers}" ] && job=$(pop_job "${heavy_queue}")
        [ -z "${job}" ] && job=$(pop_job "${queue}")
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
    echo "Failed jobs${phase:+ in phase ${phase}} (see ${log_dir}/<variant>_${dataset}-<n_splits>_seed<seed>.log):"
    grep '^FAILED' "${status}" | sed 's/^FAILED /  /'
    exit 1
fi
echo "All $(wc -l < "${status}") jobs finished."
