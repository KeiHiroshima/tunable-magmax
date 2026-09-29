#!/usr/bin/env bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# merge looks like a success.
set -o pipefail
# print() is block-buffered when stdout is a pipe while logging flushes every
# record, so without this the two interleave out of order in the .out file.
export PYTHONUNBUFFERED=1

# Every variable can be overridden from the environment, e.g.
#   merge_fn=magmax n_splits=20 bash scripts/vision/merge.sh
# scripts/vision/merge_comparison.sh relies on this to sweep merge_fn/similarity_metric.
model=${model:-ViT-B-16}
dataset=${dataset:-CIFAR100}    # CIFAR100 ImageNetR
epochs=${epochs:-10}
n_splits=${n_splits:-5}         # 5 20 50
task_seq=${task_seq:-A}
seed=${seed:-3}                 # 3 4 5
gpu_id=${gpu_id:-0}
num_train_data_each_task=${num_train_data_each_task:-500}
merge_fn=${merge_fn:-masked_magmax_with_targetdata}  # finetune magmax ties average random_mix masked_magmax_with_targetdata
similarity_metric=${similarity_metric:-labels}       # labels ot_embedded cosine_embedded mmd_embedded (masked_magmax_with_targetdata only)
target_config=${target_config:-target_data_config}   # target_data_config target_data_config_split{5,20,50}
dir_name=${dir_name:-DEFAULT_NAME}


if [ $n_splits -eq 5 ]; then
    num_target_data=1000
elif [ $n_splits -eq 20 ]; then
    num_target_data=500
else
    num_target_data=200
fi

similarity_args=()
method_tag=${merge_fn}
if [ "${merge_fn}" = "masked_magmax_with_targetdata" ]; then
    similarity_args=(--similarity_metric ${similarity_metric})
    method_tag=${merge_fn}-${similarity_metric}
fi

echo "======================================================================================"
echo "Merging for target data on ${dataset}-${n_splits} (method: ${method_tag}, target_config: ${target_config}, num_target_data: ${num_target_data}, seed: ${seed})"
echo "======================================================================================"

out_dir=outs/${model}/sequential_finetuning/class_incremental/${dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
log_dir=logs/${model}/sequential_finetuning/class_incremental/${dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
mkdir -p ${out_dir}
mkdir -p ${log_dir}

uv run python merge_for_targetdata.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --sequential-finetuning \
    --results_db ${log_dir} \
    --taskseq_pattern ${task_seq} \
    --merge_fn ${merge_fn} \
    "${similarity_args[@]}" \
    --target_config ${target_config} \
    --num_train_data_each_task ${num_train_data_each_task} \
    --num_target_data ${num_target_data} \
    --seed ${seed} \
    --gpu_id ${gpu_id} \
    --wandb_entity_name "keihiroshima" \
        |& tee -a ${out_dir}/merge-${method_tag}-${target_config}-seed:${seed}.out
