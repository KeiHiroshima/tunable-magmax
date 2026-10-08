#!/usr/bin/env bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# fine-tuning run looks like a success and the merge step goes ahead
# against checkpoints that were never written.
set -o pipefail
# print() is block-buffered when stdout is a pipe while logging flushes every
# record, so without this the two interleave out of order in the .out file.
export PYTHONUNBUFFERED=1

# Every variable can be overridden from the environment, e.g.
#   n_splits=20 seed=4 bash scripts/vision/finetune.sh
# scripts/vision/finetune_all.sh relies on this to sweep n_splits/seed.
# Training uses every visible GPU (DataParallel); restrict with CUDA_VISIBLE_DEVICES.
model=${model:-ViT-B-16}
dataset=${dataset:-CIFAR100}    # CIFAR100 ImageNetR
epochs=${epochs:-10}
n_splits=${n_splits:-5}         # 5 20 50
task_seq=${task_seq:-A}         # A B C
seed=${seed:-3}                 # 3 4 5
dir_name=${dir_name:-DEFAULT_NAME}
wandb_entity_name=${wandb_entity_name:-keihiroshima}

echo "======================================================================================"
echo "Finetuning ${model} on ${dataset}-${n_splits} (pattern: ${task_seq}) seed=${seed}"
echo "======================================================================================"

out_dir=outs/${model}/sequential_finetuning/class_incremental/${dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
log_dir=logs/${model}/sequential_finetuning/class_incremental/${dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
mkdir -p ${out_dir}
mkdir -p ${log_dir}

# tee -a: a re-run resumes (finished splits are skipped), so appending keeps the
# original training log instead of replacing it with the skip messages.
uv run python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --sequential-finetuning \
    --seed ${seed} \
    --results_db ${log_dir} \
    --taskseq_pattern ${task_seq} \
    --wandb_entity_name ${wandb_entity_name} \
        |& tee -a ${out_dir}/splits:${n_splits}-ep:${epochs}-seed:${seed}.out
