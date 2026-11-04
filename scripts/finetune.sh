#!/bin/env/bash

set -e


model=ViT-B-16
dataset=CIFAR100  # CIFAR100 ImageNetR
epochs=10
n_splits=5        # 5 20 50
task_seq=A        # A B C
seed=3            # 3 4 5
dir_name=DEFAULT_NAME

echo "======================================================================================"
echo "Finetuning ${model} on ${dataset}-${n_splits} (pattern: ${task_seq}) seed=${seed}"
echo "======================================================================================"

out_dir=outs/${model}/sequential_finetuning/class_incremental/${dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
log_dir=logs/${model}/sequential_finetuning/class_incremental/${dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
mkdir -p ${out_dir}
mkdir -p ${log_dir}

python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --sequential-finetuning \
    --seed ${seed} \
    --results_db ${log_dir} \
    --taskseq_pattern ${task_seq} \
        |& tee ${out_dir}/splits:${n_splits}-ep:${epochs}-seed:${seed}.out
