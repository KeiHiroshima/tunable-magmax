#!/bin/env/bash

set -e


model=ViT-B-16
dataset=CIFAR100  # CIFAR100 ImageNetR
epochs=10
n_splits=5        # 5 20 50
task_seq=A        # A B C
seed=3            # 3 4 5
gpu_id=0
num_train_data_each_task=500
merge_fn=masked_magmax_with_targetdata  # masked_magmax_with_targetdata finetune magmax ties average random_mix select_one_task_vector
ft_dir_name=finetune_target_data

merge_dir_name=DEFAULT_NAME

# --- Finetune ---
echo "======================================================================================"
echo "Finetuning ${model} on ${dataset}-${n_splits} (pattern: ${task_seq}) seed=${seed}"
echo "======================================================================================"

out_dir=outs/${model}/sequential_finetuning/class_incremental/${ft_dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
log_dir=logs/${model}/sequential_finetuning/class_incremental/${ft_dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
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

# --- Merge ---
echo "======================================================================================"
echo "Merging for target data on ${dataset}-${n_splits} (pattern: ${task_seq}, merge_fn: ${merge_fn})"
echo "======================================================================================"

if [ $n_splits -eq 5 ]; then
    num_target_data=1000
elif [ $n_splits -eq 20 ]; then
    num_target_data=500
else
    num_target_data=200
fi
echo "num_target_data is set to ${num_target_data}"

out_dir=outs/${model}/sequential_finetuning/class_incremental/${merge_dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
log_dir=logs/${model}/sequential_finetuning/class_incremental/${merge_dir_name}/${dataset}-${n_splits}/taskseq_${task_seq}
mkdir -p ${out_dir}
mkdir -p ${log_dir}

if [ "${merge_fn}" = "masked_magmax_with_targetdata" ]; then
    similarity_metric=labels #labels ot_embedded cosine_embedded mmd_embedded
    python merge_for_targetdata.py --model ${model} --dataset ${dataset} --epochs ${epochs} --n_splits ${n_splits} --split_strategy class --sequential-finetuning --results_db ${log_dir} --taskseq_pattern ${task_seq} --merge_fn ${merge_fn} --similarity_metric ${similarity_metric} --num_train_data_each_task ${num_train_data_each_task} --num_target_data ${num_target_data} --seed ${seed} --gpu_id ${gpu_id}
else
    python merge_for_targetdata.py --model ${model} --dataset ${dataset} --epochs ${epochs} --n_splits ${n_splits} --split_strategy class --sequential-finetuning --results_db ${log_dir} --taskseq_pattern ${task_seq} --merge_fn ${merge_fn} --num_train_data_each_task ${num_train_data_each_task} --num_target_data ${num_target_data} --seed ${seed} --gpu_id ${gpu_id}
fi
