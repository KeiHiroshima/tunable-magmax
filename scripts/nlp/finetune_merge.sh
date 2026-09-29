#!/usr/bin/env bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# fine-tuning run looks like a success and the merge step goes ahead
# against checkpoints that were never written.
set -o pipefail
# print() is block-buffered when stdout is a pipe while logging flushes every
# record, so without this the two interleave out of order in the .out files.
export PYTHONUNBUFFERED=1


model=bert-base-uncased
dataset=LSB
epochs=3
task_seq=A            # A B C
seed=3                # 3 4 5
gpu_id=0
num_target_data=200
merge_fn=masked_magmax_with_targetdata  # finetune random_mix average ties magmax masked_magmax_with_targetdata
target_config=target_data_config_lsb
wandb_entity_name=keihiroshima


out_dir=outs/${model}/nlp_classification/${dataset}/taskseq_${task_seq}
mkdir -p ${out_dir}

# --- Finetune ---
echo "======================================================================================"
echo "Finetuning ${model} on ${dataset} (pattern: ${task_seq}) seed=${seed}"
echo "======================================================================================"

uv run python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --sequential-finetuning \
    --seed ${seed} \
    --taskseq_pattern ${task_seq} \
    --wandb_entity_name ${wandb_entity_name} \
        |& tee -a ${out_dir}/epochs:${epochs}-seed:${seed}.out

# --- Merge ---
echo "======================================================================================"
echo "Merging for target data on ${dataset} (pattern: ${task_seq}, merge_fn: ${merge_fn})"
echo "======================================================================================"

uv run python merge_for_targetdata.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --sequential-finetuning \
    --taskseq_pattern ${task_seq} \
    --seed ${seed} \
    --gpu_id ${gpu_id} \
    --merge_fn ${merge_fn} \
    --target_config ${target_config} \
    --num_target_data ${num_target_data} \
        |& tee -a ${out_dir}/merge-${merge_fn}-${target_config}-epochs:${epochs}-seed:${seed}.out
