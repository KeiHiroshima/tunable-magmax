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
#   seed=4 bash scripts/nlp/finetune.sh
# scripts/nlp/finetune_all.sh relies on this to sweep seed.
# Training runs on the single GPU picked by --gpu_id (no DataParallel).
model=${model:-bert-base-uncased}
dataset=${dataset:-LSB}
epochs=${epochs:-3}
task_seq=${task_seq:-A}             # A B C
seed=${seed:-3}                     # 3 4 5
gpu_id=${gpu_id:-0}
wandb_entity_name=${wandb_entity_name:-keihiroshima}

echo "======================================================================================"
echo "Finetuning ${model} on ${dataset} (pattern: ${task_seq}) seed=${seed}"
echo "======================================================================================"

out_dir=outs/${model}/nlp_classification/${dataset}/taskseq_${task_seq}
mkdir -p ${out_dir}

# tee -a: a re-run resumes (finished tasks are skipped), so appending keeps the
# original training log instead of replacing it with the skip messages.
uv run python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --sequential-finetuning \
    --seed ${seed} \
    --taskseq_pattern ${task_seq} \
    --gpu_id ${gpu_id} \
    --wandb_entity_name ${wandb_entity_name} \
        |& tee -a ${out_dir}/epochs:${epochs}-seed:${seed}.out
