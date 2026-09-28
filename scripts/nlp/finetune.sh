#!/bin/env/bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# fine-tuning run looks like a success and the merge step goes ahead
# against checkpoints that were never written.
set -o pipefail


model=bert-base-uncased
dataset=LSB
epochs=3
task_seq=A            # A B C
seed=3                # 3 4 5

echo "======================================================================================"
echo "Finetuning ${model} on ${dataset} (pattern: ${task_seq}) seed=${seed}"
echo "======================================================================================"

out_dir=outs/${model}/nlp_classification/${dataset}/taskseq_${task_seq}
mkdir -p ${out_dir}

python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --sequential-finetuning \
    --seed ${seed} \
    --taskseq_pattern ${task_seq} \
        |& tee ${out_dir}/epochs:${epochs}-seed:${seed}.out
