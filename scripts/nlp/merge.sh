#!/bin/env/bash

model=bert-base-uncased
dataset=LSB
epochs=3
task_seq=A            # A B C
seed=3                # 3 4 5
gpu_id=0
num_target_data=200
merge_fn=masked_magmax_with_targetdata  # finetune random_mix average ties magmax masked_magmax_with_targetdata
target_config=target_data_config_lsb

echo "======================================================================================"
echo "Merging for target data on ${dataset} (merge_fn: ${merge_fn}, num_target_data: ${num_target_data})"
echo "======================================================================================"

out_dir=outs/${model}/nlp_classification/${dataset}/taskseq_${task_seq}
mkdir -p ${out_dir}

python merge_for_targetdata.py \
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
        |& tee ${out_dir}/merge-${merge_fn}-epochs:${epochs}-seed:${seed}.out
