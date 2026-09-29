#!/usr/bin/env bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# merge looks like a success.
set -o pipefail
# print() is block-buffered when stdout is a pipe while logging flushes every
# record, so without this the two interleave out of order in the .out file.
export PYTHONUNBUFFERED=1

# Every variable can be overridden from the environment, e.g.
#   merge_fn=magmax seed=4 bash scripts/nlp/merge.sh
# scripts/nlp/merge_comparison.sh relies on this to sweep merge_fn.
model=${model:-bert-base-uncased}
dataset=${dataset:-LSB}
epochs=${epochs:-3}
task_seq=${task_seq:-A}             # A B C
seed=${seed:-3}                     # 3 4 5
gpu_id=${gpu_id:-0}
num_target_data=${num_target_data:-200}
merge_fn=${merge_fn:-masked_magmax_with_targetdata}  # finetune random_mix average ties magmax masked_magmax_with_targetdata
target_config=${target_config:-target_data_config_lsb}

echo "======================================================================================"
echo "Merging for target data on ${dataset} (merge_fn: ${merge_fn}, target_config: ${target_config}, num_target_data: ${num_target_data}, seed: ${seed})"
echo "======================================================================================"

out_dir=outs/${model}/nlp_classification/${dataset}/taskseq_${task_seq}
mkdir -p ${out_dir}

# tee -a: a re-run resumes (finished target environments are skipped), so
# appending keeps the original log instead of replacing it with skip messages.
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
