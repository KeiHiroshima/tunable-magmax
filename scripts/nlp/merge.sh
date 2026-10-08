#!/usr/bin/env bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# merge looks like a success.
set -o pipefail
# print() is block-buffered when stdout is a pipe while logging flushes every
# record, so without this the two interleave out of order in the .out file.
export PYTHONUNBUFFERED=1
# torch 2.14 routes some ops T5's generate() hits (e.g. the bmm in cached
# decoding) to Triton kernels JIT-compiled on first use, which needs the
# Python headers (python3-dev); without them evaluation crashes. This keeps
# those ops on the stock CUDA kernels.
export TORCH_DISABLE_NATIVE_JIT=${TORCH_DISABLE_NATIVE_JIT:-1}

# Every variable can be overridden from the environment, e.g.
#   merge_fn=magmax model=t5-large dataset=LSB task_seq=4 bash scripts/nlp/merge.sh
# scripts/nlp/merge_comparison.sh relies on this to sweep merge_fn.
model=${model:-t5-base}             # t5-base t5-large
dataset=${dataset:-StdCL}           # StdCL (orders 1-3) LSB (orders 4-6)
task_seq=${task_seq:-1}             # StdCL: 1 2 3 / LSB: 4 5 6
seed=${seed:-3}                     # 3 4 5
epochs=${epochs:-1}
gpu_id=${gpu_id:-0}
num_target_data=${num_target_data:-200}
merge_fn=${merge_fn:-masked_magmax_with_targetdata}  # finetune random_mix average ties magmax masked_magmax_with_targetdata
target_config=${target_config:-target_data_config_lsb}
# Must match the fine-tuning run (scripts/nlp/finetune.sh picks it the same way).
if [ "${model}" = "t5-large" ]; then
    finetune_mode=${finetune_mode:-lora}
else
    finetune_mode=${finetune_mode:-full}
fi
# 1: also score single-model merges on every task's whole test split
# (O-LoRA's Average Accuracy). Ignored by masked_magmax_with_targetdata.
eval_full_testsets=${eval_full_testsets:-1}
full_flag=$([ "${eval_full_testsets}" = "1" ] && echo --eval_full_testsets || true)

echo "======================================================================================"
echo "Merging ${model} (${finetune_mode}) for target data on ${dataset} (order: ${task_seq}, merge_fn: ${merge_fn}, target_config: ${target_config}, num_target_data: ${num_target_data}, seed: ${seed})"
echo "======================================================================================"

out_dir=outs/${model}/nlp_classification/${dataset}-${finetune_mode}/taskseq_${task_seq}
mkdir -p ${out_dir}

# tee -a: a re-run resumes (finished target environments are skipped), so
# appending keeps the original log instead of replacing it with skip messages.
uv run python merge_for_targetdata.py \
    --model ${model} \
    --dataset ${dataset} \
    --finetune_mode ${finetune_mode} \
    --epochs ${epochs} \
    --sequential-finetuning \
    --taskseq_pattern ${task_seq} \
    --seed ${seed} \
    --gpu_id ${gpu_id} \
    --merge_fn ${merge_fn} \
    --target_config ${target_config} \
    --num_target_data ${num_target_data} \
    ${full_flag} \
        |& tee -a ${out_dir}/merge-${merge_fn}-${target_config}-epochs:${epochs}-seed:${seed}.out
