#!/usr/bin/env bash

set -e
# Without pipefail the `|& tee` below reports tee's status, so a failed
# fine-tuning run looks like a success and the merge step goes ahead
# against checkpoints that were never written.
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
#   model=t5-large dataset=LSB task_seq=4 seed=4 bash scripts/nlp/finetune.sh
# scripts/nlp/finetune_all.sh relies on this to sweep order x seed.
# Training runs on the single GPU picked by --gpu_id (no DataParallel).
model=${model:-t5-base}             # t5-base t5-large
dataset=${dataset:-StdCL}           # StdCL (orders 1-3) LSB (orders 4-6)
task_seq=${task_seq:-1}             # StdCL: 1 2 3 / LSB: 4 5 6
seed=${seed:-3}                     # 3 4 5
gpu_id=${gpu_id:-0}
wandb_entity_name=${wandb_entity_name:-keihiroshima}
# O-LoRA's T5 setting: 1 epoch, batch 64 (8 x 8 accumulated on one GPU).
epochs=${epochs:-1}
batch_size=${batch_size:-8}
grad_accum_steps=${grad_accum_steps:-8}
# t5-large trains LoRA adapters (merged into the weights task by task), every
# other model is fully fine-tuned. The learning rate follows the mode (1e-3
# for LoRA, 1e-4 for full) unless lr is set.
if [ "${model}" = "t5-large" ]; then
    finetune_mode=${finetune_mode:-lora}
else
    finetune_mode=${finetune_mode:-full}
fi
lr_flag=${lr:+--lr ${lr}}

echo "======================================================================================"
echo "Finetuning ${model} (${finetune_mode}) on ${dataset} (order: ${task_seq}) seed=${seed}"
echo "======================================================================================"

out_dir=outs/${model}/nlp_classification/${dataset}-${finetune_mode}/taskseq_${task_seq}
mkdir -p ${out_dir}

# tee -a: a re-run resumes (finished tasks are skipped), so appending keeps the
# original training log instead of replacing it with the skip messages.
uv run python finetune_splitted.py \
    --model ${model} \
    --dataset ${dataset} \
    --finetune_mode ${finetune_mode} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --grad_accum_steps ${grad_accum_steps} \
    ${lr_flag} \
    --sequential-finetuning \
    --seed ${seed} \
    --taskseq_pattern ${task_seq} \
    --gpu_id ${gpu_id} \
    --wandb_entity_name ${wandb_entity_name} \
        |& tee -a ${out_dir}/epochs:${epochs}-seed:${seed}.out
