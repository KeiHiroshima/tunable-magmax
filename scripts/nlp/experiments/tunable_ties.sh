# Tunable TIES (merge_fn weighted_ties_with_targetdata) on LSB (Orders 4-6),
# against finetune, TIES, MAGMAX and Tunable MAGMAX, for t5-base and t5-large.
# Sourced by scripts/nlp/launch_experiment.sh.
#
#   existing  the 18 LSB runs trained earlier (t5-base full, t5-large LoRA
#             merged per task; Orders 4-6 x seeds 3-5). The other methods are
#             already merged there, so only Tunable TIES is added; nothing is
#             fine-tuned again.
#   reuse     one LoRA adapter trained across every task (lora_chain reuse;
#             r=8, alpha=32, q and v, lr 1e-3, as the per-task LoRA), for
#             t5-base and t5-large: 18 LSB runs, fine-tuned from scratch, then
#             merged with every compared method.
#
# GPU 0 and cores 0-7 are left to other users: two workers, on GPUs 1 and 2,
# pinned to cores 8-15 and 16-23.

export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_OLORA_DATA_DIR=/home/KeiHiroshima/O-LoRA/CL_Benchmark
export datasets=LSB
export orders_LSB="4 5 6"
export seeds="3 4 5"
export epochs=1
export target_config=target_data_config_lsb
export gpu_ids="1 2"
export first_core=8
export cores_per_worker=8
phases=(
    "existing variants='t5-base:full t5-large:lora' finetune=0 merge_fns=weighted_ties_with_targetdata"
    "reuse variants='t5-base:lora-reuse t5-large:lora-reuse' finetune=1 merge_fns='finetune ties magmax masked_magmax_with_targetdata weighted_ties_with_targetdata'"
)
export log_dir=logs/nlp_tunable_ties_experiment
