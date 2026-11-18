# The --lora_chain reuse experiment (docs/vision_lora_reuse_report.md):
# full fine-tuning vs one LoRA adapter reused across splits (main), and LoRA
# reuse vs a LoRA adapter per split merged in (ablation), each fine-tuned
# from scratch and merged without merging / with MAGMAX / with Tunable MAGMAX.
# Sourced by scripts/vision/launch_experiment.sh.

# Checkpoints go to a fresh base directory; only the zero-shot model and the
# open_clip cache are linked from the earlier one (see run_info.txt).
export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_DATA_DIR=/mnt/ssd/KeiHiroshima/data
export variants="full lora-reuse lora"
export n_splits_list="50 20 5"
export seeds="3 4 5"
export epochs=10
export dataset=CIFAR100
export merge_fns="finetune magmax masked_magmax_with_targetdata"
export similarity_metrics="labels ot_embedded"
export target_config=target_data_config
export gpu_ids="0 1 2"
export cores_per_worker=8
export results_prefix=reproducing
export ft_log_prefix=DEFAULT_NAME
export log_dir=logs/vision_lora_reuse_experiment
