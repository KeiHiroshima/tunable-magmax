# The paper's Table 3 baselines, random_mix and average, on ImageNet-R for one
# LoRA adapter reused across splits (docs/imagenetr_plan.md), merged from the
# checkpoints imagenetr_lora_reuse trained. Merge only: a missing split fails
# the job. full fine-tuning's random_mix and average are the paper's results
# (placed under logs/results/vision/.../full/ImageNetR-<T>/ beforehand), not
# run here. Launch after imagenetr_lora_reuse has finished.
# Sourced by scripts/vision/launch_experiment.sh.

export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_DATA_DIR=/mnt/ssd/KeiHiroshima/data
export variants="lora-reuse"
export merge_only_variants="lora-reuse"
export n_splits_list="50 20 5"
export seeds="3 4 5"
export epochs=10
export dataset=ImageNetR
export merge_fns="random_mix average"
export similarity_metrics="labels ot_embedded"
export target_config=target_data_config
export gpu_ids="0 1 2"
export cores_per_worker=8
export results_prefix=
export ft_log_prefix=DEFAULT_NAME
export log_dir=logs/runs/vision/imagenetr_baselines
