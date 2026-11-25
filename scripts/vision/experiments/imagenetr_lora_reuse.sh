# ImageNet-R version of the lora_reuse experiment (docs/imagenetr_plan.md):
# full fine-tuning vs one LoRA adapter reused across splits, merged without
# merging / with MAGMAX / with Tunable MAGMAX.
#
# full is not trained: its checkpoints are the paper's, reused as they are
# (merge_only_variants), so a missing split fails the job instead of being
# trained again. Only lora-reuse is fine-tuned. Before launching, prepare the
# machine (docs/imagenetr_plan.md §2) and run
#   uv run python -m src.datasets.prepare_imagenetr
# Sourced by scripts/vision/launch_experiment.sh.

export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_DATA_DIR=/mnt/ssd/KeiHiroshima/data
export variants="full lora-reuse"
export merge_only_variants="full"
export n_splits_list="50 20 5"
export seeds="3 4 5"
export epochs=10
export dataset=ImageNetR
export merge_fns="finetune magmax masked_magmax_with_targetdata"
export similarity_metrics="labels ot_embedded"
export target_config=target_data_config
export gpu_ids="0 1 2"
export cores_per_worker=8
# Results go to the training method's directory (full, lora-reuse), next to
# CIFAR-100's under their own ImageNetR-<T> directories.
export results_prefix=
export ft_log_prefix=DEFAULT_NAME
export log_dir=logs/runs/vision/imagenetr_lora_reuse
