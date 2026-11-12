# Tunable TIES (--merge_fn weighted_ties_with_targetdata) against plain TIES,
# on the checkpoints the lora_reuse experiment trained (full fine-tuning and
# one LoRA adapter reused across splits). Nothing is fine-tuned again: every
# split is already there, so each job only merges. MAGMAX and Tunable MAGMAX,
# the reference points, were merged by the lora_reuse experiment into the same
# results directories.
# Sourced by scripts/vision/launch_experiment.sh.

export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_DATA_DIR=/mnt/ssd/KeiHiroshima/data
export variants="full lora-reuse"
export n_splits_list="50 20 5"
export seeds="3 4 5"
export epochs=10
export dataset=CIFAR100
export merge_fns="ties weighted_ties_with_targetdata"
export similarity_metrics="labels ot_embedded"
export target_config=target_data_config
export gpu_ids="0 1 2"
export cores_per_worker=8
export heavy_jobs=""
export heavy_workers=1
# Two phases, one after the other:
#   small  T=20 and T=5 on all three GPUs, each worker pinned to its own 8 cores.
#          TIES and Tunable TIES on full fine-tuning at T=20 hold ~86 GB each
#          (estimated from T=50's measured 216 GB, assuming it scales with T),
#          so at most two of those run at once; the third GPU takes the rest.
#   large  T=50 on GPU 0 alone: one full fine-tuning job peaks at 216 GB
#          (measured), so nothing else runs beside it.
phases=(
    "small n_splits_list='20 5' heavy_jobs='full 20' heavy_workers=2"
    "large n_splits_list=50 gpu_ids=0"
)
export results_prefix=reproducing
export ft_log_prefix=DEFAULT_NAME
export log_dir=logs/vision_tunable_ties_experiment
