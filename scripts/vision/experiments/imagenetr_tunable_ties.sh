# ImageNet-R version of the tunable_ties experiment (docs/imagenetr_plan.md):
# Tunable TIES (--merge_fn weighted_ties_with_targetdata) against plain TIES,
# on the same checkpoints as imagenetr_lora_reuse — full fine-tuning (the
# paper's) and one LoRA adapter reused across splits (trained by
# imagenetr_lora_reuse). Nothing is trained: every variant is merge only, so a
# missing split fails the job. Launch after imagenetr_lora_reuse has finished.
# Sourced by scripts/vision/launch_experiment.sh.

export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_DATA_DIR=/mnt/ssd/KeiHiroshima/data
export variants="full lora-reuse"
export merge_only_variants="full lora-reuse"
export n_splits_list="50 20 5"
export seeds="3 4 5"
export epochs=10
export dataset=ImageNetR
export merge_fns="ties weighted_ties_with_targetdata"
export similarity_metrics="labels ot_embedded"
export target_config=target_data_config
export gpu_ids="0 1 2"
export cores_per_worker=8
export heavy_jobs=""
export heavy_workers=1
# Two phases, one after the other, as for CIFAR-100 (tunable_ties.sh): the
# model is the same, so so is the memory.
#   small  T=20 and T=5 on all three GPUs. TIES and Tunable TIES on full
#          fine-tuning at T=20 hold ~86 GB each, so at most two of those run
#          at once; the third GPU takes the rest.
#   large  T=50 on GPU 0 alone: one full fine-tuning job peaks at 216 GB
#          (measured on CIFAR-100), so nothing else runs beside it.
phases=(
    "small n_splits_list='20 5' heavy_jobs='full 20' heavy_workers=2"
    "large n_splits_list=50 gpu_ids=0"
)
export results_prefix=
export ft_log_prefix=DEFAULT_NAME
export log_dir=logs/runs/vision/imagenetr_tunable_ties
