#!/usr/bin/env bash
# Fine-tune every (n_splits, seed) combination for one dataset, by calling
# scripts/vision/finetune.sh once per combination.
#
# A failing combination does not stop the sweep — the failures are listed at
# the end and the exit status is non-zero. Splits whose checkpoints already
# exist are skipped inside finetune_splitted.py, so re-running resumes the
# sweep where it stopped.
#
# Everything finetune.sh reads (dataset, epochs, task_seq, dir_name, ...) can
# be overridden from the environment and is passed through, e.g.
#   dataset=ImageNetR bash scripts/vision/finetune_all.sh
#   n_splits_list="20 50" seeds="4 5" CUDA_VISIBLE_DEVICES=0 bash scripts/vision/finetune_all.sh

n_splits_list=${n_splits_list:-"5 20 50"}
seeds=${seeds:-"3 4 5"}

script_dir=$(dirname "$0")
failed=()

for n_splits in ${n_splits_list}; do
    for seed in ${seeds}; do
        if ! n_splits=${n_splits} seed=${seed} bash "${script_dir}/finetune.sh"; then
            failed+=("n_splits=${n_splits} seed=${seed}")
        fi
    done
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All combinations finished."
else
    echo "Failed (see the corresponding .out files under outs/):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
