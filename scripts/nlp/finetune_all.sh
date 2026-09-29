#!/usr/bin/env bash
# Fine-tune LSB for every seed, by calling scripts/nlp/finetune.sh once per
# seed. LSB has a single task sequence (no n_splits), so seed is the only axis
# swept here — unlike scripts/vision/finetune_all.sh.
#
# A failing seed does not stop the sweep — the failures are listed at the end
# and the exit status is non-zero. Tasks whose checkpoints already exist are
# skipped inside finetune_splitted.py, so re-running resumes the sweep.
#
# Everything finetune.sh reads (epochs, task_seq, gpu_id, ...) can be
# overridden from the environment and is passed through, e.g.
#   epochs=10 seeds="4 5" gpu_id=1 bash scripts/nlp/finetune_all.sh

seeds=${seeds:-"3 4 5"}

script_dir=$(dirname "$0")
failed=()

for seed in ${seeds}; do
    if ! seed=${seed} bash "${script_dir}/finetune.sh"; then
        failed+=("seed=${seed}")
    fi
done

echo "======================================================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All seeds finished."
else
    echo "Failed (see the corresponding .out files under outs/):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
