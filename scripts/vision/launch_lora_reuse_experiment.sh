#!/usr/bin/env bash
# Launch the --lora_chain reuse experiment (docs/lora_reuse_plan.md §5) exactly
# as it was run: every setting below is fixed here, and the code state and
# environment are recorded next to the logs before the run starts.
#
# The run is detached from the terminal (nohup + setsid), so it survives a
# logout. Re-running this script resumes it: finished splits and result JSONs
# are skipped (scripts/vision/run_lora_reuse_experiment.sh).
#
#   bash scripts/vision/launch_lora_reuse_experiment.sh
#   tail -f logs/vision_lora_reuse_experiment/runner.log    # progress
#   cat logs/vision_lora_reuse_experiment/status            # finished jobs
#   kill -- -$(cat logs/vision_lora_reuse_experiment/runner.pid)   # stop everything

set -euo pipefail
cd "$(dirname "$0")/../.."

# --- the experiment --------------------------------------------------------
# Checkpoints go to a fresh base directory; only the zero-shot model and the
# open_clip cache are linked from the earlier one (see run_info.txt).
export MAGMAX_BASE_DIR=/mnt/ssd/KeiHiroshima/tunable_magmax
export MAGMAX_DATA_DIR=/mnt/ssd/KeiHiroshima/data
export variants="full lora-reuse lora"        # main: full vs lora-reuse; ablation: lora vs lora-reuse
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
# ---------------------------------------------------------------------------

mkdir -p "${log_dir}"
if [ -f "${log_dir}/runner.pid" ] && kill -0 "$(cat "${log_dir}/runner.pid")" 2>/dev/null; then
    echo "already running (pid $(cat "${log_dir}/runner.pid")); see ${log_dir}/runner.log" >&2
    exit 1
fi

# Record what is about to run: the code (commit + any uncommitted changes),
# the settings above, and the machine. Appended, so a resumed run keeps the
# record of every launch.
stamp=$(date '+%Y%m%d-%H%M%S')
{
    echo "===== launch ${stamp} ====="
    echo "commit: $(git rev-parse HEAD) ($(git symbolic-ref -q HEAD || echo detached))"
    echo "uncommitted changes: $(git status -s | wc -l) files (diff: code_${stamp}.diff)"
    echo "settings:"
    for v in MAGMAX_BASE_DIR MAGMAX_DATA_DIR variants n_splits_list seeds epochs dataset \
             merge_fns similarity_metrics target_config gpu_ids cores_per_worker \
             results_prefix ft_log_prefix log_dir; do
        echo "  ${v}=${!v}"
    done
    echo "base dir links:"
    ls -l "${MAGMAX_BASE_DIR}/checkpoints/ViT-B-16" | grep -- '->' | sed 's/^/  /'
    echo "host: $(hostname), $(nproc) cores"
    nvidia-smi -L | sed 's/^/  /'
    echo "python: $(uv run python -c 'import sys, torch, open_clip; print(sys.version.split()[0], "torch", torch.__version__, "open_clip", open_clip.__version__)' 2>/dev/null)"
    echo
} >> "${log_dir}/run_info.txt"
# Short options only: test/characterization/test_cli_surface.py checks every
# long option a script uses against the CLI fixtures.
git diff HEAD > "${log_dir}/code_${stamp}.diff"
git status -s -uall | sed -n 's/^?? //p' | while read -r f; do
    echo "===== untracked file: ${f}"
    cat "${f}"
done >> "${log_dir}/code_${stamp}.diff"

# setsid gives the run its own process group, so the kill above stops every
# worker with it.
nohup setsid bash scripts/vision/run_lora_reuse_experiment.sh \
    >> "${log_dir}/runner.log" 2>&1 < /dev/null &
echo $! > "${log_dir}/runner.pid"
echo "started (pid $(cat "${log_dir}/runner.pid")); log: ${log_dir}/runner.log"
