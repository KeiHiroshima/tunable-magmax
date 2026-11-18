#!/usr/bin/env bash
# Launch one of the experiments in scripts/nlp/experiments/ exactly as it was
# run: every setting is fixed in its file, and the code state and environment
# are recorded next to the logs before the run starts. The NLP counterpart of
# scripts/vision/launch_experiment.sh.
#
# The run is detached from the terminal (nohup + setsid), so it survives a
# logout. Re-running this script resumes it: finished tasks and result JSONs
# are skipped (scripts/nlp/run_parallel_experiment.sh).
#
#   bash scripts/nlp/launch_experiment.sh tunable_ties
#   tail -f logs/nlp_tunable_ties_experiment/runner.log    # progress
#   cat logs/nlp_tunable_ties_experiment/status.*          # finished jobs
#   kill -- -$(cat logs/nlp_tunable_ties_experiment/runner.pid)   # stop everything
#
# An experiment whose settings define `phases` runs in that many steps, one
# after another: each entry is "<name> VAR=value ...", overriding the
# settings for that step. A phase starts when the previous one has finished,
# whether or not its jobs all succeeded.

set -euo pipefail
cd "$(dirname "$0")/../.."

# Internal: the detached run itself, every phase in turn, started by the
# launch at the bottom. Selected by an environment variable rather than an
# option, since test/characterization/test_cli_surface.py checks every long
# option a script uses.
if [ -n "${LAUNCH_EXPERIMENT_RUN:-}" ]; then
    # shellcheck source=/dev/null
    source "scripts/nlp/experiments/${LAUNCH_EXPERIMENT_RUN}.sh"
    if [ -z "${phases+x}" ]; then
        exec bash scripts/nlp/run_parallel_experiment.sh
    fi
    failed=0
    for entry in "${phases[@]}"; do
        read -r name overrides <<< "${entry}"
        echo "$(date '+%F %T') ===== phase ${name}: ${overrides}"
        # The overrides are shell assignments (quoted values allowed), applied
        # in a subshell so they do not carry over into the next phase.
        (
            eval "export ${overrides}"
            phase=${name} bash scripts/nlp/run_parallel_experiment.sh
        ) || failed=1
    done
    exit "${failed}"
fi

experiment=${1:?usage: launch_experiment.sh <name>, one of: $(ls scripts/nlp/experiments | sed 's/\.sh$//' | tr '\n' ' ')}
settings=scripts/nlp/experiments/${experiment}.sh
[ -f "${settings}" ] || { echo "no such experiment: ${settings}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${settings}"

mkdir -p "${log_dir}"
if [ -f "${log_dir}/runner.pid" ] && kill -0 "$(cat "${log_dir}/runner.pid")" 2>/dev/null; then
    echo "already running (pid $(cat "${log_dir}/runner.pid")); see ${log_dir}/runner.log" >&2
    exit 1
fi

# Record what is about to run: the code (commit + any uncommitted changes),
# the settings, the data and the machine. Appended, so a resumed run keeps
# the record of every launch.
stamp=$(date '+%Y%m%d-%H%M%S')
{
    echo "===== launch ${stamp}: ${settings} ====="
    echo "commit: $(git rev-parse HEAD) ($(git symbolic-ref -q HEAD || echo detached))"
    echo "uncommitted changes: $(git status -s | wc -l) files (diff: code_${stamp}.diff)"
    echo "settings:"
    grep -v '^#' "${settings}" | grep -v '^$' | sed 's/^export //; s/^/  /'
    echo "data: ${MAGMAX_OLORA_DATA_DIR:-${HOME}/O-LoRA/CL_Benchmark}"
    echo "host: $(hostname), $(nproc) cores"
    nvidia-smi -L | sed 's/^/  /'
    echo "GPU processes at launch:"
    nvidia-smi | sed -n '/Processes:/,$p' | sed 's/^/  /'
    echo "python: $(uv run python -c 'import sys, torch, transformers; print(sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)' 2>/dev/null)"
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
LAUNCH_EXPERIMENT_RUN=${experiment} nohup setsid bash scripts/nlp/launch_experiment.sh \
    >> "${log_dir}/runner.log" 2>&1 < /dev/null &
echo $! > "${log_dir}/runner.pid"
echo "started (pid $(cat "${log_dir}/runner.pid")); log: ${log_dir}/runner.log"
