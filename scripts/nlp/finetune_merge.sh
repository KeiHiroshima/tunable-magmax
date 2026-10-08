#!/usr/bin/env bash
# Fine-tune one run (scripts/nlp/finetune.sh), then merge it
# (scripts/nlp/merge.sh). Both read the same variables, which can be
# overridden from the environment, e.g.
#   model=t5-large dataset=LSB task_seq=4 merge_fn=magmax bash scripts/nlp/finetune_merge.sh

set -e

script_dir=$(dirname "$0")
bash "${script_dir}/finetune.sh"
bash "${script_dir}/merge.sh"
