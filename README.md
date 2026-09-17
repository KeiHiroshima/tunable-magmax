# Tunable MAGMAX: Preference-Aware Model Merging for Continual Learning ([arXiv](https://arxiv.org/abs/2605.20803))

This is the official repository for the paper:

> **Tunable MAGMAX: Preference-Aware Model Merging for Continual Learning**<br>
> Kei Hiroshima, Kento Uchida, Shinichi Shirakawa, Yokohama National University<br>
> International Conference on Pattern Recognition 2026

<p align="center">
<img style="width:80%;" alt="thumbnail" src="./img/problem_setting.png">
</p>

> **Abstract:** Continual learning (CL) aims to train models sequentially on multiple tasks while mitigating catastrophic forgetting of previously learned knowledge. Recent advances in large pre-trained models (LPMs) and model merging techniques, such as MAGMAX, have demonstrated effective CL performance by combining task-specific parameters. However, existing methods primarily focus on average performance across all tasks and do not adequately address how to construct models accommodating different deployment environments or varying user preferences. This paper proposes a model merging framework, termed Tunable MAGMAX, which enables preference-aware control of task-specific performance in CL. Our method introduces a preference vector that controls the number of elements selected from each task vector during model merging, allowing us to adjust the merged model performance according to their deployment needs. We further propose a method for automatically constructing appropriate preference vectors by leveraging small amounts of target environment data and datasets from model training tasks, thereby eliminating the need for manual specification. The experimental result on CL benchmark tasks demonstrates that Tunable MAGMAX effectively controls task-wise performance and successfully adapts merged models to various target environments. The proposed Tunable MAGMAX achieves superior or comparable performance to baseline methods, making it a practical solution for deploying CL models to various environments where the preferences of each task performance differ.


## Installation

Install with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```
This installs PyTorch from the CUDA 12.8 wheel index configured in `pyproject.toml` (`[tool.uv.sources]` / `[[tool.uv.index]]`). If your GPU/driver needs a different CUDA version, edit that index URL before running `uv sync`. All commands below are run as `uv run python ...` instead of `python ...`.

The vision pipeline (CIFAR100/ImageNetR) also works under conda instead:
```bash
conda env create
conda activate magmax
```
If that does not work, the env was created by the following commands:
```bash
conda create --name magmax python=3.10
conda activate magmax
```


## Usage

Two entry points, both thin: they parse arguments and dispatch on `--dataset` to
a backend that implements `finetune(args)` / `merge_and_evaluate(args)`
(`src/backends/registry.py`).

* `finetune_splitted.py` — trains one model per task, saving a checkpoint each
* `merge_for_targetdata.py` — merges those checkpoints and evaluates the result

| `--dataset` | Backend | Backbone |
|---|---|---|
| `CIFAR100`, `ImageNetR` | `vision_backend.py` | CLIP ViT |
| `LSB` | `nlp_classification_backend.py` | BERT + per-task head |
| `CITB19`, `CITB38` | `nlp_seq2seq_backend.py` | BERT2BERT |

### Step 0: Configure paths

```bash
export MAGMAX_BASE_DIR=/path/to/checkpoints   # where checkpoints are written
export MAGMAX_DATA_DIR=/path/to/data          # where the vision datasets live
```

Or edit the defaults in `src/config.py`. `MAGMAX_DATA_DIR` is unused by the NLP
backends, which stream from the Hugging Face Hub (cached under `HF_HOME`).

### Step 1: Fine-tuning

```bash
bash scripts/finetune.sh                 # edit the variables at the top first
```

Or directly:

```bash
uv run python finetune_splitted.py \
    --model ViT-B-16 --dataset CIFAR100 --n_splits 5 \
    --split_strategy class --sequential-finetuning \
    --epochs 10 --seed 3 --taskseq_pattern A --batch_size 32
```

Checkpoints land in
`$MAGMAX_BASE_DIR/checkpoints/{model}/[sequential_finetuning/]{group}/{scope}/ft-pattern_{p}-epochs-{e}-seed:{s}/finetuned_{i}.pt`.
A run that is interrupted can be resumed by re-issuing the same command:
checkpoints that already exist are skipped, and the next task still continues
from the one before it.

> **The `scripts/*.sh` wrappers call bare `python`.** Activate the environment
> first (`source .venv/bin/activate`, or `conda activate magmax`) — `uv run bash
> scripts/...` does not put the virtualenv on `PATH` for the inner call.

### Step 2: Merging and evaluation

```bash
bash scripts/merge.sh
```

Or directly:

```bash
uv run python merge_for_targetdata.py \
    --model ViT-B-16 --dataset CIFAR100 --n_splits 5 \
    --split_strategy class --sequential-finetuning \
    --epochs 10 --seed 3 --taskseq_pattern A --batch_size 32 \
    --merge_fn masked_magmax_with_targetdata --similarity_metric labels \
    --num_train_data_each_task 500 --num_target_data 1000 \
    --target_config target_data_config --results_db logs/my-run
```

`--merge_fn` selects the strategy; they are all defined in one place,
`src/merging/registry.py`:

| `--merge_fn` | Method | NLP support |
|---|---|---|
| `masked_magmax_with_targetdata` | **Tunable MAGMAX (proposed)** | `LSB` only |
| `magmax` | MAGMAX | yes |
| `ties` | TIES-Merging | yes |
| `average` | Model Soup | yes |
| `random_mix` | Rand Mix | yes |
| `finetune` | Baseline: keep the last task's model, no merging | yes |

`--similarity_metric` applies to `masked_magmax_with_targetdata` only and picks
how the preference vector is estimated: `labels` (label-distribution
similarity), or `ot_embedded` / `cosine_embedded` / `mmd_embedded` (feature-space
distances). The raw `cosine` / `mmd` / `ot` variants skip the encoder and
compare pixels directly.

### Target environments

A target environment is a mixture of some of the trained tasks, in some ratio.
`--target_config NAME` reads `configs/NAME.json`:

```json
{"dataset_configs": [
  {"num_task_to_be_fetched": 2,          // -1 means every task
   "ratio_task_to_be_fetched": [0.8, 0.2], // [-1] means an even split
   "variants": [{"target_id": 1, "random_seed": 42}]}
]}
```

Each entry describes one *kind* of environment; its `variants` are independent
draws of that kind, each identified by `target_id` and sampled with its own
`random_seed`. Shipped configs:

| Config | Contents |
|---|---|
| `target_data_config` | 26 environments: 5 mixing ratios x 5 seeds, plus one all-tasks entry |
| `target_data_config_lsb` | the same shape, for LSB |
| `target_data_config_split{5,20,50}` | sweeps over how many tasks a mixture draws from; sized per `--n_splits` |
| `target_data_config_smoke` | 3 environments, for a quick end-to-end check |

`--num_target_data` sets how many examples an environment holds (the
`split{N}` configs carry their own sizes and override it). 10% of each task's
draw is held out as the meta dataset the preference vector is estimated from.

### Where results go

**Vision** writes one JSON per target environment under `--results_db`:

```
${results_db}/{merge_fn}/[{similarity_metric}/]{merge_fn}_lambda0.5_[{metric}_]target{id}_seed{s}.json
```

It holds `overall_accuracy` (micro-averaged over examples), `average_accuracy`
(mean over tasks), `taskwise_accuracies`, the preference vector, the target
environment's composition, and — for the proposed method — `num_unaligned` /
`num_params_all`.

**NLP ignores `--results_db`** and writes into the checkpoint directory instead:

```
.../ft-pattern_{p}-epochs-{e}-seed:{s}/
├── merge_{merge_fn}_results.json              # baselines: per-task accuracy only
└── masked_magmax_with_targetdata/
    └── target{id}_seed{s}.json                # proposed: full record per environment
```

### Combined run

```bash
bash scripts/finetune_merge.sh
```

### Key arguments

| Flag | Default | Notes |
|---|---|---|
| `--model` | — | `ViT-B-16` / `ViT-L-14`, or a Hugging Face name for NLP |
| `--dataset` | — | see the backend table above |
| `--n_splits` | 2 | vision only: how many class-incremental tasks |
| `--split_strategy` | — | vision only; `class` for the paper's setting |
| `--sequential-finetuning` | off | continue each task from the previous one. **The paper's setting** — without it every task restarts from the pre-trained model |
| `--taskseq_pattern` | `A` | fixed task order; `B`/`C` are deterministic reshuffles |
| `--epochs` | 10 | |
| `--batch_size` | 128 | the paper's value. **ViT-B/16 at 128 needs more than 16 GB** — pass `--batch_size 32` on a smaller card |
| `--lr` | 1e-5 | |
| `--warmup_ratio` | 0.1 | see below |
| `--seed` | 5 | the paper uses 3/4/5 |
| `--num_train_data_each_task` | — | **required when merging**: how many training examples per task the similarity is measured against |
| `--num_target_data` | 1000 | examples per target environment |
| `--target_config` | `target_data_config` | see above |
| `--results_db` | — | where vision writes results |
| `--gpu_id` | 0 | picks the device used for *evaluation* (`cuda:N`) |

Training wraps the model in `DataParallel` across every visible GPU regardless
of `--gpu_id`, so use `CUDA_VISIBLE_DEVICES` to restrict which cards it uses.

### Warmup

`--warmup_ratio` (default `0.1`) sets the fraction of each task's schedule spent
in linear warmup before the cosine decay starts. The same value works for every
setting, so there is normally no reason to pass it.

It is a fraction rather than a step count because task lengths vary by orders of
magnitude — 40 optimizer steps per task for ImageNet-R-50 against 790 for
CIFAR-100-5, and within one LSB run from 80 steps for `cb` to 203,000 for
`yelp`. A step count that suits one of those covers another's whole schedule,
which leaves the learning rate ramping linearly from zero, never reaching `--lr`
and never decaying. A ratio also rescales by itself when `--epochs` or
`--batch_size` change.

Values outside `(0, 1)` are rejected at the command line, since a ratio of 1 or
more is exactly the broken schedule described above.


## NLP tasks

| `--dataset` | Benchmark | Backbone | Proposed method |
|---|---|---|---|
| `LSB` | Long Sequence Benchmark — 15 text classification tasks | `BertClassifier` | supported |
| `CITB19` | CITB InstrDialog — 19 instruction-following tasks | BERT2BERT | not yet |
| `CITB38` | CITB InstrDialog++ — 38 tasks | BERT2BERT | not yet |

```bash
uv run python finetune_splitted.py \
    --model bert-base-uncased --dataset LSB \
    --epochs 3 --taskseq_pattern A --seed 3 --sequential-finetuning

uv run python merge_for_targetdata.py \
    --model bert-base-uncased --dataset LSB \
    --epochs 3 --taskseq_pattern A --seed 3 --sequential-finetuning \
    --merge_fn masked_magmax_with_targetdata \
    --target_config target_data_config_lsb --num_target_data 200
```

Unlike vision, the NLP target environments need no similarity estimate: the
tasks are separate datasets, so which task an example came from is known by
construction and the preference vector *is* the mixing ratio
(`src/nlp/target_data.py`).

The seq2seq backends (`CITB19`/`CITB38`) evaluate with generation loss rather
than accuracy, and support the baselines only.

`--taskseq_pattern` (`A`/`B`/`C`) selects a fixed task order, defined in
`TASK_ORDER_PATTERNS` in `src/nlp/long_sequence_benchmark.py` /
`src/nlp/citb_superni.py`.

## Tests

```bash
uv run pytest test/ -q
```

Around 240 tests, roughly 20 seconds. Most need no GPU, no network and no
dataset: they run the real code against synthetic task vectors and stand-in
datasets, and cover the checkpoint layout, the merge registry, the learning-rate
schedule, target-environment construction, and the fine-tuning loop's
resume behaviour. A handful (`test/test_nlp_*.py`) do download a model and a few
datasets on first run, cached under `datasets/hf_cache`.


## Third-Party Code

### MAGMAX
Source: https://github.com/danielm1405/magmax<br>
Paper: Marczak, D., Twardowski, B., Trzciński, T., & Cygert, S. (2024).<br>
[MagMax: Leveraging Model Merging for Seamless Continual Learning](http://arxiv.org/abs/2407.06322). ECCV2024<br>
License: No license (as of 2026-05-20). Used with the intent to comply with any future license.<br>
Files: src/modeling.py, src/heads.py, src/merging/task_vector.py, src/merging/ties.py,
       src/datasets/imagenetr.py, src/datasets/registry.py, src/args.py,
       src/utils.py, src/trainer.py, src/eval.py, src/datasets/common.py, src/datasets/cifar100.py
Modified files: `finetune_spilitted.py, merge_for_targetdata.py, and files in src/`

### perceptionCLIP
Source: https://github.com/umd-huang-lab/perceptionCLIP<br>
Paper: An, B., Zhu, S., Panaitescu-Liess, M.-A., Mummadi, C. K., & Huang, F. (2024).<br>
[PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts](https://arxiv.org/abs/2308.01313). ICLR 2024<br>
License: MIT License (Copyright (c) 2023 CMU Locus Lab)<br>
Files: `src/datasets/templates.py (partial)`

### TIES-Merging
Source: https://github.com/prateeky2806/ties-merging<br>
Paper: Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023).<br>
[TIES-Merging: Resolving Interference When Merging Models](http://arxiv.org/abs/2306.01708). NeurIPS 2023<br>
License: BSD 3-Clause License (Copyright (c) 2022 Salesforce, Inc.)<br>
Files: `src/merging/ties.py`


## Citation
If you find this work useful, please consider citing it:
```bibtex
@inproceedings{hiroshima2026tunablemagmax,
    title     = {Tunable {MAGMAX}: Preference-Aware Model Merging for Continual Learning},
    author    = {Kei Hiroshima and Kento Uchida and Shinichi Shirakawa},
    booktitle = {International Conference on Pattern Recognition}
    year      = {2026}
}
```
