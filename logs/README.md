# logs/ の構成と索引

実験の結果と実行の記録を置くディレクトリ。このファイル以外は `.gitignore` の対象で、リポジトリには入らない。どの実験のデータがどこにあるかは、このファイルで管理する。

## 構成

```
logs/
├── README.md     このファイル（構成のルールと索引）
├── results/      評価結果の JSON。集計スクリプトの入力
│   ├── vision/ViT-B-16/sequential_finetuning/class_incremental/<学習の方式>/<dataset>-<T>/taskseq_<順序>/<merge_fn>/
│   └── nlp/<model>/<設定>/ft-pattern_<order>-epochs-<e>-seed:<s>/<merge_fn>/
├── runs/         実行の記録（run ごとのログ、run_info.txt、実行時のコードの差分、標準出力）
│   ├── vision/<実験名>/
│   └── nlp/<実験名>/
├── summaries/    集計した CSV
│   └── nlp/
└── legacy/       今の実験では使わない古い結果（構成は results/ と同じ）
```

## ルール

1. **結果は `results/`、実行の記録は `runs/`、集計は `summaries/` に分ける。** 古くて使わない結果は `legacy/` に移す。
2. **`results/` の下は、ツールが書き出すパスをそのまま保つ。**
   - vision：`scripts/vision/merge.sh` が `logs/results/vision/<model>/...` に書く。
   - NLP：結果はチェックポイントの横に書かれるので、`rsync` で `logs/results/nlp/<model>/` にコピーする（[docs/nlp_aggregation.md](../docs/nlp_aggregation.md)）。
3. **vision の結果は、学習の方式のディレクトリに置く。** 方式ごとに、正とする学習を 1 つだけ置く。
   - 学習の方式は `full`（全パラメータ学習）、`lora-merge`（タスクごとに LoRA を付け、学習後に重みへ足し込む。O-LoRA の方式）、`lora-reuse`（1 つの LoRA を全タスクで学習し続ける）。`scripts/vision/merge.sh` と `run_parallel_experiment.sh` は、既定でこの名前に書く。
   - **同じ方式を学習し直す前に、今の結果を `legacy/` に移す。** 同じディレクトリに書くと、ファイル名が同じなので上書きされるか、終わった環境として飛ばされる。
   - 1 つのディレクトリの中の手法は、すべて同じチェックポイントからマージしたものにする。別の学習の結果を混ぜると、手法どうしを比べられなくなる。
   - 試運転は `results_prefix` を付けて、`<results_prefix>/<学習の方式>/` に分ける。
   - `reproducing` は、この規則を決める前（2026-09）の実験の名前で、そのまま残している（下の「残っている課題」）。
4. **実行の記録は `runs/<ドメイン>/<実験名>/` に置く。** 各実験スクリプトの `log_dir` で指定する。
5. **実験を 1 つ足したら、下の索引に 1 行足す。**

## 索引

| 期間 | 実験 | レポート | 結果 | 実行の記録 | 実行したマシン（`run_info.txt` の記録） |
|---|---|---|---|---|---|
| 2026-05 | 元の論文の結果（旧コード） | — | `legacy/vision/.../merging_target_data{,_cameraready,_testrun}/` | — | 不明 |
| 2026-09 | vision の再現実験（全パラメータ学習） | [reproduction_report.md](../docs/reproduction_report.md) | `results/vision/.../reproducing/` | — | 記録なし |
| 2026-10 | vision の LoRA（タスクごと）と Core Space | [vision_lora_report.md](../docs/vision_lora_report.md)、[core_space.md](../docs/core_space.md) | `legacy/vision/.../reproducing-lora/` | `runs/vision/lora_sweep/`、`runs/vision/lora_core_sweep/` | 記録なし |
| 2026-11 | vision の LoRA reuse | [vision_lora_reuse_report.md](../docs/vision_lora_reuse_report.md) | `results/vision/.../{full,lora-merge,lora-reuse}/` | `runs/vision/lora_reuse/` | zeus01 |
| 2026-11 | vision の Tunable TIES | [vision_tunable_ties_report.md](../docs/vision_tunable_ties_report.md) | `results/vision/.../{full,lora-reuse}/` の `ties`、`weighted_ties_with_targetdata` | `runs/vision/tunable_ties/` | zeus01 |
| 2026-10 | NLP の StdCL / LSB（6 手法） | [nlp_results_report.md](../docs/nlp_results_report.md)、[nlp_lsb_report.md](../docs/nlp_lsb_report.md) | `results/nlp/t5-base/{StdCL,LSB}/`、`results/nlp/t5-large/{StdCL-lora,LSB-lora}/` | `runs/nlp/sweep/`、`runs/nlp/stdout/` | 記録なし（t5-base の LoRA だけ別のマシン） |
| 2026-11 | NLP の Tunable TIES（LSB） | [nlp_tunable_ties_report.md](../docs/nlp_tunable_ties_report.md) | `results/nlp/*/LSB*/` の `weighted_ties_with_targetdata`、`results/nlp/*/LSB-lora-reuse/` | `runs/nlp/tunable_ties/` | zeus03 |

集計の CSV：

- `summaries/nlp/summary.csv`：2026-10 の NLP の集計
- `summaries/nlp/tunable_ties_lsb.csv`：2026-11 の NLP の Tunable TIES の集計

### 残っている課題

- **`results/vision/.../reproducing/`（2026-09、全パラメータ学習）**：`full/`（2026-11）と同じ方式の、別の学習の結果である。ルール 3 に従えば `legacy/` に移すところだが、`full/` にない random_mix と average の結果がここにしかないので、まだ移していない。`full/` のチェックポイント（zeus01）で random_mix と average をマージすれば、`full/` だけで足りるようになり、`reproducing/` を `legacy/` に移せる。

### このマシン（zeus03）にないデータ

- **t5-base の LSB-lora**（[nlp_lsb_report.md](../docs/nlp_lsb_report.md) の t5-base LoRA）：別のマシンで実行した。結果は `results/nlp/t5-base/LSB-lora/` に入るはずだが、まだコピーしていない。

## 2026-11-24 の整理の記録

`logs/` の直下にあったものを、上の構成へ移した。表は最終的な置き場所を示す（いったん `lora_reuse/<学習の方式>/` に置いてから、階層をなくした）。`reproducing-lora`（2026-10、lora-merge）は、同じ方式の 2026-11 の学習（`lora-merge/`）があるので `legacy/` に移した。ファイルの中身は変えていない。移動前後で、全ファイルのチェックサムが一致することを確かめた（結果の JSON は 16,585 個）。

| 移動前 | 移動後 |
|---|---|
| `ViT-B-16/.../reproducing/` | `results/vision/.../reproducing/` |
| `ViT-B-16/.../reproducing-lora/` | `legacy/vision/.../reproducing-lora/` |
| `ViT-B-16_/.../reproducing/` | `results/vision/.../full/` |
| `ViT-B-16_/.../reproducing-lora/` | `results/vision/.../lora-merge/` |
| `ViT-B-16_/.../reproducing-lora-reuse/` | `results/vision/.../lora-reuse/` |
| `ViT-B-16/.../merging_target_data{,_cameraready,_testrun}/` | `legacy/vision/.../`（同じ名前） |
| `nlp/t5-base/`、`nlp/t5-large/` | `results/nlp/t5-base/`、`results/nlp/t5-large/` |
| `nlp/stdout/` | `runs/nlp/stdout/` |
| `nlp/summary.csv` | `summaries/nlp/summary.csv` |
| `nlp/tunable_ties_lsb_summary.csv` | `summaries/nlp/tunable_ties_lsb.csv` |
| `vision_lora_sweep/` | `runs/vision/lora_sweep/` |
| `vision_lora_core_sweep/` | `runs/vision/lora_core_sweep/` |
| `vision_lora_reuse_experiment/` | `runs/vision/lora_reuse/` |
| `vision_tunable_ties_experiment/` | `runs/vision/tunable_ties/` |
| `nlp_sweep/` | `runs/nlp/sweep/` |
| `nlp_tunable_ties_experiment/` | `runs/nlp/tunable_ties/` |

`...` は `ViT-B-16/sequential_finetuning/class_incremental` を指す。

- **`ViT-B-16_` について**：2026-11 の vision の 2 つの実験（zeus01）は、結果を `reproducing*` に書いた。この名前は 2026-09 の再現実験と同じだが、中身は別の学習の結果である（同じ名前のファイル 1,617 個のすべてで精度が違う）。そこで、このマシン（zeus03）にコピーするときに `ViT-B-16_` として分けてあった。今回、学習の方式の名前（`full`、`lora-merge`、`lora-reuse`）に変えた。今後の実行で同じことが起きないよう、`merge.sh` と `run_parallel_experiment.sh` が既定でこの名前に書くようにした。
- **消したもの**（19 ファイル）：
  - macOS の `.DS_Store`（6 個）
  - 終わった実行の `queue*`（中身は空）、`*.lock`、`runner.pid`
  - 空のディレクトリ `DEFAULT_NAME-lora/` と `finetune_target_data/`
