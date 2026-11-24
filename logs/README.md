# logs/ の構成と索引

実験の結果と実行の記録を置くディレクトリ。このファイル以外は `.gitignore` の対象で、リポジトリには入らない。どの実験のデータがどこにあるかは、このファイルで管理する。

## 構成

```
logs/
├── README.md     このファイル（構成のルールと索引）
├── results/      評価結果の JSON。集計スクリプトの入力
│   ├── vision/ViT-B-16/sequential_finetuning/class_incremental/<実験名>/<dataset>-<T>/taskseq_<順序>/<merge_fn>/
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
3. **vision の結果のディレクトリ名は実験名にする。** `scripts/vision/experiments/<名前>.sh` の `results_prefix` で決まる。
   - 同じ名前を 2 つの実験で使うと、結果を上書きしてしまう。
   - `reproducing` は 2026-09 の再現実験の名前で、新しい実験では使わない。
4. **実行の記録は `runs/<ドメイン>/<実験名>/` に置く。** 各実験スクリプトの `log_dir` で指定する。
5. **実験を 1 つ足したら、下の索引に 1 行足す。**

## 索引

| 期間 | 実験 | レポート | 結果 | 実行の記録 | 実行したマシン（`run_info.txt` の記録） |
|---|---|---|---|---|---|
| 2026-05 | 元の論文の結果（旧コード） | — | `legacy/vision/.../merging_target_data{,_cameraready,_testrun}/` | — | 不明 |
| 2026-09 | vision の再現実験（全パラメータ学習） | [reproduction_report.md](../docs/reproduction_report.md) | `results/vision/.../reproducing/` | — | 記録なし |
| 2026-10 | vision の LoRA（タスクごと）と Core Space | [vision_lora_report.md](../docs/vision_lora_report.md)、[core_space.md](../docs/core_space.md) | `results/vision/.../reproducing-lora/` | `runs/vision/lora_sweep/`、`runs/vision/lora_core_sweep/` | 記録なし |
| 2026-11 | vision の LoRA reuse | [vision_lora_reuse_report.md](../docs/vision_lora_reuse_report.md) | `results/vision/.../lora_reuse{,-lora,-lora-reuse}/` | `runs/vision/lora_reuse/` | zeus01 |
| 2026-11 | vision の Tunable TIES | [vision_tunable_ties_report.md](../docs/vision_tunable_ties_report.md) | `results/vision/.../lora_reuse{,-lora-reuse}/` の `ties`、`weighted_ties_with_targetdata` | `runs/vision/tunable_ties/` | zeus01 |
| 2026-10 | NLP の StdCL / LSB（6 手法） | [nlp_results_report.md](../docs/nlp_results_report.md)、[nlp_lsb_report.md](../docs/nlp_lsb_report.md) | `results/nlp/t5-base/{StdCL,LSB}/`、`results/nlp/t5-large/{StdCL-lora,LSB-lora}/` | `runs/nlp/sweep/`、`runs/nlp/stdout/` | 記録なし（t5-base の LoRA だけ別のマシン） |
| 2026-11 | NLP の Tunable TIES（LSB） | [nlp_tunable_ties_report.md](../docs/nlp_tunable_ties_report.md) | `results/nlp/*/LSB*/` の `weighted_ties_with_targetdata`、`results/nlp/*/LSB-lora-reuse/` | `runs/nlp/tunable_ties/` | zeus03 |

集計の CSV：

- `summaries/nlp/summary.csv`：2026-10 の NLP の集計
- `summaries/nlp/tunable_ties_lsb.csv`：2026-11 の NLP の Tunable TIES の集計

### このマシン（zeus03）にないデータ

- **t5-base の LSB-lora**（[nlp_lsb_report.md](../docs/nlp_lsb_report.md) の t5-base LoRA）：別のマシンで実行した。結果は `results/nlp/t5-base/LSB-lora/` に入るはずだが、まだコピーしていない。

## 2026-11-24 の整理の記録

`logs/` の直下にあったものを、上の構成へ移した。ファイルの中身は変えていない。移動前後で、全ファイルのチェックサムが一致することを確かめた（結果の JSON は 16,585 個）。

| 移動前 | 移動後 |
|---|---|
| `ViT-B-16/.../reproducing/` | `results/vision/.../reproducing/` |
| `ViT-B-16/.../reproducing-lora/` | `results/vision/.../reproducing-lora/` |
| `ViT-B-16_/.../reproducing/` | `results/vision/.../lora_reuse/` |
| `ViT-B-16_/.../reproducing-lora/` | `results/vision/.../lora_reuse-lora/` |
| `ViT-B-16_/.../reproducing-lora-reuse/` | `results/vision/.../lora_reuse-lora-reuse/` |
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

- **`ViT-B-16_` について**：2026-11 の vision の 2 つの実験（zeus01）は、結果を `reproducing*` に書いた。この名前は 2026-09 の再現実験と同じだが、中身は別の学習の結果である（同じ名前のファイル 1,617 個のすべてで精度が違う）。そこで、このマシン（zeus03）にコピーするときに `ViT-B-16_` として分けてあった。今回、実験名の `lora_reuse` に名前を変えた。今後の実行で同じことが起きないよう、2 つの実験スクリプトの `results_prefix` も `lora_reuse` に直した。
- **消したもの**（19 ファイル）：
  - macOS の `.DS_Store`（6 個）
  - 終わった実行の `queue*`（中身は空）、`*.lock`、`runner.pid`
  - 空のディレクトリ `DEFAULT_NAME-lora/` と `finetune_target_data/`
