# NLPタスク（StdCL / LSB）の結果の集計

[postprocessing/aggregate_nlp_logs.py](../postprocessing/aggregate_nlp_logs.py) を使い、`scripts/nlp/` で実行したマージ・評価の結果を、論文 Table 2 と同じ形の表（手法 × ターゲット環境）にまとめる手順を説明する。
学習から評価までの流れは [nlp_pipeline.md](nlp_pipeline.md) を参照。

## build_comparison_table.py との違い

| | [build_comparison_table.py](../postprocessing/build_comparison_table.py) | [aggregate_nlp_logs.py](../postprocessing/aggregate_nlp_logs.py) |
|---|---|---|
| 読み込む場所 | チェックポイントの横にある結果（`$MAGMAX_BASE_DIR/checkpoints/...`） | `logs/nlp/` にコピーした結果 |
| 1回の実行で集計する範囲 | 指定した model・dataset・Order の1組 | `logs/nlp/` で見つかったすべての model・設定・Order |
| Order をまとめた表 | なし | あり（Order ごとの表に加えて、設定ごとに全 Order をまとめた表も出す） |

複数のマシンで学習した結果を1か所に集めて比べたいときは、aggregate_nlp_logs.py を使う。

## 1. 結果の JSON を `logs/nlp/` にコピーする

スクリプトは次の構成のディレクトリを読む（`logs/` は `.gitignore` の対象なので、コミットされない）。

```
logs/nlp/{model}/{scope}/ft-pattern_{order}-epochs-{e}-seed:{s}/{merge_fn}/target{id}_seed{s}.json
```

- `scope` は `StdCL` / `LSB`（full）、または `StdCL-lora` / `LSB-lora`（LoRA）。
- この構成は、チェックポイントディレクトリ `$MAGMAX_BASE_DIR/checkpoints/{model}/sequential_finetuning/nlp_classification/` の下と同じ。そのため、その下の JSON だけを model ごとにコピーすればよい。チェックポイント（`.pt`）はコピーしない。

```bash
# 例：このマシンの t5-base の結果をコピーする
rsync -a --include='*/' --include='*.json' --exclude='*' --prune-empty-dirs \
    /mnt/ssd/KeiHiroshima/tunable_magmax/checkpoints/t5-base/sequential_finetuning/nlp_classification/ \
    logs/nlp/t5-base/

# 例：別のマシンにある t5-large の結果をコピーする
rsync -a --include='*/' --include='*.json' --exclude='*' --prune-empty-dirs \
    <host>:/path/to/checkpoints/t5-large/sequential_finetuning/nlp_classification/ \
    logs/nlp/t5-large/
```

`ft-pattern_*` に当てはまらないディレクトリは読み飛ばされる。たとえば `logs/nlp/stdout/` に `outs/` の `.out` ファイルを置いておいても集計には影響しない。

**注意**：スクリプトは、見つかった各 run について、すべての手法（finetune, random_mix, average, ties, magmax, masked_magmax_with_targetdata）の、すべての target id の JSON を読む。途中までしか終わっていない run が1つでもあると、`FileNotFoundError` で止まる。その場合は、その run のディレクトリを `logs/nlp/` から外してから実行する。

## 2. 集計を実行する

```bash
# 表を標準出力に出す
uv run python postprocessing/aggregate_nlp_logs.py

# 表に加えて、全セルを縦長の CSV に保存する
uv run python postprocessing/aggregate_nlp_logs.py --out logs/nlp/summary.csv

# 全タスクを均等に混ぜた環境の列も表に加える
uv run python postprocessing/aggregate_nlp_logs.py --include_all_tasks
```

| 引数 | 既定値 | 意味 |
|---|---|---|
| `--root` | `logs/nlp` | 集計するディレクトリ |
| `--target_config` | `target_data_config_lsb` | 列の定義に使う `configs/<name>.json`。マージ時の `--target_config` と同じものを指定する |
| `--include_all_tasks` | なし | 全タスク均等の環境（LSB 設定ファイルの target 26）を `All tasks` 列として加える。`Average` 列の計算には含めない |
| `--digits` | 2 | 小数点以下の桁数（精度は % で表示） |
| `--out` | なし | 全セルを CSV に保存するパス |

## 3. 出力の読み方

`(model, scope)` の組ごとに、次の表を出力する。

1. Order ごとの表（例：`t5-base StdCL  Order 1`）
2. 全 Order をまとめた表（例：`t5-base StdCL  Orders 1,2,3 pooled`）

```
### t5-large LSB-lora  Orders 4,5,6 pooled (per seed: mean over orders)   [mean±std over seeds [3, 4, 5], accuracy %, * = best]
                    D_tar,1      D_tar,2            D_tar,3         D_tar,4         D_tar,5      Average
                 (0.5, 0.5)   (0.8, 0.2) (0.33, 0.33, 0.33) (0.4, 0.4, 0.2) (0.6, 0.2, 0.2)
Baseline        66.28±0.90   68.16±0.82         63.14±1.36      64.10±1.14      65.46±1.02   65.43±1.00
...
MAGMAX          71.85±0.69   73.07±1.08         67.58±0.51      69.26±0.44      69.87±0.65   70.33±0.56
Tunable MAGMAX  74.47±0.24*  78.71±1.99*        69.55±0.48*     70.47±0.47*     73.88±1.84*  73.42±0.63*
```

- **行**：手法。`Tunable MAGMAX` が提案手法（`masked_magmax_with_targetdata`）。
- **列**：`D_tar,k` は設定ファイルの k 番目のエントリ（混合比が同じ環境の種類）で、2行目にその混合比が出る。`Average` は5つの `D_tar` 列の平均。
- **セル**：`seed 間の平均 ± 標準偏差`（%）。`*` は各列で最も高い平均値。

### 平均と標準偏差の計算方法

[make_tables.ipynb](../postprocessing/make_tables.ipynb) や build_comparison_table.py と同じく、次の順で計算する。

1. seed ごとに、各列に属する target id（各5環境）の `overall_accuracy` を平均する。まとめた表では、さらに Order についても平均する。
2. `Average` 列も、seed ごとに5つの `D_tar` 列を平均して求める。
3. 最後に、seed ごとの値の平均と標準偏差（ddof=1）を取る。

そのため、標準偏差は **seed 間のばらつきだけ**を表す。環境間や Order 間のばらつきは含まれない。

### CSV の形式

`--out` で保存する CSV は、表の1セルを1行とした縦長の形式になる。

| 列 | 内容 |
|---|---|
| `model` | `t5-base` など |
| `setting` | `StdCL`、`LSB-lora` など（`scope`） |
| `order` | Order 番号。全 Order をまとめた表の行は `all` |
| `method` | 手法の表示名 |
| `column` | `D_tar,1` ～ `D_tar,5`、`Average`、`All tasks` |
| `mean`, `std` | 0–1 の比率（% ではない） |
| `n_seeds` | 集計に使った seed の数 |
