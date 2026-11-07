# 実装計画：単一の LoRA を全タスクで使い回す（`--lora_chain reuse`）

LoRA の逐次学習で、タスクごとに新しいアダプタを付けて前のアダプタを重みに足し込む今の方式に加えて、1 つの LoRA 行列（A, B）を全タスクで学習し続ける方式を追加する計画をまとめる。

- 状態：**計画のみ。実装はまだしていない。**
- ブランチ：`exp/lora-single-adapter`（`origin/scaling` の b09594c から作成）
- 関連文書：[vision_lora_report.md](vision_lora_report.md)（今の方式の vision の結果）、[core_space.md](core_space.md)（Core Space でのマージ）

## 1. 目的

今の LoRA の逐次学習は O-LoRA（式 9）に従っている。タスク t の学習を始める前に、それまでのアダプタをすべて事前学習済みの重み W₀ に足し込み、新しいアダプタを付けて学習する。このため、タスク t のタスクベクトル τ_t はアダプタ 0..t の累積和になり、ランクは最大 t·r まで大きくなる。

この計画では、もう 1 つの逐次学習の方式を追加する。

- 1 つの LoRA 行列（A, B）をタスク 2 以降でも使い回し、前のタスクの最終値から学習を続ける。
- タスクが切り替わるときに、W₀ に直前のタスクの LoRA を足し込むことはしない。モデルは常に W₀ + (α/r)·B A の形になる。

こうすると、どのタスクのタスクベクトルもランク r 以下の 1 つの LoRA になる。2 つの方式で学習したモデルに同じマージ手法をかけて、逐次学習のやり方が、マージ手法（MAGMAX、提案手法など）と Core Space でのマージの結果をどう変えるかを比べられるようにする。

既定の動作（今の方式）は変えない。新しい方式は、オプションを指定したときだけ使う。

## 2. 今の方式と新しい方式の違い

| | 今の方式（`merge`、O-LoRA 式 9） | 新しい方式（`reuse`） |
|---|---|---|
| タスク t の開始時のモデル | W₀ + Σ_{j<t} (α/r)B_jA_j（足し込み済み） | **W₀ のまま**（足し込まない） |
| タスク t の開始時の LoRA | 新しい A（Kaiming 初期化）、B = 0 | **タスク t−1 の最終値の A、B** から続けて学習する |
| タスク t の後のモデル | W₀ + Σ_{j≤t} (α/r)B_jA_j | W₀ + (α/r)B_tA_t |
| タスクベクトル τ_t | アダプタ 0..t の累積和 | **(α/r)B_tA_t（そのアダプタだけ）** |
| τ_t のランク | 最大 t·r | 最大 r |
| 保存するもの | タスクごとのアダプタ `adapter_{t}.pt` | 同じ（中身はタスク t の後の A、B） |

どちらでも、τ_t は「タスク t の後のモデル − W₀」である。タスクベクトルの意味は変わらないので、マージ関数、類似度の重み、評価は変更しない。

## 3. 実装内容

### 3.1 オプション（[src/args.py](../src/args.py)）

- `--lora_chain {merge,reuse}` を追加する。既定値は `merge`（今の動作）。
- 次の組み合わせは起動時にエラーにする。
  - `--lora_chain reuse` で `--finetune_mode lora` でないとき。
  - `--lora_chain reuse` で `--sequential-finetuning` がないとき。独立に学習する場合は、どちらの方式でも毎回 W₀ から新しいアダプタで学習するので、`reuse` の意味がない。
- `--finetune_mode` に新しい値（例：`lora_reuse`）を足す方法も考えられる。しかし `finetune_mode == "lora"` を確認している箇所が 10 か所ほどある（学習率の既定値、Core Space の確認、`_lora()` など）。値を増やすとこれらの漏れが起きやすいので、別のオプションにする。

### 3.2 LoRA モジュール（[src/lora.py](../src/lora.py)）

- `load_adapter_into(model, adapter)` を追加する。保存したアダプタの A、B を、注入済みの `LoRALinear` にコピーする。キーと形が一致することを確認する。
- `extract_adapter` が保存する辞書に `"chain": "merge" | "reuse"` を書き込む。
- `adapter_task_vectors(adapters, sequential, chain)` に `chain` を追加する。`reuse` のときは、各アダプタの差分 (α/r)B_tA_t をそのまま返す。
- 読み込んだアダプタの `chain` が実行時のオプションと一致することを確認する。別の方式で学習したアダプタを取り違えてマージするのを防ぐため。

### 3.3 学習ループ

NLP（[src/nlp/finetune_nlp.py](../src/nlp/finetune_nlp.py)）と vision（[src/backends/vision_backend.py](../src/backends/vision_backend.py) の `_run_sequential_finetuning`）を同じように変更する。

`reuse` のときの各タスクの手順：

1. W₀ を読み込む。`merge_adapter_into` でそれまでのアダプタを足し込む処理は**しない**。
2. `inject_lora` で LoRA を付ける。
3. タスク t ≥ 1 では、`load_adapter_into` で `adapter_{t-1}.pt` の A、B を読み込む。タスク 0 は今と同じ初期化（A は Kaiming、B = 0）。
4. 学習して、`adapter_{t}.pt` に保存する。

補足：

- **中断からの再開**：学習済みのタスクは今と同じく飛ばす。次のタスクは、直前に保存されたアダプタ `adapter_{t-1}.pt` から始める。今のコードが守っている「直前のタスクから続ける」という性質（`finetune_nlp.py` の不変条件 I3）を、`reuse` でも守る。
- **オプティマイザの状態**：AdamW のモーメントとスケジューラは、今と同じくタスクごとに作り直す。引き継ぐのはパラメータの値だけ。

### 3.4 保存先を分ける

- 2 つの方式は、どちらも `adapter_{idx}.pt` という同じ名前で保存する。同じディレクトリを使うと、`reuse` の実行が今の方式のアダプタを学習済みとみなして学習を飛ばしてしまう。そのため、保存先を分ける。

| 対象 | 今の方式 | `reuse` |
|---|---|---|
| NLP のチェックポイント（`_ckpt_dir`） | `{dataset}-lora` | `{dataset}-lora-reuse` |
| vision のチェックポイント（`_ckpt_dir`） | `{scope}-lora` | `{scope}-lora-reuse` |
| vision の結果（`scripts/vision/merge.sh` の `dir_name`） | `reproducing-lora` | `reproducing-lora-reuse` |

- NLP の結果はチェックポイントの横に保存され、postprocessing はそのパスを手で組み立てている（`postprocessing/` は `src/` を import しない）。このため、[postprocessing/results.py](../postprocessing/results.py) の `nlp_result_path`、[build_comparison_table.py](../postprocessing/build_comparison_table.py)、[aggregate_nlp_logs.py](../postprocessing/aggregate_nlp_logs.py) にも `--lora_chain` を追加する。

### 3.5 マージと Core Space

- マージ側の変更は、vision と NLP の `merge_and_evaluate` が `chain` を指定して `adapter_task_vectors` を呼ぶことだけ。
- `CoreSpace.from_adapters`（[src/merging/core_space.py](../src/merging/core_space.py)）は変更しない。基底は今と同じく、全タスクの B を横に、A を縦に並べた行列の SVD から作る。`reuse` では τ_t = B_tA_t がその張る空間に入るので、回転して戻しても情報は失われない。今の docstring は「τ_i はアダプタ 0..i の累積和」という前提で書かれているので、その説明だけを直す。

### 3.6 スクリプトと文書

- `scripts/{vision,nlp}/{finetune,merge}.sh`、`scripts/vision/run_lora_sweep.sh`、`scripts/nlp/run_t5-base_lora_lsb_single_gpu.sh` に、環境変数 `lora_chain`（既定値 `merge`）を追加する。
- README のオプションの表に `--lora_chain` を追加する。
- この文書を、実装した内容に合わせて更新する（2 つの方式の違いの説明として残す）。

## 4. テスト内容

[test/test_nlp_lora.py](../test/test_nlp_lora.py) と [test/test_vision_lora.py](../test/test_vision_lora.py) に追加する。小さなモデルで動かし、GPU は使わない。

| テスト | 確認すること |
|---|---|
| 開始時の状態 | タスク t ≥ 1 の学習開始時に、ベースの重みが W₀ と一致し（足し込まれていない）、A、B がタスク t−1 の最終値と一致する |
| タスクベクトル | τ_t がアダプタ t だけの差分 (α/r)B_tA_t と一致する。W₀ + τ_t で作ったモデルの出力が、学習を終えたときのモデルの出力と一致する |
| 中断からの再開 | 途中のタスクまでのアダプタがある状態から再開すると、最初から通して実行したときと同じアダプタから各タスクが始まる |
| 保存先の分離 | `merge` と `reuse` のチェックポイントのディレクトリが重ならない |
| 取り違えの防止 | `chain` が実行時のオプションと違うアダプタを読み込むとエラーになる |
| オプションの確認 | `reuse` を `--finetune_mode full` や `--sequential-finetuning` なしで指定するとエラーになる |
| Core Space | `reuse` のアダプタから作った基底で、τ_t を回転して戻すと元に戻る |
| 既定の動作 | 今の `merge` のテストが変更なしで通る |

`test/characterization/test_cli_surface.py` など、オプションの一覧を確認しているテストも合わせて更新する。

## 5. 動作確認と実験

1. 単体テストがすべて通ることを確認する。
2. vision の小さな予備実験：CIFAR-100-5、seed 3。設定は今の LoRA の実験と同じ（q・v、r = 8、α = 32、lr 1e-3、10 epoch）。学習が進むことと、マージまで通ることを確かめる。
3. 本実験：CIFAR-100 の T = 5 / 20 / 50 × seed 3 / 4 / 5。今の方式の結果（[vision_lora_report.md](vision_lora_report.md)）と、`build_comparison_table.py` で同じ形の表にして比べる。

## 6. 決めていないこと

実装を始める前に決める。括弧内は、特に指定がなければ採用する案。

1. **オプションの形**：別のオプション `--lora_chain {merge,reuse}` にするか、`--finetune_mode lora_reuse` を足すか（別のオプションにする）。
2. **オプティマイザの状態**：タスクごとに作り直すか、AdamW のモーメントも引き継ぐか（作り直す。今の動作と同じ）。
3. **最初に実装する範囲**：vision と NLP を同時に実装するか、vision だけを先に実装するか（同時。学習ループの変更は両方とも同じ形のため）。
