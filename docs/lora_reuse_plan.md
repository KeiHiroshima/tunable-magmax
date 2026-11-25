# 実装計画：単一の LoRA を全タスクで使い回す（`--lora_chain reuse`）

LoRA の逐次学習で、タスクごとに新しいアダプタを付けて前のアダプタを重みに足し込む今の方式に加えて、1 つの LoRA 行列（A, B）を全タスクで学習し続ける方式を追加する計画をまとめる。

- 状態：**実装・単体テスト・動作確認・本実験（§5）はすべて完了。結果は [vision_lora_reuse_report.md](vision_lora_reuse_report.md)。**
- ブランチ：`exp/lora-single-adapter`（`origin/scaling` の b09594c から作成）
- 関連文書：[vision_lora_reuse_report.md](vision_lora_reuse_report.md)（この方式の vision の結果）、[vision_lora_report.md](vision_lora_report.md)（今の方式の vision の結果）

## 1. 目的

今の LoRA の逐次学習は O-LoRA（式 9）に従っている。タスク t の学習を始める前に、それまでのアダプタをすべて事前学習済みの重み W₀ に足し込み、新しいアダプタを付けて学習する。このため、タスク t のタスクベクトル τ_t はアダプタ 0..t の累積和になり、ランクは最大 t·r まで大きくなる。

この計画では、もう 1 つの逐次学習の方式を追加する。

- 1 つの LoRA 行列（A, B）をタスク 2 以降でも使い回し、前のタスクの最終値から学習を続ける。
- タスクが切り替わるときに、W₀ に直前のタスクの LoRA を足し込むことはしない。モデルは常に W₀ + (α/r)·B A の形になる。

こうすると、どのタスクのタスクベクトルもランク r 以下の 1 つの LoRA になる。2 つの方式で学習したモデルに同じマージ手法をかけて、逐次学習のやり方が、マージ手法（MAGMAX、提案手法など）の結果をどう変えるかを比べられるようにする。

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
- `--finetune_mode` に新しい値（例：`lora_reuse`）を足す方法も考えられる。しかし `finetune_mode == "lora"` を確認している箇所が 10 か所ほどある（学習率の既定値、`--merge_space core` の確認、`_lora()` など）。値を増やすとこれらの漏れが起きやすいので、別のオプションにする。
- バックエンドでは `getattr(args, "lora_chain", "merge")` で読む。テストなどで `args` を手で組み立てていて、このフィールドがない場合は今の方式になる（vision の `_lora()` と同じ扱い）。

### 3.2 LoRA モジュール（[src/lora.py](../src/lora.py)）

- アダプタのファイル形式は変えない（`{"config", "modules"}` のまま）。どちらの方式で学習したかはファイルに書かず、保存先のディレクトリを分けることで区別する（3.4 節）。
- `load_adapter_into(model, adapter, config)` を追加する。保存したアダプタの A、B を、注入済みの `LoRALinear` にコピーする。`reuse` のときだけ呼ばれる。コピーする前に次を確認し、どれか 1 つでも合わなければ `ValueError` にする。
  - `adapter["config"]` が実行時の `asdict(config)`（r、α、dropout、target_modules）と一致すること。r が同じで α だけ違う場合、形の確認は通ってしまい、スケールが黙って変わるため。
  - アダプタのモジュール名の集合が、モデルの `LoRALinear` の集合と一致すること。
  - 各モジュールの A、B の形が一致すること。
- `adapter_task_vectors(adapters, sequential, chain="merge")` に `chain` を追加する。累積和をとるのは `sequential and chain == "merge"` のときだけにする。`reuse` の τ_t = (α/r)B_tA_t は、今ある独立学習の分岐（`adapters_to_vector([a])`）と同じ計算なので、新しい分岐は作らない。`chain` の既定値を `merge` にするので、今の呼び出し方はそのまま使える。

### 3.3 学習ループ

NLP（[src/nlp/finetune_nlp.py](../src/nlp/finetune_nlp.py)）と vision（[src/backends/vision_backend.py](../src/backends/vision_backend.py) の `_run_sequential_finetuning`）を同じように変更する。

`reuse` のときの各タスクの手順：

1. W₀ を読み込む。`merge_adapter_into` でそれまでのアダプタを足し込む処理は**しない**。
2. `inject_lora` で LoRA を付ける（NLP は `prepare_model` の後、vision は分類ヘッドを作った後。今と同じ位置）。
3. タスク t ≥ 1 では、`load_adapter_into` で `adapter_{t-1}.pt` の A、B を読み込む（3.2 節の確認を含む）。タスク 0 は今と同じ初期化（A は Kaiming、B = 0）。
4. 学習して、`adapter_{t}.pt` に保存する。

vision では、テストしやすくするために、各タスクの開始時のモデルを作る部分（手順 1 と 3）を `_run_sequential_finetuning` から小さな関数に切り出す。

- `_lora_start_encoder(args, lora_base_path, adapters_so_far)`：W₀ を読み込み、`merge` のときだけそれまでのアダプタを足し込む。
- `_resume_adapter(image_encoder, adapters_so_far, config, args)`：`reuse` で t ≥ 1 のとき、直前のアダプタを読み込む。`merge` のときは何もしない。

NLP の `finetune_task_sequence` は、今のテスト（`lora_run`）と同じようにループ全体をテストできるので、関数は切り出さない。

補足：

- **中断からの再開**：学習済みのタスクは今と同じく飛ばす。次のタスクは、直前に保存されたアダプタ `adapter_{t-1}.pt`（`adapters_so_far[-1]`）から始める。今のコードが守っている「直前のタスクから続ける」という性質（`finetune_nlp.py` の不変条件 I3）を、`reuse` でも守る。モジュールの docstring にある不変条件の説明に、`reuse` の場合の意味を書き足す。
- **オプティマイザの状態**：AdamW のモーメントとスケジューラは、今と同じくタスクごとに作り直す。引き継ぐのはパラメータの値だけ。

### 3.4 保存先とログの出力先を分ける

2 つの方式は、どちらも `adapter_{idx}.pt` という同じ名前で保存する。同じディレクトリを使うと、`reuse` の実行が今の方式のアダプタを学習済みとみなして学習を飛ばし、マージでも別の方式のアダプタを読んでしまう。アダプタのファイルには方式を書かない（3.2 節）ので、取り違えはディレクトリを分けることだけで防ぐ。

ログも、`tee -a` で追記しているファイルや固定のディレクトリがあり、分けないと 2 つの方式のログが混ざる。そこで、`-lora` を付けている箇所はすべて、`reuse` のときに `-lora-reuse` にする。`merge` のときの名前は今のまま変えない。

**チェックポイントと結果**

| 対象 | 今の方式 | `reuse` |
|---|---|---|
| NLP のチェックポイントと結果（`_ckpt_dir`、結果はその下） | `{dataset}-lora` | `{dataset}-lora-reuse` |
| vision のチェックポイント（`_ckpt_dir`） | `{dataset}-{n_splits}-lora` | `{dataset}-{n_splits}-lora-reuse` |
| vision の結果（`scripts/vision/merge.sh` の `dir_name`、`--results_db` になる） | `reproducing-lora` | `reproducing-lora-reuse` |

**ログ**

| 対象 | 今の方式 | `reuse` |
|---|---|---|
| vision の学習ログ（[scripts/vision/finetune.sh](../scripts/vision/finetune.sh) の `dir_name`、`outs/` と `logs/` の両方） | `DEFAULT_NAME-lora` | `DEFAULT_NAME-lora-reuse` |
| vision のマージのログ（`merge.sh` の `outs/`） | `reproducing-lora` | `reproducing-lora-reuse`（上の `dir_name` と共通） |
| NLP の学習とマージのログ（`scripts/nlp/{finetune,merge}.sh` の `out_dir`） | `outs/{model}/nlp_classification/{dataset}-lora` | `outs/{model}/nlp_classification/{dataset}-lora-reuse` |
| vision の一括実行（[scripts/vision/run_lora_sweep.sh](../scripts/vision/run_lora_sweep.sh) の `log_dir`） | `logs/vision_lora_sweep` | `logs/vision_lora_reuse_sweep` |
| NLP の一括実行（`scripts/nlp/run_t5-base_lora_lsb_single_gpu.sh` のログのファイル名） | `{model}_{dataset}-lora_order{n}.log` | `{model}_{dataset}-lora-reuse_order{n}.log` |
| wandb（vision の `finetune` と `merge_and_evaluate`） | 今のまま | run の名前の末尾に `-lora-reuse` を付け、タグに `lora-reuse` を足す |

- 各スクリプトでは、`finetune_mode` と `lora_chain` から接尾辞（`lora` または `lora-reuse`）を 1 回だけ決め、`-lora` を付けていた箇所はすべてその変数を使う。こうして、1 か所だけ付け忘れることを防ぐ。
- 一括実行を 1 度に 1 つしか動かせない制限（`run_lora_sweep.sh` の注意書き）は、同じ方式どうしの話である。`merge` と `reuse` はチェックポイントもログも別なので、同時に動かしてよい。注意書きをそう直す。

**postprocessing**

NLP の結果はチェックポイントの横に保存され、postprocessing はそのパスを手で組み立てている（`postprocessing/` は `src/` を import しない）。

- [postprocessing/results.py](../postprocessing/results.py) の `nlp_result_path` に `lora_chain="merge"` を追加し、`reuse` のときは `{dataset}-lora-reuse` を使う。
- [build_comparison_table.py](../postprocessing/build_comparison_table.py) に `--lora_chain` を追加し、`nlp_result_path` に渡す。
- [aggregate_nlp_logs.py](../postprocessing/aggregate_nlp_logs.py) はディレクトリの名前から scope を見つけるので、`{dataset}-lora-reuse` は別の表として自動で出る。コードは変えず、docstring のパスの説明（`{dataset}[-lora]`）だけを直す。

### 3.5 マージ

- マージ側の変更は、vision と NLP の `merge_and_evaluate`（NLP は `_load_task_vectors`）が `chain` を指定して `adapter_task_vectors` を呼ぶことだけ。

### 3.6 スクリプトと文書

- `scripts/{vision,nlp}/{finetune,merge}.sh`、`scripts/vision/run_lora_sweep.sh`、`scripts/nlp/run_t5-base_lora_lsb_single_gpu.sh` に、環境変数 `lora_chain`（既定値 `merge`）を追加し、`--lora_chain` として渡す。ログの出力先は 3.4 節の通りに分ける。
- README のオプションの表に `--lora_chain` を追加する。
- この文書を、実装した内容に合わせて更新する（2 つの方式の違いの説明として残す）。

## 4. テスト内容

小さなモデルで動かし、GPU は使わない。NLP は [test/test_nlp_lora.py](../test/test_nlp_lora.py) の小さな T5（`_tiny_t5`）、vision は [test/test_vision_lora.py](../test/test_vision_lora.py) の小さなエンコーダ（`_encoder`）を使う。出力を比べるときは、dropout が効かないよう、どちらのモデルも eval モードにする（NLP の `_logits` は eval モードで計算している）。

### 4.1 NLP（`test/test_nlp_lora.py`）

今の `lora_run` に引数 `chain` を足し、`args` に `lora_chain` を入れる。`train_task` の代わりの関数は、今の記録（開始時の出力）に加えて、開始時のベースの重みと A、B の値も記録する。

| テスト | 手順 | 確認すること |
|---|---|---|
| ベースの重みに足し込まない | `reuse` で 3 タスク実行 | すべてのタスクで、開始時の各 `LoRALinear.base.weight` が W₀ と一致する |
| 直前のアダプタから始まる | 同上 | タスク 0 の開始時は B = 0。タスク t ≥ 1 の開始時の A、B が、保存した `adapter_{t-1}.pt` の A、B と一致する |
| 開始時のモデルは直前のタスクのモデル | 同上 | タスク t ≥ 1 の開始時の出力が、`_apply_task_vector(τ_{t-1}, W₀)` で作ったモデルの出力と一致する |
| タスクベクトルはアダプタ 1 つ分 | 同上の後、`lora_chain="reuse"` で `_load_task_vectors` を呼ぶ | τ_t が `adapter_deltas(adapter_t)` と一致する。各キーの行列のランクが r 以下 |
| 中断からの再開 | `reuse` で 2 タスク実行し、記録を消してから 3 タスクで再実行 | 学習するのはタスク 2 だけ。タスク 2 の開始時の A、B が `adapter_1.pt` と一致する |
| 保存先の分離 | `_ckpt_dir` を `merge`、`reuse` で呼ぶ | 末尾がそれぞれ `{dataset}-lora/...`、`{dataset}-lora-reuse/...` になる |

### 4.2 vision（`test/test_vision_lora.py`）

vision の学習ループ全体はデータセットと wandb に依存するので、3.3 節で切り出す 2 つの関数をテストする。

| テスト | 確認すること |
|---|---|
| 開始時のモデル（`_lora_start_encoder`） | `reuse` では、アダプタがあっても W₀ のまま返る。`merge` では今と同じく足し込まれる |
| 直前のアダプタの読み込み（`_resume_adapter`） | `reuse` で t ≥ 1 なら A、B が直前のアダプタと一致する。t = 0 と `merge` では何もしない（B = 0 のまま） |
| タスクベクトルでモデルが再現できる | 2 タスクを `reuse` の手順で作り（今の `test_merged_adapters_and_task_vectors_rebuild_the_lora_model` と同じ形）、W₀ + τ_最後 の出力が、学習を終えたときの LoRA モデルの出力と一致する |
| 保存先の分離 | `_ckpt_dir` の末尾が、`merge` では `CIFAR100-5-lora/...`、`reuse` では `CIFAR100-5-lora-reuse/...` になる |

### 4.3 共通

| テスト | 確認すること |
|---|---|
| 設定の整合チェック（`load_adapter_into`、[src/lora.py](../src/lora.py) の単体テスト） | α だけを変えたアダプタ、モジュールが 1 つ足りないアダプタ、A の形が違うアダプタを読み込むと、それぞれ `ValueError` になる。一致するアダプタでは A、B がコピーされる |
| `adapter_task_vectors` | `sequential=True, chain="reuse"` の結果が、各アダプタの差分そのもの。`chain` を指定しない場合は、今と同じく累積和 |
| オプションの確認（[test/characterization/test_cli_surface.py](../test/characterization/test_cli_surface.py)） | `--lora_chain` の既定値が `merge`。`reuse` を `--finetune_mode full` で指定するとエラー。`--sequential-finetuning` なしで指定するとエラー |
| postprocessing（`test/test_postprocessing_results.py`） | `nlp_result_path(..., finetune_mode="lora", lora_chain="reuse")` のパスに `{dataset}-lora-reuse` が入る。`lora_chain` を指定しない場合は今と同じパス |
| 既定の動作 | 今のテスト（`lora_chain` のない `Namespace` を使うものも含む）が変更なしで通る |

## 5. 動作確認と実験

1. 単体テストがすべて通ることを確認する。→ **完了**
2. vision の小さな予備実験：CIFAR-100-5、seed 3。設定は今の LoRA の実験と同じ（q・v、r = 8、α = 32、lr 1e-3、10 epoch）。学習が進むこと、マージまで通ること、チェックポイントとログが 3.4 節の `-lora-reuse` の場所に出ることを確かめる。→ **完了**。時間を短くするため、10 epoch ではなく 1 epoch で実行した
3. 本実験：CIFAR-100 の T = 5 / 20 / 50 × seed 3 / 4 / 5。今の方式の結果（[vision_lora_report.md](vision_lora_report.md)）と、`build_comparison_table.py` で同じ形の表にして比べる。→ **完了**。比べる基準の全パラメータ学習と今の方式（merge）も、同じコードで学習し直した。手法は fine-tune、MAGMAX、Tunable MAGMAX（OT、Label）に絞った。結果は [vision_lora_reuse_report.md](vision_lora_reuse_report.md)

## 6. 実装前に決めたこと

実装では、3 つとも括弧内の案を採用した。

1. **オプションの形**：別のオプション `--lora_chain {merge,reuse}` にするか、`--finetune_mode lora_reuse` を足すか（別のオプションにする）。
2. **オプティマイザの状態**：タスクごとに作り直すか、AdamW のモーメントも引き継ぐか（作り直す。今の動作と同じ）。
3. **最初に実装する範囲**：vision と NLP を同時に実装するか、vision だけを先に実装するか（同時。学習ループの変更は両方とも同じ形のため）。
