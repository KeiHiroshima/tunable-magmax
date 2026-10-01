# NLPタスク（StdCL / LSB）の処理の流れ

`scripts/nlp/` 以下のシェルスクリプトから NLP タスクを実行したときに、どのファイルのどの関数がどの順で呼ばれるかをまとめる。
対象は O-LoRA（Wang et al., Findings of EMNLP 2023）の T5 設定に基づく `--dataset StdCL`（4タスク、Order 1–3）と `--dataset LSB`（15タスク、Order 4–6）。
CITB（`CITB19`/`CITB38`）は別のバックエンド（[nlp_seq2seq_backend.py](../src/backends/nlp_seq2seq_backend.py)）で動くため、ここでは扱わない。

vision 側の流れは [vision_pipeline.md](vision_pipeline.md) を参照。

## 凡例

各ステップに、使われる実装の種類を次のタグで示す。

| タグ | 意味 |
|---|---|
| **[NLP固有]** | StdCL/LSB（T5）のためだけの実装。vision からは呼ばれない |
| **[共通]** | vision と NLP の両方から呼ばれる実装 |
| **[共通・NLP分岐]** | 共通の実装だが、NLP のときだけ通る分岐や引数がある |

行番号は執筆時点のもの。リンク先の関数名で検索すると確実に辿れる。

---

## 0. シェルスクリプトの呼び出し関係

```
finetune_all.sh ──(Order × seed でループ)──> finetune.sh ──> finetune_splitted.py
merge_comparison_all.sh ──(Order × seed)──> merge_comparison.sh ──(merge_fn ごと)──> merge.sh ──> merge_for_targetdata.py
finetune_merge.sh ──> finetune.sh の後に merge.sh
```

| スクリプト | 役割 | 種別 |
|---|---|---|
| [finetune_all.sh](../scripts/nlp/finetune_all.sh) | `dataset` に応じて回す Order を決め（StdCL は `1 2 3`、LSB は `4 5 6`）、seed `3 4 5` との組合せごとに `finetune.sh` を呼ぶ。一部の実行が失敗しても止まらず、最後に失敗した実行を一覧表示する | [NLP固有] |
| [merge_comparison_all.sh](../scripts/nlp/merge_comparison_all.sh) | 上と同じ Order × seed のループで `merge_comparison.sh` を呼ぶ | [NLP固有] |
| [merge_comparison.sh](../scripts/nlp/merge_comparison.sh) | `merge_fns`（finetune, random_mix, average, ties, magmax, masked_magmax_with_targetdata）を1つずつ `merge.sh` に渡す。vision と違い、similarity metric のループはない | [NLP固有] |
| [finetune.sh](../scripts/nlp/finetune.sh) | 変数の既定値を決めて `finetune_splitted.py` を起動する。`model=t5-large` なら `finetune_mode=lora`、それ以外は `full`。学習の既定値は `epochs=1`、`batch_size=8`、`grad_accum_steps=8`。`lr` は環境変数で指定したときだけ `--lr` として渡す | [NLP固有] |
| [merge.sh](../scripts/nlp/merge.sh) | `merge_for_targetdata.py` を起動する。`finetune_mode` は `finetune.sh` と同じ規則で決まり、`--eval_full_testsets` が既定で有効 | [NLP固有] |
| [finetune_merge.sh](../scripts/nlp/finetune_merge.sh) | `finetune.sh` の後に `merge.sh` を実行する | [NLP固有] |

`finetune.sh` と `merge.sh` は、ほかに次の処理をする。

- `TORCH_DISABLE_NATIVE_JIT=1` を export する。torch 2.14 は T5 の `generate()` で使う一部の演算を Triton カーネルに回すが、そのビルドに必要な Python ヘッダがこのマシンにはない。この変数で、それらの演算を通常の CUDA カーネルのまま実行させる。
- 標準出力を `outs/{model}/nlp_classification/{dataset}-{finetune_mode}/taskseq_{order}/` に追記する。

## 1. Python 側の入口（学習・マージ共通）

1. [finetune_splitted.py](../finetune_splitted.py) と [merge_for_targetdata.py](../merge_for_targetdata.py) が `parse_arguments()` を呼ぶ。 **[共通]**
2. [args.py:67](../src/args.py#L67) の `parse_arguments` が引数を解析し、続けて [args.py:46](../src/args.py#L46) の `_resolve_training_defaults` を呼ぶ。 **[共通・NLP分岐]**
   - `--dataset` が StdCL/LSB のときに限り、未指定の値を次のように補う。
     - 学習率：full なら 1e-4、lora なら 1e-3
     - weight decay：0
     - 学習率スケジュール：`constant`
   - Order がそのデータセットのもの（StdCL なら 1–3、LSB なら 4–6）でなければエラーにする。
   - vision/CITB のときは従来どおり 1e-5 / 0.1 / cosine を補う。
3. [backends/registry.py:18](../src/backends/registry.py#L18) の `resolve_backend` が、`StdCL`/`LSB` を [nlp_classification_backend.py](../src/backends/nlp_classification_backend.py) に振り分ける。 **[共通]**
   - `finetune_splitted.py` からは `finetune(args)`、`merge_for_targetdata.py` からは `merge_and_evaluate(args)` が呼ばれる。
   - `finetune_splitted.py` は学習の前に `wandb.init` も行う。

## 2. ファインチューニング（`finetune.sh`）

入口は [nlp_classification_backend.py:72](../src/backends/nlp_classification_backend.py#L72) の `finetune(args)`。 **[NLP固有]**

### 2.1 タスク列の構築 **[NLP固有]**

[`_build_tasks`](../src/backends/nlp_classification_backend.py#L56) から [long_sequence_benchmark.py:192](../src/nlp/long_sequence_benchmark.py#L192) の `build_task_sequence` を呼ぶ。

- [`task_order`](../src/nlp/long_sequence_benchmark.py#L98)：Order 番号からタスク名の並びを取り出す（論文 Table 7。定義は同じファイルの `TASK_ORDERS`）。
- [`load_task`](../src/nlp/long_sequence_benchmark.py#L178) → [`load_examples`](../src/nlp/long_sequence_benchmark.py#L165)：`$MAGMAX_OLORA_DATA_DIR/{カテゴリ}/{タスク}/{train,test,labels}.json` を読み込む。既定のデータ置き場は `~/O-LoRA/CL_Benchmark`。
- [`build_prompt`](../src/nlp/long_sequence_benchmark.py#L111)：O-LoRA と同じ形式の入力文を作る。
  ```
  Task:{カテゴリ}\nDataset:{タスク}\n{指示文}Option: {ラベル1, ラベル2, ...} \n{本文}\nAnswer:
  ```
  COPA だけは指示文と Option 行がない。
- [`Seq2SeqCollator`](../src/nlp/long_sequence_benchmark.py#L127)：バッチ内で最長の系列に合わせてパディングしながらトークン化する（入力は最大512、出力は最大50トークン）。正解ラベル文字列 `targets` もバッチに含めて渡す。
- 結果は [task_spec.py](../src/task_spec.py) の `TaskSpec` に入る。 **[共通]**（NLP の2つのバックエンドで共有している）

### 2.2 タスクごとの学習ループ

[finetune_nlp.py:58](../src/nlp/finetune_nlp.py#L58) の `finetune_task_sequence`。 **[NLP固有]**（StdCL/LSB と CITB で共有しているが、vision は使わない）

1. [`_ensure_zeroshot_checkpoint`](../src/nlp/finetune_nlp.py#L43)：`checkpoints/{model}/zeroshot.pt` がまだなければ作る。
   - モデルは [modeling_nlp.py:9](../src/nlp/modeling_nlp.py#L9) の `build_t5` で作る（fp32、dropout 0.1）。 **[NLP固有]**
   - 保存先のパスは [config.py:8](../src/config.py#L8) の `get_zeroshot_checkpoint` が決める。 **[共通]**
2. 各タスクについて保存先を決める。
   - full：[paths.py:55](../src/paths.py#L55) の `finetuned_path`（`finetuned_{i}.pt`）。 **[共通]**
   - lora：[paths.py:60](../src/paths.py#L60) の `adapter_path`（`adapter_{i}.pt`）。 **[NLP固有]**
   - 保存先がすでにあれば、そのタスクは学習せずに飛ばす。中断しても同じコマンドで続きから再開できる。
3. **開始時のモデル**
   - full：直前のタスクのチェックポイントを読み込む（[utils.py](../src/utils.py) の `torch_load`。 **[共通]**）。
   - lora：`zeroshot.pt` を読み込み、それまでの adapter をすべて、学習した順に重みへ統合する（[lora.py:100](../src/nlp/lora.py#L100) `merge_adapter_into`、O-LoRA 式9）。 **[NLP固有]**
4. `prepare_model`（[modeling_nlp.py:23](../src/nlp/modeling_nlp.py#L23) `freeze_shared_embeddings`）：T5 の共有埋め込みを凍結する。 **[NLP固有]**
5. lora の場合のみ、[lora.py:64](../src/nlp/lora.py#L64) の `inject_lora` で全 attention の q と v を `LoRALinear` に差し替え、A と B 以外のパラメータをすべて凍結する（r=8、alpha=32、dropout 0.1）。 **[NLP固有]**

### 2.3 学習

[trainer_nlp.py:69](../src/nlp/trainer_nlp.py#L69) の `train_seq2seq_task` から [`_run_epochs`](../src/nlp/trainer_nlp.py#L25) を呼ぶ。 **[NLP固有]**

- AdamW で、`requires_grad` が立ったパラメータだけを更新する。
- `grad_accum_steps` 個のマイクロバッチごとに1回 optimizer を step する。既定値の 8 × 8 で実効バッチ 64 になる。
- 学習率は [trainer.py:22](../src/trainer.py#L22) の `build_scheduler` から [utils.py:41](../src/utils.py#L41) の `constant_lr` で決まる。 **[共通・NLP分岐]**（`--lr_schedule constant` は NLP の既定値で、vision は cosine の分岐を通る）

### 2.4 保存

- full：モデル全体を `finetuned_{i}.pt` に保存する（`torch_save`）。 **[共通]**
- lora：[lora.py:78](../src/nlp/lora.py#L78) の `extract_adapter` で取り出した A と B だけを `adapter_{i}.pt` に保存する（t5-large で約9MB）。 **[NLP固有]**

保存先ディレクトリは [`_ckpt_dir`](../src/backends/nlp_classification_backend.py#L48)（**[NLP固有]**）が決める。実際のパスの組み立ては [paths.py:40](../src/paths.py#L40) の `checkpoint_dir`（**[共通]**）が行う。

```
$MAGMAX_BASE_DIR/checkpoints/{model}/sequential_finetuning/nlp_classification/{dataset}[-lora]/ft-pattern_{order}-epochs-1-seed:{s}/
```

## 3. マージと評価（`merge.sh`）

入口は [nlp_classification_backend.py:161](../src/backends/nlp_classification_backend.py#L161) の `merge_and_evaluate(args)`。 **[NLP固有]**
構造は vision の [eval.py:146](../src/eval.py#L146) `evaluate_merged_fts_on_target_data` と対応しているが、関数は別実装になっている。

### 3.1 準備

1. `_build_tasks` でタスク列を作り直す（評価用データを使うため）。 **[NLP固有]**
2. **タスクベクトルの読み込み**：[`_load_task_vectors`](../src/backends/nlp_classification_backend.py#L87)。 **[NLP固有]**
   - full：[task_vector.py:16](../src/merging/task_vector.py#L16) の `TaskVector(zeroshot, finetuned_i)` で、各タスク後のモデルと zero-shot モデルとの差分（θ_i − θ_0）を取る。 **[共通]**
   - lora：[lora.py:108](../src/nlp/lora.py#L108) の `adapters_to_vector` で adapter 0..i の ΔW を累積和し、`TaskVector(vector=…)` にする。キーは q と v の重みだけ。 **[NLP固有]**
3. [`_tokenizer`](../src/backends/nlp_classification_backend.py#L66) で、評価時の生成結果をデコードするための tokenizer を読み込む。 **[NLP固有]**
4. [merging/registry.py:79](../src/merging/registry.py#L79) の `get_merge_spec(args.merge_fn)` でマージ手法を決める。 **[共通]**

### 3.2 ターゲット環境ごとのループ

1. 環境の一覧：[target_env.py:96](../src/target_env.py#L96) の `load_target_envs` が `configs/target_data_config_lsb.json` を読む。 **[共通]**
2. 結果ファイル `{merge_fn}/target{id}_seed{s}.json` に `overall_accuracy` がすでにあれば、その環境は飛ばす（[utils.py:73](../src/utils.py#L73) `has_evaluation_result`）。 **[共通]**
3. 混ぜるタスクの選択：[target_env.py:125](../src/target_env.py#L125) の `select_target_tasks`。 **[共通]**
4. **マージの分岐**
   - **提案手法**（`masked_magmax_with_targetdata`、`spec.needs_target_data`）：環境ごとに次の2段階でマージする。
     1. [target_data.py:28](../src/nlp/target_data.py#L28) の `build_target_weights` で、環境の混合比をそのまま選好ベクトルにする。NLP ではタスクがそれぞれ別のデータセットなので、どのタスクから何割引いたかは構築時点でわかっており、類似度で推定する必要がない。 **[NLP固有]**
     2. [task_vectors.py:83](../src/merging/task_vectors.py#L83) の `mask_and_merge_by_weights` で、選好ベクトルに従ってタスクベクトルを要素ごとに選んで統合する。 **[共通]**（vision の提案手法も最後にこの関数を呼ぶ）
   - **ベースライン**：[merging/registry.py:88](../src/merging/registry.py#L88) の `apply_merge` で、最初に1回だけマージし、全環境で使い回す。 **[共通]**
     - 実体は [task_vectors.py](../src/merging/task_vectors.py) の `merge_max_abs`（magmax）と `merge_rnd_mix`、および [ties.py](../src/merging/ties.py)。
     - `finetune` はマージせず、最後のタスクのベクトルをそのまま返す。
5. **モデルの再構成**：[`_apply_task_vector`](../src/backends/nlp_classification_backend.py#L103) で zero-shot + `scaling_coef`(0.5) × マージ済みベクトルを作る。 **[NLP固有]**
   - vision は `TaskVector.apply_to`（**[共通]**）を使う。NLP がそれを使わないのは、LoRA のベクトルが q と v のキーしか持たず、`apply_to` だと足りないキーごとに警告が大量に出るため。計算内容は同じ。
6. **評価**：[`_evaluate_on_target_env`](../src/backends/nlp_classification_backend.py#L123)。 **[NLP固有]**
   - [target_data.py:39](../src/nlp/target_data.py#L39) の `sample_target_eval_subset` で、各タスクの test から混合比に応じた件数を抽出する。
   - [trainer_nlp.py:81](../src/nlp/trainer_nlp.py#L81) の `generation_correct_and_total` で貪欲生成し、出力を正規化してから正解ラベルとの完全一致で数える（[`normalize_answer`](../src/nlp/long_sequence_benchmark.py#L120)）。
7. `overall_accuracy`、`taskwise_accuracies` などを JSON に保存する。提案手法の場合は `weights_each_task`、`num_unaligned`、`num_params_all` も加わる。 **[NLP固有]**（vision とは JSON の形と保存先が違う）

### 3.3 全テストセットでの評価

`--eval_full_testsets` を付けたとき（`merge.sh` の既定で有効）だけ、[`_evaluate_on_full_testsets`](../src/backends/nlp_classification_backend.py#L141) が動く。 **[NLP固有]**

- 対象はベースラインだけ。提案手法は環境ごとにモデルが変わるので、1つのモデルとしての Average Accuracy を持たない。
- 全タスクの test 全件を評価し、論文の Average Accuracy（`average_accuracy`）を `{merge_fn}/full_testsets_seed{s}.json` に保存する。

### 3.4 出力先

結果はチェックポイントディレクトリの中に書かれる。`--results_db` は使わない。

```
.../nlp_classification/{dataset}[-lora]/ft-pattern_{order}-epochs-1-seed:{s}/{merge_fn}/target{id}_seed{s}.json
.../nlp_classification/{dataset}[-lora]/ft-pattern_{order}-epochs-1-seed:{s}/{merge_fn}/full_testsets_seed{s}.json
```

## 4. 結果の集計（任意）

[build_comparison_table.py](../postprocessing/build_comparison_table.py) は、[results.py](../postprocessing/results.py) の `nlp_result_path` で上の JSON の場所を組み立て、手法 × 環境の表（seed 間の平均 ± 標準偏差）を CSV に出力する。 **[共通・NLP分岐]**（`--backend nlp`。`--finetune_mode lora` を付けると `{dataset}-lora` 側を読む）

```bash
uv run python postprocessing/build_comparison_table.py --backend nlp \
    --model t5-large --finetune_mode lora --dataset LSB --taskseq_pattern 4 \
    --target_config target_data_config_lsb --seeds 3,4,5
```

---

## まとめ：実装の共有関係

| 処理 | 共通の実装 | NLP 固有の実装 |
|---|---|---|
| 引数解析と既定値 | [args.py](../src/args.py)（`_resolve_training_defaults` の中で NLP 分岐） | — |
| バックエンドの選択 | [backends/registry.py](../src/backends/registry.py) | — |
| データとプロンプト | [task_spec.py](../src/task_spec.py) | [long_sequence_benchmark.py](../src/nlp/long_sequence_benchmark.py) |
| モデル | — | [modeling_nlp.py](../src/nlp/modeling_nlp.py)、[lora.py](../src/nlp/lora.py) |
| タスク列の学習ループ | — | [finetune_nlp.py](../src/nlp/finetune_nlp.py)（CITB と共有） |
| 1タスクの学習 | [trainer.py](../src/trainer.py) `build_scheduler`、[utils.py](../src/utils.py) `constant_lr`/`cosine_lr` | [trainer_nlp.py](../src/nlp/trainer_nlp.py) `_run_epochs` |
| チェックポイントのパス | [paths.py](../src/paths.py)、[config.py](../src/config.py) | `adapter_path`（paths.py 内）、`_ckpt_dir` |
| タスクベクトル | [task_vector.py](../src/merging/task_vector.py) | `adapters_to_vector`（lora.py） |
| マージ手法 | [merging/registry.py](../src/merging/registry.py)、[task_vectors.py](../src/merging/task_vectors.py)、[ties.py](../src/merging/ties.py) | — |
| 選好ベクトル | — | [target_data.py](../src/nlp/target_data.py) `build_target_weights`（vision は [similarity.py](../src/merging/similarity.py) で推定） |
| ターゲット環境 | [target_env.py](../src/target_env.py) | [target_data.py](../src/nlp/target_data.py) `sample_target_eval_subset` |
| 評価 | [utils.py](../src/utils.py) `has_evaluation_result` | [trainer_nlp.py](../src/nlp/trainer_nlp.py) `generation_correct_and_total`、[nlp_classification_backend.py](../src/backends/nlp_classification_backend.py) |
| 集計 | [postprocessing/](../postprocessing/) | — |
