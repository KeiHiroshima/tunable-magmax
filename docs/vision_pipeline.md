# visionタスク（CIFAR100 / ImageNetR）の処理の流れ

`scripts/vision/` 以下のシェルスクリプトから vision タスクを実行したときに、どのファイルのどの関数がどの順で呼ばれるかをまとめる。
対象は CLIP ViT（`ViT-B-16` など）を使ったクラス増分学習（`--split_strategy class`）で、データセットは `--dataset CIFAR100` / `ImageNetR`、タスク数は `--n_splits` 5/20/50。

NLP 側の流れは [nlp_pipeline.md](nlp_pipeline.md) を参照。

## 凡例

各ステップに、使われる実装の種類を次のタグで示す。

| タグ | 意味 |
|---|---|
| **[vision固有]** | vision のためだけの実装。NLP からは呼ばれない |
| **[共通]** | vision と NLP の両方から呼ばれる実装 |
| **[共通・vision分岐]** | 共通の実装だが、vision のときだけ通る分岐や引数がある |

行番号は執筆時点のもの。リンク先の関数名で検索すると確実に辿れる。

---

## 0. シェルスクリプトの呼び出し関係

```
finetune_all.sh ──(n_splits × seed でループ)──> finetune.sh ──> finetune_splitted.py
merge_comparison_all.sh ──(n_splits × seed)──> merge_comparison.sh ──(merge_fn [× similarity_metric])──> merge.sh ──> merge_for_targetdata.py
finetune_merge.sh ──> finetune_splitted.py の後に merge_for_targetdata.py（変数はスクリプト内に直書き）
```

| スクリプト | 役割 | 種別 |
|---|---|---|
| [finetune_all.sh](../scripts/vision/finetune_all.sh) | `n_splits_list`（既定 `5 20 50`）× `seeds`（`3 4 5`）の組合せごとに `finetune.sh` を呼ぶ。一部の実行が失敗しても止まらず、最後に失敗した組合せを一覧表示する | [vision固有] |
| [merge_comparison_all.sh](../scripts/vision/merge_comparison_all.sh) | 上と同じ n_splits × seed のループで `merge_comparison.sh` を呼ぶ | [vision固有] |
| [merge_comparison.sh](../scripts/vision/merge_comparison.sh) | `merge_fns` を1つずつ `merge.sh` に渡す。提案手法 `masked_magmax_with_targetdata` だけは `similarity_metrics`（既定 `labels ot_embedded`）ごとに呼ぶ | [vision固有] |
| [finetune.sh](../scripts/vision/finetune.sh) | `finetune_splitted.py` を `--split_strategy class --sequential-finetuning` 付きで起動する。既定値は `ViT-B-16`、`CIFAR100`、`epochs=10`、`n_splits=5`、`task_seq=A`、`seed=3` | [vision固有] |
| [merge.sh](../scripts/vision/merge.sh) | `merge_for_targetdata.py` を起動する。`num_target_data` は `n_splits` から決まる（5→1000、20→500、50→200）。提案手法のときだけ `--similarity_metric` を渡す。`--num_train_data_each_task 500` | [vision固有] |
| [finetune_merge.sh](../scripts/vision/finetune_merge.sh) | 学習とマージを続けて実行する。変数は環境変数で上書きできず、スクリプト内に直書きされている | [vision固有] |

ログの出力先は次の2つ。

- 標準出力：`outs/{model}/sequential_finetuning/class_incremental/{dir_name}/{dataset}-{n_splits}/taskseq_{p}/`
- 結果の JSON：`logs/…`（同じ構成、`--results_db` として渡す）

`dir_name` は `finetune.sh` では `DEFAULT_NAME`、`merge.sh` では `reproducing` が既定値。これが影響するのはログの置き場所だけで、チェックポイントの場所は変わらない。

## 1. Python 側の入口（学習・マージ共通）

1. [finetune_splitted.py](../finetune_splitted.py) と [merge_for_targetdata.py](../merge_for_targetdata.py) が `parse_arguments()` を呼ぶ。 **[共通]**
2. [args.py:67](../src/args.py#L67) の `parse_arguments` が引数を解析し、続けて [args.py:46](../src/args.py#L46) の `_resolve_training_defaults` を呼ぶ。 **[共通・vision分岐]**
   - vision では未指定の値に 学習率 1e-5、weight decay 0.1、`cosine` スケジュールを補う。
   - `--finetune_mode lora` が指定されていたらエラーにする（LoRA は StdCL/LSB のみ）。
   - 最後に `seed_everything` で乱数を固定する。
3. [backends/registry.py:18](../src/backends/registry.py#L18) の `resolve_backend` が、`CIFAR100`/`ImageNetR` を [vision_backend.py](../src/backends/vision_backend.py) に振り分ける。 **[共通]**
   - `finetune_splitted.py` からは `finetune(args)`、`merge_for_targetdata.py` からは `merge_and_evaluate(args)` が呼ばれる。

## 2. ファインチューニング（`finetune.sh`）

入口は [vision_backend.py:120](../src/backends/vision_backend.py#L120) の `finetune(args)`。`wandb.init` をしてから [`_run_sequential_finetuning`](../src/backends/vision_backend.py#L46) を呼ぶ。 **[vision固有]**

### 2.1 split（タスク）ごとのループ **[vision固有]**

`split_idx = 0 .. n_splits-1` の順に次を繰り返す。

1. **保存先と再開**：保存先は [paths.py:55](../src/paths.py#L55) の `finetuned_path`（`finetuned_{i}.pt`、**[共通]**）。すでにあればその split は飛ばす。次の split は直前の split のファイルを直接読むので、途中から再開しても学習の連鎖は保たれる。
   - 保存先ディレクトリは [`_ckpt_dir`](../src/backends/vision_backend.py#L36)（**[vision固有]**）が決め、パスの組み立ては [paths.py:40](../src/paths.py#L40) の `checkpoint_dir`（**[共通]**）が行う。
   ```
   $MAGMAX_BASE_DIR/checkpoints/{model}/sequential_finetuning/class_incremental/{dataset}-{n_splits}/ft-pattern_{p}-epochs-{e}-seed:{s}/
   ```
2. **開始時のモデル**
   - split 0：[modeling.py:11](../src/modeling.py#L11) の `ImageEncoder` で open_clip の事前学習済み CLIP を読み込む。テキストエンコーダも保持する（`keep_lang=True`）。
   - split 1 以降：直前の split の `finetuned_{i-1}.pt` を読み込む（`--sequential-finetuning`）。
   - `--load` が指定されていればそのファイルを読み込む。
3. **zero-shot チェックポイント**：split 0 で `checkpoints/{model}/zeroshot.pt` がまだなければ保存する（[config.py:8](../src/config.py#L8) `get_zeroshot_checkpoint` **[共通]**、[modeling.py:42](../src/modeling.py#L42) `ImageEncoder.save`）。
4. **データセット**：[datasets/registry.py:80](../src/datasets/registry.py#L80) の `get_dataset` で、[cifar100.py](../src/datasets/cifar100.py) または [imagenetr.py](../src/datasets/imagenetr.py) を CLIP の前処理付きで作る。
   - クラスの並び順（`default_class_order`）は `--taskseq_pattern` A/B/C で決まる（各ファイルの `class_order_dict`）。
5. **split 分のデータと分類ヘッド**：[cl_utils.py:11](../src/cl_utils.py#L11) の `get_dataset_and_classifier_for_split`。
   - [datasets/common.py:39](../src/datasets/common.py#L39) の `get_class_incremental_classes_and_subset_indices` で、この split が担当するクラスの train/test だけに絞る。ラベルは 0 始まりに振り直す。
   - [heads.py:56](../src/heads.py#L56) の `build_subset_classification_head` で、そのクラス名とテンプレート文から、CLIP テキストエンコーダを使ったゼロショット分類ヘッドを作る。
6. **学習の準備**：[trainer.py](../src/trainer.py)
   - [`setup_model_for_training`](../src/trainer.py#L10)：`ImageClassifier` を作り、ヘッドとテキストエンコーダを凍結して、見えている全 GPU に `DataParallel` で載せる。`--gpu_id` は学習には関係しない。 **[vision固有]**
   - [`build_loss_fn`](../src/trainer.py#L54)：CrossEntropy（`--ls` > 0 のときは label smoothing）。 **[vision固有]**
   - [`build_optimizer_and_scheduler`](../src/trainer.py#L46)：AdamW と、[`build_scheduler`](../src/trainer.py#L22) → [utils.py:23](../src/utils.py#L23) `cosine_lr` による学習率スケジュール（`--warmup_ratio` 0.1 の warmup の後に cosine 減衰）。 **[共通・vision分岐]**（`build_scheduler` は NLP と共有し、vision は cosine の分岐を通る）
   - [datasets/common.py:148](../src/datasets/common.py#L148) の `get_dataloader`。 **[vision固有]**
7. **学習**：`epochs` 回、[trainer.py:68](../src/trainer.py#L68) の `run_training_epoch` を呼ぶ。各ステップで学習率を更新し、forward と CrossEntropy、勾配クリップ 1.0、optimizer step を行う。epoch ごとに `wandb.log` する。 **[vision固有]**
8. **保存**：`DataParallel` から取り出した image encoder を `finetuned_{i}.pt` に保存する（`ImageEncoder.save`）。分類ヘッドは保存しない。 **[vision固有]**

## 3. マージと評価（`merge.sh`）

入口は [vision_backend.py:146](../src/backends/vision_backend.py#L146) の `merge_and_evaluate(args)`。 **[vision固有]**

### 3.1 準備 **[vision固有]**

1. 各 split について [task_vector.py:16](../src/merging/task_vector.py#L16) の `TaskVector(zeroshot.pt, finetuned_i.pt)` を作る（θ_i − θ_0）。 **[共通]**
2. [merging/registry.py:79](../src/merging/registry.py#L79) の `get_merge_spec(args.merge_fn)` でマージ手法を決める。 **[共通]**
3. 係数 `coeff = 0.5` で `wandb.init` し、[eval.py:146](../src/eval.py#L146) の `evaluate_merged_fts_on_target_data` を呼ぶ。

### 3.2 `evaluate_merged_fts_on_target_data` **[vision固有]**

1. `--num_train_data_each_task` が指定されているか確認する。類似度の計算に使う。
2. `get_dataset` でデータセットを作る。
3. [datasets/common.py:189](../src/datasets/common.py#L189) の `construct_train_subset_each_task`：各タスクの train から `num_train_data_each_task` 件（500）ずつ抽出する。提案手法で、ターゲット環境と各タスクとの類似度を測るために使う。
4. **ターゲット環境ごとのループ**：[target_env.py:96](../src/target_env.py#L96) の `load_target_envs` が `configs/{target_config}.json` を読む。 **[共通]**
   - 結果ファイルに `overall_accuracy` がすでにあればその環境は飛ばす（[utils.py:73](../src/utils.py#L73) `has_evaluation_result` **[共通]**）。
   - CIFAR100-50 の target 26（全タスク均等）も飛ばす。1タスクあたりのデータが少なすぎて meta データを取れないため。
5. **ターゲットデータの構築**：[datasets/common.py:273](../src/datasets/common.py#L273) の `construct_target_dataset`。 **[vision固有]**
   - [`_plan_samples_per_task`](../src/datasets/common.py#L237) が、どのタスクから何件取るかを決める。タスクの選択は [target_env.py:125](../src/target_env.py#L125) の `select_target_tasks`（**[共通]**）で、どのタスクに大きい比率が割り当たるかもシャッフルで決まる。
   - 各タスクの test から抽出し、そのうち **10% を meta データ**（選好ベクトルの推定用）、残りを評価用にする。
   - 件数が足りないタスクは、警告を出して取れる分だけ取る（[`_clamp_to_available`](../src/datasets/common.py#L219)）。
6. **マージ**：[eval.py:96](../src/eval.py#L96) の `_build_merged_encoder`。
   - **提案手法**（`spec.needs_target_data`）：[`_compute_similarity_weights`](../src/eval.py#L81) から [task_vectors.py:341](../src/merging/task_vectors.py#L341) の `merge_max_abs_masked_with_targetdata` を呼ぶ。 **[vision固有]**
     1. タスクごとに、meta データとの類似度を `--similarity_metric` に応じて計算する（[similarity.py](../src/merging/similarity.py)）。
        - `labels`：[`count_labels`](../src/merging/similarity.py#L510) で、meta データのうちそのタスクのクラスに属する件数を数える。
        - `cosine`/`mmd`/`ot`：[`compute_cosine_similarity`](../src/merging/similarity.py#L419)、[`compute_mmd_similarity`](../src/merging/similarity.py#L367)、[`compute_otdd_similarity`](../src/merging/similarity.py#L463) で、train の部分集合と meta データの分布を比べる。`_embedded` が付くと、そのタスクのモデルの特徴空間（`FeatureCost`）で比べる。OT には [otdd.py](../src/merging/otdd.py) を使う。
     2. 類似度を正規化して選好ベクトル `weights_each_task` にする。
     3. [task_vectors.py:83](../src/merging/task_vectors.py#L83) の `mask_and_merge_by_weights` でマージする。 **[共通]**（NLP の提案手法も同じ関数を使う。違うのは選好ベクトルの作り方だけ）
     4. `num_unaligned` と `num_params_all` を、評価より先に結果 JSON へ書き込む。
   - **ベースライン**：[merging/registry.py:88](../src/merging/registry.py#L88) の `apply_merge`。 **[共通]**
     - `magmax` は [task_vectors.py:53](../src/merging/task_vectors.py#L53) の `merge_max_abs`、`random_mix` は `merge_rnd_mix`、`ties` は [ties.py](../src/merging/ties.py)、`average` は和を取ってからタスク数で割る。
     - `finetune` はマージせず、最後のタスクのベクトルをそのまま返す。
     - vision ではこの処理が環境ごとに毎回走る（NLP は1回だけ計算して使い回す）。
   - zero-shot + 0.5 × マージ済みベクトルを [task_vector.py:93](../src/merging/task_vector.py#L93) の `TaskVector.apply_to` で作る。 **[共通]**
7. **評価**：[eval.py:22](../src/eval.py#L22) の `eval_given_dataset`。 **[vision固有]**
   - [heads.py:71](../src/heads.py#L71) の `get_classification_head` で、データセット全クラスのゼロショット分類ヘッドを作る。マージ後のモデルは、全クラスの中から予測する。
   - `ViT-L-14` のときだけ `DataParallel` を使う。それ以外は `--gpu_id` の GPU で評価する。
   - タスクごとに [utils.py:162](../src/utils.py#L162) の `do_eval` で top-1 精度を出す。
   - タスク平均（`average_accuracy`）と、サンプル単位の全体精度（`overall_accuracy`）を計算する。
8. **保存**：[eval.py:134](../src/eval.py#L134) の `_save_target_eval_results`。既存の JSON（提案手法が先に書いた `num_unaligned` など）と統合して保存する。 **[vision固有]**
   ```
   {results_db}/{merge_fn}/[{similarity_metric}/]{merge_fn}_lambda0.5_[{similarity_metric}_]target{id}_seed{s}.json
   ```
   中身は `overall_accuracy`、`average_accuracy`、`taskwise_accuracies`、`target_dataset_info` など。

## 4. 結果の集計（任意）

- [postprocessing/make_tables.ipynb](../postprocessing/make_tables.ipynb)：論文の表を作るノートブック（vision 専用）。
- [build_comparison_table.py](../postprocessing/build_comparison_table.py) `--backend vision`：[results.py](../postprocessing/results.py) の `vision_result_path` で上の JSON の場所を組み立て、表を CSV に出力する。 **[共通・vision分岐]**

```bash
uv run python postprocessing/build_comparison_table.py --backend vision \
    --results_db logs/ViT-B-16/sequential_finetuning/class_incremental/reproducing/CIFAR100-5/taskseq_A \
    --target_config target_data_config --seeds 3,4,5 \
    --competitors finetune,random_mix,average,ties,magmax,masked_magmax_with_targetdata-labels
```

---

## まとめ：実装の共有関係

| 処理 | 共通の実装 | vision 固有の実装 |
|---|---|---|
| 引数解析と既定値 | [args.py](../src/args.py)（`_resolve_training_defaults` で vision 用の既定値） | — |
| バックエンドの選択 | [backends/registry.py](../src/backends/registry.py) | — |
| データ | — | [datasets/](../src/datasets/)（`registry.py`、`cifar100.py`、`imagenetr.py`、`common.py`）、[cl_utils.py](../src/cl_utils.py) |
| モデル | — | [modeling.py](../src/modeling.py)（CLIP `ImageEncoder`）、[heads.py](../src/heads.py)（ゼロショットヘッド） |
| タスク列の学習ループ | — | [vision_backend.py](../src/backends/vision_backend.py) `_run_sequential_finetuning` |
| 1タスクの学習 | [trainer.py](../src/trainer.py) `build_scheduler`、[utils.py](../src/utils.py) `cosine_lr` | [trainer.py](../src/trainer.py) のそれ以外（`setup_model_for_training`、`run_training_epoch` など） |
| チェックポイントのパス | [paths.py](../src/paths.py)、[config.py](../src/config.py) | `_ckpt_dir`（vision_backend.py） |
| タスクベクトル | [task_vector.py](../src/merging/task_vector.py) | — |
| マージ手法 | [merging/registry.py](../src/merging/registry.py)、[task_vectors.py](../src/merging/task_vectors.py) `mask_and_merge_by_weights` などのマージ関数、[ties.py](../src/merging/ties.py) | [task_vectors.py](../src/merging/task_vectors.py) `merge_max_abs_masked_with_targetdata` |
| 選好ベクトル | — | [similarity.py](../src/merging/similarity.py)、[otdd.py](../src/merging/otdd.py) |
| ターゲット環境 | [target_env.py](../src/target_env.py) | [datasets/common.py](../src/datasets/common.py) `construct_target_dataset`、`construct_train_subset_each_task` |
| 評価 | [utils.py](../src/utils.py) `has_evaluation_result` | [eval.py](../src/eval.py)、[utils.py](../src/utils.py) `do_eval` |
| 集計 | [postprocessing/](../postprocessing/) | `make_tables.ipynb` |

## 補足（コードを読んでいて気づいた点）

- **ImageNetR では `--taskseq_pattern` が効いていない可能性がある**。[datasets/registry.py](../src/datasets/registry.py#L120) の `get_dataset` は `args` を CIFAR100 にしか渡さない（`args=args_ if dataset_name == "CIFAR100" else None`）。そのため [imagenetr.py](../src/datasets/imagenetr.py) は常にパターン A のクラス順を使い、B/C を指定してもエラーにならないまま A で実行される。
- 学習は `DataParallel` で、見えている全 GPU を使う。使う GPU を限定したいときは `CUDA_VISIBLE_DEVICES` を設定する。`--gpu_id` が影響するのは評価だけ。
