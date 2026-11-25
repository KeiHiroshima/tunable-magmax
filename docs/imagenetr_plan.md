# 実装計画：ImageNet-R で CIFAR-100 と同じ実験を行う

CIFAR-100 で行った vision の実験（[vision_lora_reuse_report.md](vision_lora_reuse_report.md)、[vision_tunable_ties_report.md](vision_tunable_ties_report.md)）を、ImageNet-R（`--dataset ImageNetR`）でも行う計画をまとめる。

- 状態：**実装・単体テスト・ローカルでの動作確認（ダミーのデータとチェックポイント）は完了。サーバーでの整備（§2）、試し実行と本実験（§4）は未実施。**
- ブランチ：`exp/imagenet-r`（`origin/scaling` の bed4b2e から作成）
- 関連文書：[vision_pipeline.md](vision_pipeline.md)（vision の処理の流れ）、[logs/README.md](../logs/README.md)（結果と実行の記録の置き場所）

## 1. 目的と範囲

CIFAR-100 と同じ比較を ImageNet-R-T（T ∈ {5, 20, 50}）で行う。

| 学習の方式 | チェックポイント | マージ手法 |
|---|---|---|
| **full**（全パラメータ学習） | **論文のときのものを流用する（学習しない）** | fine-tune、MAGMAX、Tunable MAGMAX（Label、OT）、TIES、Tunable TIES（Label、OT） |
| **lora-reuse**（1 つの LoRA を全タスクで学習し続ける） | 新しく学習する | fine-tune、MAGMAX、Tunable MAGMAX（Label、OT）、TIES、Tunable TIES（Label、OT）、**random_mix、average** |

- full の random_mix と average は、論文の結果（流用するチェックポイントからマージしたもの）を使う。今回は実行しない。
- 次のものは今回の範囲に入れない。
  - lora-merge（タスクごとの LoRA。O-LoRA の方式）
  - Core Space（`--merge_space core`）
- 共通の設定は CIFAR-100 と同じにする。
  - seed は 3 / 4 / 5、タスク順序は A（ImageNet-R には A しかない）、10 epoch、λ = 0.5
  - ターゲット環境は `configs/target_data_config.json`
  - ターゲットデータ数は T=5 / 20 / 50 でそれぞれ 1000 / 500 / 200。タスクごとの train は 500 件
- ImageNet-R の train/test の分割は、論文のときのものを流用する。

### 1.1 マージ元のチェックポイントをそろえる

1 つの結果ディレクトリ（`full/`、`lora-reuse/`）の中の手法は、すべて同じチェックポイントからマージする（[logs/README.md](../logs/README.md) のルール 3）。そのために次の 3 つを守る。

1. すべての実験を、同じサーバーの同じ `MAGMAX_BASE_DIR` で行う。
2. 学習するのは、最初の実験の lora-reuse だけにする。それ以外のジョブは「マージだけ」（§3.2）で動かす。チェックポイントが欠けていれば、学習し直さずにジョブを失敗させる。
3. 起動するたびに、`zeroshot.pt` の sha256 と、使うチェックポイントの一覧を実行の記録に残す（§3.3）。

## 2. 実験を行うマシンで、実験の前に整備すること

このマシン（keisubuntu）からはサーバーが見えないので、次の作業はサーバー上で行う。**全部が終わってから実験を起動する。** §3.1 の準備用のスクリプトで、1〜3 が整っているかを確かめられる。

1. **full のチェックポイントを、今のコードが読むパスに置く。**
   - 置き場所：`$MAGMAX_BASE_DIR/checkpoints/ViT-B-16/sequential_finetuning/class_incremental/ImageNetR-{T}/ft-pattern_A-epochs-10-seed:{s}/finetuned_{0..T-1}.pt`
   - 対象：T ∈ {5, 20, 50} × seed ∈ {3, 4, 5} の 9 通り
   - 論文のときのパスが違う場合は、シンボリックリンクでこのパスに合わせる。中身はコピーも変更もしない。
2. **`zeroshot.pt` を、full のチェックポイントを学習したときと同じものにする。**
   - 置き場所：`$MAGMAX_BASE_DIR/checkpoints/ViT-B-16/zeroshot.pt`。open_clip のキャッシュ `checkpoints/ViT-B-16/cachedir/open_clip` も同じ場所に置く。
   - タスクベクトルは θ − θ₀ なので、θ₀ が違うとマージ結果が変わる。
   - `zeroshot_unpacked.pt`（LoRA 用）は、なければ `zeroshot.pt` から自動で作られる。すでにある場合は、同じ `zeroshot.pt` から作ったものであることを確かめる。
3. **ImageNet-R の分割を置く。**
   - 置き場所：`$MAGMAX_DATA_DIR/imagenet-r/{train,test}/<クラス名>/`（論文のときの分割）
   - 今回の変更で、`train/`・`test/` がなければ分割し直さずにエラーにする（§3.1）。
4. **論文の full の random_mix・average の結果 JSON を、今の構成の場所に置く。**
   - 置き場所：`logs/results/vision/ViT-B-16/sequential_finetuning/class_incremental/full/ImageNetR-{T}/taskseq_A/{random_mix,average}/`
   - ファイル名は今のコードが書く名前（`{merge_fn}_lambda0.5_target{id}_seed{s}.json`）にそろえる。中身に `overall_accuracy`、`average_accuracy`、`taskwise_accuracies` があることを確かめる（集計スクリプトがこのキーを読む）。
   - `logs/README.md` の索引に、どの実行の結果をどこから持ってきたかを書く。
5. **論文の、それ以外の手法の結果を `logs/results/.../full/ImageNetR-*/` に置かない。**
   - 対象：finetune、magmax、masked_magmax_with_targetdata、ties
   - 結果 JSON があると、その環境は「評価済み」として飛ばされ、今回のコードで評価し直されない。残す場合は `logs/legacy/` に置く。
6. **lora-reuse の ImageNet-R のチェックポイントと結果がないことを確かめる。**
   - 対象：`checkpoints/.../ImageNetR-{T}-lora-reuse/` と `logs/results/.../lora-reuse/ImageNetR-*/`
   - あると、学習や評価が飛ばされる。古いものがあれば `legacy/` に移す。
7. **環境を整える。**
   - `git pull` で `exp/imagenet-r` を取得し、`uv sync` を実行する（サーバーは cu126）。
   - `wandb` にログインしておく（学習は `wandb.init` を呼ぶ）。
8. **準備用のスクリプトを実行する**（§3.1）。サブセットを作り、1〜3 が整っているかを確かめる。エラーがなくなるまで直してから実験を起動する。
9. **メモリを確かめる。** full の T=50 の TIES と Tunable TIES は、1 ジョブで約 216 GB を使う（CIFAR-100 での実測。モデルが同じなので ImageNet-R でも同じ）。CIFAR-100 のときと同じサーバーで行う。

## 3. 実装内容

### 3.1 データの準備（[src/datasets/imagenetr.py](../src/datasets/imagenetr.py)、`src/datasets/prepare_imagenetr.py`）

今のコードには、並列に実行したときに問題になる箇所が 2 つある。

- **分割**：`imagenet-r/train`・`test` がないと、シードなしの `random_split` で分割し直し、ファイルを移動する。パスを間違えただけで、別の分割で実験が進んでしまう。
- **サブセット**：初めて読むときに `imagenet-r_subsets/<md5>/` にシンボリックリンクを作り、ディレクトリがあれば作成済みとみなす。3 つのワーカーが同時に作ると、`FileExistsError` で落ちるか、作りかけのディレクトリを使ってクラスが欠けたまま学習が進む。

そこで、次のように変える。

1. **サブセットを作るのは準備用のスクリプトだけにする。** 実験の前に 1 回だけ実行する。
   ```bash
   uv run python -m src.datasets.prepare_imagenetr
   ```
   このスクリプトは次のことをする。
   - `$MAGMAX_DATA_DIR/imagenet-r/{train,test}` があることを確かめる。
   - train と test のサブセットを作る。一時ディレクトリに作ってから `rename` で置き換えるので、途中で止まっても作りかけのものは残らない。作りかけのサブセット（`config.json` がないもの）が残っていれば、作り直す。
   - クラス数（200）、train・test の画像数、T ごとの「タスクあたりの最少画像数」を表示する。
   - full のチェックポイントと `zeroshot.pt`（§2 の 1、2）がそろっているかを確かめ、欠けているものを一覧にする。
   - 問題があれば、終了コードを 0 以外にする。
2. **データセットを読むときは、完成したサブセットだけを使う。** サブセットが完成していなければ（`config.json` は最後に書かれる）、準備用のスクリプトを実行するよう求めるエラーにする。
3. **分割がなければエラーにする。** `train/`・`test/` がないときに分割し直すコードは消す。

### 3.2 マージだけのジョブ（[scripts/vision/run_parallel_experiment.sh](../scripts/vision/run_parallel_experiment.sh)）

- `merge_only_variants` を追加する（例：`merge_only_variants="full"`、すべてなら `"full lora-reuse"`）。
- ここに含まれる variant のジョブでは、`finetune.sh` を呼ばない。その代わり、チェックポイント（full は `finetuned_{i}.pt`、LoRA は `adapter_{i}.pt`）が全 split そろっているかを確かめ、欠けていればそのジョブを失敗させる。
- チェックポイントのパスは、学習とマージが使う `_ckpt_dir`（[src/backends/vision_backend.py](../src/backends/vision_backend.py)）から求める。シェルでパスを組み立て直さない。
- `dataset=ImageNetR` のとき、起動時にサブセットが完成しているかを確かめ、完成していなければ何も始めずに止まる。

### 3.3 実行の記録（[scripts/vision/launch_experiment.sh](../scripts/vision/launch_experiment.sh)）

起動するたびに、`run_info.txt` に次の 2 つを足す。

- `zeroshot.pt` と `zeroshot_unpacked.pt` の sha256
- その時点にある、そのデータセットのチェックポイントの一覧（パス・サイズ・更新時刻）。`checkpoints_<日時>.txt` に書き出す。

### 3.4 実験の設定（`scripts/vision/experiments/`）

| ファイル | variants | merge_fns | 学習 | 実行の記録 |
|---|---|---|---|---|
| `imagenetr_lora_reuse.sh` | full、lora-reuse | finetune、magmax、masked_magmax_with_targetdata（labels、ot_embedded） | lora-reuse だけ学習する（`merge_only_variants="full"`） | `logs/runs/vision/imagenetr_lora_reuse/` |
| `imagenetr_tunable_ties.sh` | full、lora-reuse | ties、weighted_ties_with_targetdata（labels、ot_embedded） | しない（`merge_only_variants="full lora-reuse"`） | `logs/runs/vision/imagenetr_tunable_ties/` |
| `imagenetr_baselines.sh` | lora-reuse | random_mix、average | しない（`merge_only_variants="lora-reuse"`） | `logs/runs/vision/imagenetr_baselines/` |

- 結果は `logs/results/vision/ViT-B-16/sequential_finetuning/class_incremental/{full,lora-reuse}/ImageNetR-{5,20,50}/taskseq_A/<merge_fn>/` に書かれる。CIFAR-100 の結果と同じディレクトリに並ぶが、データセット名のディレクトリで分かれるので混ざらない。
- `imagenetr_tunable_ties.sh` は、CIFAR-100 の `tunable_ties.sh` と同じ 2 段階で動かす。
  - **small**：T=20、5 を GPU 3 枚で動かす。full の T=20 は同時に 2 ジョブまで。
  - **large**：T=50 を GPU 0 の 1 枚だけで、1 つずつ動かす。
- `imagenetr_lora_reuse.sh` の full の T=50 は MAGMAX 系だけなので、CIFAR-100 の `lora_reuse.sh` と同じく、段階に分けずに動かす。

### 3.5 CIFAR-100 専用の書き方を直す

- `run_parallel_experiment.sh` のコメント（「Fine-tune and merge CIFAR-100」など）を、データセットに依存しない書き方にする。
- [src/eval.py](../src/eval.py) で T=50 の target 26 を飛ばすときのログ（「cifar100-50」）を直す。飛ばす処理はそのまま残す。T=50 で全 50 タスクから均等に 200 件取ると、タスクあたり 4 件で meta データが 0 件になるので、ImageNet-R でも必要である。

### 3.6 テスト

- サブセットの作成：一時ディレクトリから置き換わること、作りかけのものを作り直すこと、未完成のサブセットを読もうとするとエラーになること。
- 分割がないときにエラーになり、ファイルが移動されないこと。
- マージだけのジョブ：チェックポイントの確認が、`_ckpt_dir` と同じパスを見ること。欠けていれば失敗すること。
- 既存のテスト（`test_cli_surface.py` のシェルスクリプトのオプションの確認など）が通ること。シェルスクリプトから呼ぶ新しいコマンドには、ロングオプションを使わない。

## 4. 実行の手順（サーバー上）

```bash
# 0. §2 の整備をすべて終える
uv run python -m src.datasets.prepare_imagenetr               # エラーがないことを確かめる

# 1. 試し実行：結果を trial/ に分け、本番の結果ディレクトリを汚さない
#    （本番の実験が試し実行の結果を「評価済み」として飛ばさないように）
#    full（マージだけ）と lora-reuse を、ImageNetR-5・seed 3 で 1 ジョブずつ
source scripts/vision/experiments/imagenetr_lora_reuse.sh
results_prefix=trial ft_log_prefix=trial n_splits_list=5 seeds=3 \
    log_dir=logs/runs/vision/imagenetr_trial bash scripts/vision/run_parallel_experiment.sh

# 2. 本実験：1 つずつ、前のものが終わってから起動する
#    （どれも同じ GPU 3 枚を使うので、同時に動かすと GPU とメモリを取り合う）
bash scripts/vision/launch_experiment.sh imagenetr_lora_reuse
bash scripts/vision/launch_experiment.sh imagenetr_tunable_ties   # lora-reuse の学習が終わってから
bash scripts/vision/launch_experiment.sh imagenetr_baselines      # 同上
```

起動したら、次のファイルで進み具合と記録を確かめる（例は 1 つ目の実験）。

- `logs/runs/vision/imagenetr_lora_reuse/runner.log`：進み具合
- `logs/runs/vision/imagenetr_lora_reuse/status`：終わったジョブ
- `logs/runs/vision/imagenetr_lora_reuse/run_info.txt`：コミット、設定、`zeroshot*.pt` の sha256
- `logs/runs/vision/imagenetr_lora_reuse/checkpoints_<日時>.txt`：起動した時点のチェックポイントの一覧

試し実行では、次のことを確かめる。

- データの読み込みとマージ・評価が最後まで通ること。
- テストデータが足りないときの警告（`_clamp_to_available`）の出方。論文にも、ImageNet-R-20/50 では混合比を厳密には守れないとある。
- lora-reuse の 1 split あたりの学習時間。本実験にかかる時間を見積もる。

試し実行で学習した lora-reuse のチェックポイントは、本番と同じ場所（`checkpoints/.../ImageNetR-5-lora-reuse/ft-pattern_A-epochs-10-seed:3/`）に書かれる。設定が本番と同じなので、本実験はそれをそのまま使う（学習済みの split は飛ばされる）。設定を変えて試した場合は、本実験の前に消す。

## 5. 結果のまとめ

- `docs/vision_imagenetr_report.md` に、CIFAR-100 の 2 つの報告書と同じ形式でまとめる。
  - 本実験：full と lora-reuse の比較（fine-tune、MAGMAX、Tunable MAGMAX）
  - Tunable TIES と TIES の比較
  - random_mix、average を含めた論文 Table 3 と同じ形式の表
- 注意として、次のことを報告書に書く。
  - full の random_mix・average だけが旧コードで評価した結果であること。
  - 旧コードの random_mix は環境ごとに混合を決めていたが、今のコードは 1 回決めた混合を全環境で使うこと（[vision_tunable_ties_report.md](vision_tunable_ties_report.md) §1）。
- `logs/README.md` の索引に 1 行足す。
