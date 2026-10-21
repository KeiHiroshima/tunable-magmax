# Core Space でのマージ（`--merge_space core`）

LoRA で学習したタスクベクトルを、全タスクで共通の基底へ回転してからマージする機能の設計と使い方をまとめる。
回転の方法は Core Space Merging（Panariello et al., *Accurate and Efficient Low-Rank Model Merging in Core Space*, NeurIPS 2025, [refs/AccurateandEfficientLow-RankModelMerginginCoreSpace.pdf](../refs/AccurateandEfficientLow-RankModelMerginginCoreSpace.pdf)）に従う。
実装は [src/merging/core_space.py](../src/merging/core_space.py)。

## 概要

- `--merge_space full`（既定値）は、これまでどおり回転を加えず、LoRA を適用した重み行列（q/v）の要素ごとにマージする。
- `--merge_space core` は、各タスクベクトルを共通基底での小さな「core 行列」に回転してから**既存のマージ関数をそのまま**かけ、結果を元の座標へ戻す。
- 回転の基底は LoRA の A・B から作るため、`--finetune_mode lora` のときだけ使える（それ以外は起動時にエラー）。
- 対象：vision（CIFAR100 / ImageNetR）と StdCL / LSB。CITB（seq2seq）は LoRA がないため対象外。

## 処理の流れ

1 つの LoRA 重み（例：`...attn.q.weight`）について示す。d_out × d_in は重みの形（ViT-B/16 の q/v は 768 × 768）、r = 8、α = 32、T はタスク数、R = T·r。

| 手順 | 処理 | 行列と次元 | full との違い |
|---|---|---|---|
| 1 | アダプタを読み込む | A_j: r × d_in、B_j: d_out × r | なし |
| 2 | タスクベクトルを作る | τ_i = Σ_{j≤i} (α/r) B_j A_j（ind-ft は j = i のみ）: d_out × d_in | なし |
| 3 | 類似度から重み w を計算（提案手法のみ） | w ∈ ℝ^T。元の座標の τ_i を使う | なし |
| 4 | 共通基底を作る | V_ref = svd([A_1; …; A_T]) の右特異ベクトル: d_in × R<br>U_ref = svd([(α/r)B_1, …, (α/r)B_T]) の左特異ベクトル: d_out × R | **追加** |
| 5 | 回転する | M_i = U_refᵀ τ_i V_ref: R × R | **追加** |
| 6 | マージする | M_merged = f(M_1, …, M_T): R × R | 入力が M_i になる |
| 7 | 逆回転する | ΔW = U_ref M_merged V_refᵀ: d_out × d_in | **追加** |
| 8 | 適用して評価する | W_0 + λ ΔW | なし |

- 手順 4 の基底は 1 回の実行につき一度だけ作る（目標環境に依存しない）。
- R が重みの次元を超えるときは、各側の基底が min(d, R) 本になり、M_i は min(d_out, R) × min(d_in, R) になる。それでも情報は失われない。
- 基底は必ずアダプタ (A, B) から作る。密な τ_i を SVD して作ると論文とは別の基底になる。

### 元の実装との対応

論文と元の実装（[core-space-merging](https://github.com/apanariello4/core-space-merging) の `get_core_matrices`）は、A_i・B_i をそれぞれ SVD し、R_B = U_refᵀ U_B と Q_A = V_Aᵀ V_ref で揃えて core 行列を作る。これを展開すると

```
M_i = (U_refᵀ U_B)(Σ_B V_Bᵀ U_A Σ_A)(V_Aᵀ V_ref) = U_refᵀ B_i A_i V_ref
```

となり、手順 5 の U_refᵀ τ_i V_ref と同じ行列になる。ここでは密な τ_i から計算するため、逐次学習（τ_i がアダプタ 0..i の和）にもそのまま使える。両者が数値的に一致することは `test/test_core_space.py::test_core_matrices_from_dense_task_vectors_match_the_reference_formula` で確認している。

### 線形・非線形のマージ

- 線形なマージ（`average`）は、full と core で結果が完全に一致する（論文の式 (11)）。
- 要素ごとに選ぶマージ（`magmax`、`ties`、`random_mix`、提案手法）は、core 行列の要素について選ぶので結果が変わる。core の 1 要素 M[j,k] は、ΔW 全体に広がる rank-1 成分 M[j,k]·u_j v_kᵀ の係数である。
- `finetune`（マージしない）は回転しない。

## 要素数の数え方

`--merge_space core` では、提案手法の予算（タスク i に割り当てる要素数 = int(要素数 × w_i)）は **core 行列 M_i の要素**で数える。1 重みあたり R² 個で、モデル全体ではその和になる。結果 JSON の `num_params_all` と `num_unaligned` も同じ単位で、full の値とは直接比べられない。

ViT-B/16（q/v × 12 層 = 24 重み）の場合：

| T | R | 1 重みの R² | 全体（core） | 全体（full） |
|---|---|---|---|---|
| 5 | 40 | 1,600 | 38,400 | 14,155,776 |
| 20 | 160 | 25,600 | 614,400 | 14,155,776 |
| 50 | 400 | 160,000 | 3,840,000 | 14,155,776 |

## 実行方法

スクリプトの環境変数 `merge_space` で切り替える（既定値 `full`）。学習は変わらないので、学習済みのアダプタがあればマージだけが走る。

```bash
# vision（CIFAR100 の 5/20/50 分割）
merge_space=core merge_fns="average ties magmax masked_magmax_with_targetdata" \
    bash scripts/vision/run_lora_sweep.sh

# t5-base × LSB
merge_space=core bash scripts/nlp/run_t5-base_lora_lsb_single_gpu.sh
```

結果は `<merge_fn>-core` という名前で、full の結果とは別に保存される。

- vision：`${results_db}/{merge_fn}-core/[{metric}/]{merge_fn}-core_lambda0.5_[{metric}_]target{id}_seed{s}.json`
- NLP：`.../ft-pattern_{p}-epochs-{e}-seed:{s}/{merge_fn}-core/target{id}_seed{s}.json`

どちらの JSON にも `merge_space` を書き込む。

## 集計

[postprocessing/build_comparison_table.py](../postprocessing/build_comparison_table.py) と [postprocessing/aggregate_nlp_logs.py](../postprocessing/aggregate_nlp_logs.py) に `--merge_space {full,core,both}` を追加した（既定値 `full`）。`both` は full と core を並べ、core 側の手法名には「(Core)」が付く。

```bash
uv run python postprocessing/aggregate_nlp_logs.py --merge_space both
```

## 今後の課題

- **マージ係数 λ の調整**：λ = 0.5（vision は固定、NLP は `--scaling_coef` の既定値）は full 空間のマージ向けに決めた値である。core 空間では同じノルムがずっと少ない要素に集まるため、`magmax` や `ties` の結果の大きさが full と変わり、0.5 が合わない可能性がある。今後ハイパーパラメータとして調整する可能性があるが、今回は対応していない（コード中の `TODO(--merge_space core)` を参照）。
- **提案手法のマスクの意味**：core 空間では 1 要素が重み全体に効く rank-1 成分の係数になる。full 空間の要素ほど冗長ではない（論文 Fig. 4）ため、マスクで捨てる情報が大きくなる可能性がある。効果は実験で確かめる。
- **ties の閾値**：TIES の上位 20% も core 行列の要素で決まる。
