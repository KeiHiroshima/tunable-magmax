# Tunable TIES の結果のまとめ（vision と NLP）

TIES-Merging に Tunable MAGMAX と同じ選好ベクトルを入れた **Tunable TIES**（`--merge_fn weighted_ties_with_targetdata`）を、vision（CIFAR-100）と NLP（LSB）の両方で評価した。ここでは 2 つの結果を並べて、共通する傾向と違いをまとめる。

- 手法の定義、vision の詳細：[vision_tunable_ties_report.md](vision_tunable_ties_report.md)
- NLP の詳細：[nlp_tunable_ties_report.md](nlp_tunable_ties_report.md)

## 1. 手法（要約）

タスク i のタスクベクトルを τ_i、ターゲット環境の選好ベクトルを w とする。TIES の 3 段階のうち、後の 2 段階に w を入れる。

1. **trim**：各 τ_i の絶対値の上位 20 % を残す（TIES と同じ）。
2. **符号の選出**：sign_p = sign(Σ_i w_i·τ̂_i[p])。
3. **平均**：符号が一致したタスクだけで、w で重み付けした平均を取る。

w がすべて等しいと TIES と一致し、w_i = 0 のタスクはマージから外れる。

## 2. 2 つの実験の違い

| | vision | NLP |
|---|---|---|
| ベンチマーク | CIFAR-100 のクラス増加、T = 5 / 20 / 50、タスク順序 A | LSB（15 タスクのテキスト分類）、Order 4 / 5 / 6 |
| モデル | ViT-B/16 | t5-base、t5-large |
| 学習の設定 | 全パラメータ学習、LoRA reuse | t5-base 全パラ、t5-base LoRA reuse、t5-large LoRA（タスクごとに足し込む）、t5-large LoRA reuse |
| seed | 3 / 4 / 5 | 3 / 4 / 5 |
| 選好ベクトル | ターゲットのデータから推定する。Label（ラベル分布から）と OT（埋め込みの最適輸送から）の 2 通り | 環境の混合比そのもの（どのタスクから何割を引いたか）。推定はしない |
| ターゲット環境 | `target_data_config.json`（T = 5、20 は 26 環境、T = 50 は 25 環境） | `target_data_config_lsb.json`（25 環境 + 全タスク均等） |
| 予測の方式 | クラスの分類ヘッド | ラベル文字列の生成（完全一致） |

NLP の選好ベクトルはタスクの混合比そのものなので、vision の Label に近い（どちらもターゲットがどのタスク・クラスから来たかの情報を使う）。OT はターゲットの埋め込みだけから推定するので、情報がずっと少ない。

## 3. 結果

値は、ターゲット環境での精度の平均（D1–D5 の平均）の差で、単位はポイント（%）。各設定 3 seed の平均。

### 3.1 Tunable TIES − TIES

| ドメイン | 設定 | 選好ベクトル | 差 |
|---|---|---|---|
| vision | 全パラ T = 5 / 20 / 50 | Label | +2.1 / +4.3 / +7.9 |
| | | OT | +0.8 / +1.1 / +1.9 |
| | LoRA reuse T = 5 / 20 / 50 | Label | +1.4 / +3.9 / +6.4 |
| | | OT | +0.8 / +1.7 / −0.6 |
| NLP | t5-base 全パラ | 混合比 | +3.00 |
| | t5-base LoRA reuse | 混合比 | +0.12 |
| | t5-large LoRA | 混合比 | −0.40 |
| | t5-large LoRA reuse | 混合比 | +4.90 |

### 3.2 Tunable TIES − Tunable MAGMAX（同じ選好ベクトルどうし）

| ドメイン | 設定 | 選好ベクトル | 差 |
|---|---|---|---|
| vision | 全パラ T = 5 / 20 / 50 | Label | −0.7 / −0.4 / −0.5 |
| | | OT | −0.3 / −0.5 / −1.5 |
| | LoRA reuse T = 5 / 20 / 50 | Label | −2.9 / −2.3 / −2.7 |
| | | OT | −1.6 / −2.5 / −1.9 |
| NLP | t5-base 全パラ | 混合比 | −0.18 |
| | t5-base LoRA reuse | 混合比 | −3.37 |
| | t5-large LoRA | 混合比 | −3.54 |
| | t5-large LoRA reuse | 混合比 | −0.98 |

### 3.3 環境ごとに TIES を上回った割合

| ドメイン | 設定 | 割合 |
|---|---|---|
| vision（Label） | 全パラ T = 5 / 20 / 50 | 90 % / 96 % / 99 % |
| | LoRA reuse T = 5 / 20 / 50 | 90 % / 97 % / 95 % |
| vision（OT） | 全パラ T = 5 / 20 / 50 | 60 % / 64 % / 64 % |
| | LoRA reuse T = 5 / 20 / 50 | 83 % / 82 % / 39 % |
| NLP（混合比） | t5-base 全パラ / t5-base LoRA reuse / t5-large LoRA / t5-large LoRA reuse | 70 % / 57 % / 64 % / 78 % |

vision は環境 × seed（78 または 75 ケース）、NLP は環境 × Order × seed（225 ケース）で数えた。

## 4. 共通する傾向

**選好ベクトルを入れると、多くの場合 TIES より良くなる**
- 平均で TIES を上回ったのは、16 設定のうち 14 設定だった。
  - vision の Label：6 設定すべて
  - vision の OT：6 設定のうち 5 設定
  - NLP：4 設定のうち 3 設定。ただし明確なのは 2 設定（+3.0、+4.9）で、残りの 1 設定はほぼ同じ（+0.12）
- 環境の単位では、すべての設定で TIES を上回るケースのほうが多かった。ただし vision の OT の LoRA reuse T = 50（39 %）だけは例外である。
- 選好ベクトルを入れる仕組みは、MAGMAX に限らず TIES でも働く。

**平均では、Tunable MAGMAX を上回った設定はない**
- 16 設定のすべてで、Tunable MAGMAX より低かった。
- 全パラメータ学習では差が小さい。vision の Label で −0.4〜−0.7、NLP の t5-base で −0.18 である。混合比ごとの列では、Tunable MAGMAX を上回る列もある（vision の T = 50 の D3・D4、NLP の t5-base の D_tar,2・D_tar,5）。
- LoRA reuse の差は、vision では全パラメータ学習より大きかった（Label で −2.3〜−2.9）。NLP でも t5-base の LoRA reuse は −3.37 で、全パラメータ学習（−0.18）より差が大きい。

**TIES の土台の強さと、選好を入れた後の順位は一致しない**
- vision の T = 20、50 と NLP の LoRA reuse では、TIES が MAGMAX と同程度か上回る。それでも選好を入れると、Tunable MAGMAX のほうが上になる。

## 5. 違い

**NLP では改善が安定しない**
- vision の Label では、T が大きいほど改善が大きく、36 セル（2 設定 × 3 T × 6 列）のすべてで TIES を上回った。
- NLP では、同じ設定の中でも Order によって符号が変わる。たとえば t5-large LoRA では、Order 4 / 5 / 6 で +1.95 / −6.83 / +3.69 だった。
- 原因の 1 つは、NLP にだけ見られる、特定のタスクの精度が 0 % 近くまで落ちる現象である。LSB の Order 5 で、系列の初めに学習する MultiRC や BoolQ と、5 番目の CB を混ぜた環境で起きる。Tunable TIES で 0 % 近くまで落ちたタスクは、77 件のうち 76 件がここで起きた（[nlp_tunable_ties_report.md](nlp_tunable_ties_report.md) §2.5）。
  - Tunable MAGMAX でも起きるので、選好ベクトルで少数のタスクに絞る手法に共通の問題と考えられる。
  - vision はクラスの分類ヘッドで予測するので、出力がラベルの集合から外れることはない。NLP はラベル文字列を生成するので、別のタスクのラベルで答えると 0 % になる。この違いが関係している可能性がある（生成した文字列を保存していないので、確かめていない）。

**選好ベクトルの情報量**
- NLP の選好ベクトルは混合比そのもの（推定の誤差がない）なのに、改善は vision の Label より小さく、ばらつきも大きい。選好ベクトルの正確さだけでは、Tunable TIES の改善の大きさは決まらない。

## 6. 検証していないこと

両方の実験に共通する、まだ調べていない点：

- **Tunable MAGMAX との差の理由**：重み付き平均による値の縮み、上位 20 % の trim、符号の選出のどれが効いているか。top-k は 20 % の 1 通りだけで試した。
- **λ の調整**：λ = 0.5 は、どの手法についても調整していない。
- **同じ情報を使う対照手法**：ターゲットの情報を使う単純な手法（例：選好ベクトルで重み付けしたタスクベクトルの平均）がまだない。
- **NLP で特定のタスクの精度が落ちる原因**：生成した文字列を保存し、どんな出力になっているかを確かめる必要がある。
- **一般性**：vision は CIFAR-100 とタスク順序 A だけ、NLP は LSB だけで実行した。どちらも 3 seed で、統計検定はしていない。
