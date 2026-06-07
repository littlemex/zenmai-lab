# 計測結果サマリ

ベースライン、ablation、num_envs スイープの結果を一表にまとめる。
完了したセルから順に埋めている。空欄は計測中／未完了。

## 1. ベースライン (G1-Rough, num_envs=4096, 100 iters)

| 環境 | 構成 | seeds | mean_total_fps | std | wall (s) | App Launch |
|------|------|------:|---------------:|----:|---------:|-----------:|
| g6e.2xlarge | デフォルト | 5 | 71,460 | ±222 | 174 | 6.2 s |
| g6e.4xlarge | デフォルト | 5 | 71,892 | ±290 | 170 | 6.2 s |

**観察**: vCPU 倍増 (8→16) の効果は +0.6%。CPU は律速ではない。

## 2. num_envs スケーリング (g6e.4xlarge, G1-Rough)

| num_envs | mean_total_fps | スケール率 | wall (s) | SM max | MEM max | VRAM max | PWR max |
|---------:|---------------:|-----------:|---------:|-------:|--------:|---------:|--------:|
| 4,096 | 72,316 | 1.00× | 167 | 67% | 51% | 7,297 MB (16%) | 182 W |
| 8,192 | 98,563 | 1.36× | 247 | 78% | 71% | 9,951 MB (22%) | 210 W |
| 16,384 | 112,503 | 1.56× | 432 | 93% | 96% | 15,237 MB (33%) | 246 W |
| **24,576** | **116,518** | **1.61×** | 366 | _要計測_ | _要計測_ | _要計測_ | _要計測_ |
| **32,768** | **117,163** | **1.62×** | 487 | _要計測_ | _要計測_ | _要計測_ | _要計測_ |

**観察**: num_envs=16384 で SM/MEM/PWR が同時飽和。L40S のメモリ帯域 864 GB/s が天井。
24576 / 32768 までスイープしても **fps はわずか +5% で頭打ち**。OOM はせずどちらも完走したが、メモリ帯域がボトルネックなので並列度を増やしても効果が逓減する。**実用上の sweet spot は num_envs=16384**（fps と wall_clock のバランス）。VRAM は 32768 envs でもまだ 50% 以下のはず（要 dmon 確認）。

## 3. タスク種別の影響 (4096 envs)

| タスク | 環境 | seeds | mean_total_fps | std | vs G1-Rough |
|--------|------|------:|---------------:|----:|------------:|
| G1-Rough | g6e.2xlarge | 5 | 71,460 | ±222 | (基準) |
| **G1-Flat** | g6e.2xlarge | 3 | **84,508** | ±142 | **+18.3%** |

**観察**: height_scanner と rough terrain raycast を外すだけで +18.3%。
`Velocity-Flat-G1-v0` は同じ G1 ロボットでフラット地形のみ。

## 4. PhysX ablation (G1-Rough, num_envs=16384, GPU は L40S 共通)

| ablation 名 | 内容 | host | seeds | mean_total_fps | vs default 112,503 |
|-------------|------|------|------:|---------------:|-------------------:|
| (default) | パッチなし | 4xl | 1 | 112,503 | (基準) |
| contact-scope-ankle | contact_forces を torso + ankle (3 リンク) のみに | 4xl | 1 | 112,955 | **+0.4%** (誤差) |
| **solver-iter-half** | G1 articulation の solver 反復を半減 (pos 8→4, vel 4→1) | 4xl | 3 | **148,240 ± 503** | **+31.7%** ★ 本命 |
| height-scan-halffreq | update_period を 2× | 2xl | 2 | 113,829 | +1.2% |
| height-scan-lowres | resolution 0.10→0.15, size 1.6×1.0→1.0×0.6 | 2xl | 2 | 115,116 | +2.3% |
| height-scan-none (G1-Rough での) | height_scanner=None + plane terrain | 2xl | 2 | エラー | observation が height_scan を参照していて Rough cfg では設定不整合 |
| **combined** | contact-scope + solver-iter + height-scan-halffreq | 4xl | 3 | **151,512 ± 552** | **+34.7%** ★ 最大 |

**観察**:
- ★ **`solver-iter-half` が圧倒的勝者: +31.7%**。Section 5 (nsys) のとおり、PhysX articulation TGS solver が GPU 時間の **48%** を占めており、その solver 反復回数を半減すれば直接 fps に反映される。学習品質 (max_rewards) は default と差なし。
- `contact_forces` の prim_path 絞り込みは **fps 変化なし** (+0.4%)。「センサー個数」ベースの仮説は外れた。Section 5 の kernel ランキングでも `convexTrimeshNarrowphase` は 2.4% でしかない。
- `height_scanner` 系は **+1〜2% 程度**。raycast コストはあるが Section 5 で確認済みの 3.3% と整合。
- `height_scanner=None` を Rough cfg に当てても動かない (observation 側に依存)。Flat タスクへ切り替えるのが正しく、Section 3 で **+18.3%** 実測。

## 5. nsys プロファイル: GPU 時間内訳 (g6e.2xl, n=4096, 30 iters)

`scripts/profile_nsys.sh` 経由で全 30 iter を `nsys profile --trace=cuda,nvtx` でラップ。`scripts/benchmarks/benchmark_rsl_rl.py` の生実行 (71k fps) より低めの fps_total = 56,512 になるが nsys オーバーヘッド込み。重要なのは絶対値ではなく **カーネル時間配分**。

### Top 10 GPU カーネル (累積時間 %)

| % | カーネル | カテゴリ |
|---:|---------|----------|
| **39.1** | `artiSolveInternalConstraintsTGS1T` | PhysX TGS solver (articulation 内部制約) |
| **9.0** | `stepArticulation1TTGS` | PhysX articulation step |
| 3.3 | `raycast_mesh_kernel` | Warp raycast (height_scan) |
| 3.2 | `at::native::index_kernel` | PyTorch indexing |
| 2.5 | `updateBodiesLaunch_Part2` | PhysX body update |
| 2.4 | `convexTrimeshNarrowphase` | PhysX 接触ナローフェーズ |
| 2.4 | `computeUnconstrainedSpatialInertiaLaunchPartial1T` | PhysX |
| 2.1 | `index_put_kernel` | PyTorch |
| 2.0 | `elu_backward_kernel` | PyTorch (NN backward) |
| 1.9 | `computeUnconstrainedAccelerationsLaunch1T` | PhysX |

**観察**:
- **PhysX articulation 系 (1, 2, 5, 7, 10) で約 55%**。残り は別の PhysX カーネル群と PyTorch RL 計算。
- Raycast 単独は **3.3%** だけ。height-scan ablation で +1〜2% の効果しか出なかった事実と完全に整合。
- PyTorch 関連 (index, elu_backward, etc.) で **約 10%**。RL ネットワークサイズ削減や `num_mini_batches=2` などで触れる範囲。

`results/g1-nsys-test5/profile/42/kernels.csv` が完全なリスト (top 数百カーネル)。`gpu-gaps.txt` も同ディレクトリ。

## 6. 推奨優先度 (実測ベース、効果順)

| 順 | 推奨 | 実測効果 | リスク・条件 |
|---:|------|---------|-------------|
| **1** | **`solver-iter-half`** (G1 articulation の solver 反復を半減) | **+31.7%** | 学習収束への影響を要確認。本ベンチでは max_rewards に有意な変化なし。試すコストが低くリターン大 |
| **2** | **num_envs を 16384 へ** (4096 から増やす) | **+56% (vs 4096)** | VRAM 33% 利用、余裕あり。num_envs > 16384 はリターン逓減で +5% のみ |
| 3 | G1-Rough → G1-Flat (rough 不要なら) | +18% (4096) | 学習要件次第。rough 地形が要件なら採用不可 |
| 4 | num_envs を 24576-32768 へ | +5% (vs 16384) | 帯域飽和で逓減。wall_clock も増える |
| 5 | `height-scan-halffreq` / `lowres` | +1〜2% | 観測周波数 / 精度低下のトレードオフ |
| 6 | `combined` (1 + 5 + contact-scope) | **+34.7%** (vs default 16384) / **+112%** (vs default 4096) | 加算的に効いた。デフォルト 4096 に対して **fps が 2.12 倍**。30h → 約 14h |
| - | `contact-scope-ankle` | +0.4% (誤差) | 推奨しない。実装コスト > 効果 |
| - | g6e.2xl → g6e.4xl (vCPU 倍) | +0.6% (誤差) | CPU はボトルネックでない。インスタンスを増やすなら GPU 側で |

**最大の発見**: nsys プロファイルが「**PhysX articulation solver が GPU 時間 48% を占める**」と教えてくれて、`solver-iter-half` を狙い打ちで効かせて **+31.7%** を実測した。「センサー個数」「num_envs スケーリング」のような直感的な仮説より、**実測ホットスポットからの逆算**が圧倒的に当たった。

## 7. 残課題

- [ ] g7e.2xlarge (Blackwell + GDDR7 1597 GB/s) で num_envs=16384 ベンチ
- [ ] g6e.12xlarge (4× L40S) で `--distributed` マルチGPU
- [ ] Isaac Lab v2.2 vs v2.3 の差分
- [ ] 顧客向けセルフ診断スクリプト整備
