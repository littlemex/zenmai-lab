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
| solver-iter-half | G1 articulation の solver 反復を半減 | 4xl | 3 | _retry running_ | _ |
| **height-scan-halffreq** | update_period を 2× | 2xl | 2 | **113,829** | **+1.2%** |
| **height-scan-lowres** | resolution 0.10→0.15, size 1.6×1.0→1.0×0.6 | 2xl | 2 | **115,116** | **+2.3%** |
| height-scan-none (G1-Rough での) | height_scanner=None + plane terrain | 2xl | 2 | エラー | observation が height_scan を参照していて Rough cfg では設定不整合 |
| **combined** | contact-scope + solver-iter + height-scan-halffreq | 4xl | 3 | _retry running_ | _ |

**観察 (現時点)**:
- `contact_forces` の prim_path 絞り込み（23 → 3 リンク）は **fps 変化なし** (+0.4%)。「センサー個数」ベースの仮説は外れる。GPU 帯域への寄与は無視できる。
- `height_scanner` の頻度・解像度を半減すると **+1〜2% 程度の改善**。raycast コストはあるが思ったほど大きくない。
- `height_scanner=None` を Rough cfg に当てても動かない（observation 側に依存がある）。Flat タスクへ切り替えるのが正しい。
- G1-Flat (Rough → Flat) は section 3 で **+18.3%** を実測 = ある意味 height-scan-none + raycaster 撤去 + terrain 単純化の合算効果。

## 5. 推奨優先度（実測完了後に確定）

実測で fps 改善が確認できた順に番号を振る。

1. ⏳ num_envs を上げる (4096→16384 で実測 +56%)
2. ⏳ contact-scope-ankle (実験 4)
3. ⏳ height-scan の調整 (実験 6)
4. ⏳ solver-iter-half (実験 4)
5. ⏳ combined stack (実験 7)

## 6. 残課題

- [ ] g7e.2xlarge (Blackwell + GDDR7 1597 GB/s) で num_envs=16384 ベンチ
- [ ] g6e.12xlarge (4× L40S) で `--distributed` マルチGPU
- [ ] Isaac Lab v2.2 vs v2.3 の差分
- [ ] 顧客向けセルフ診断スクリプト整備
