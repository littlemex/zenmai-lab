# 実験ノート

`benchmark_rsl_rl.py` を Isaac Lab v2.2.0 / Isaac Sim 5.1.0 で複数条件下で計測した記録。
詳細な数値は `results/*/summary.csv` に置いている。本ノートはコマンドと結論のサマリ。

## 共通条件

| 項目 | 値 |
|------|----|
| Isaac Sim | 5.1.0-rc.19 |
| Isaac Lab | release/2.2.0 (v0.44.9) |
| RL framework | rsl_rl (PPO) |
| Task (今のところ) | `Isaac-Velocity-Rough-G1-v0` (Unitree G1, 凹凸地形ロコモーション) |
| Driver | NVIDIA 580.126.09 / CUDA 13.0 |
| GPU 警告 (両 instance 共通) | `ECC enabled`, `PCIe link width current (8) and maximum (16) for device 0 don't match` |
| Region | ap-northeast-1 |
| 実行モード | `--headless` |

---

## 実験 1: g6e.2xlarge ベースライン

**目的**: 単一インスタンスでのベースライン取得。後続の実験との比較対象。

**インスタンス**: g6e.2xlarge (8 vCPU AMD EPYC 7R13 @ 2.6 GHz, L40S 46 GB)

**コマンド** (SSM モード)

```bash
cd zenmai-lab/experiments/isaac-lab-bench
cp ssm.env.sample ssm.env
$EDITOR ssm.env  # INSTANCE_ID, AWS_REGION, RUN_TAG_PREFIX=g6e2xl
bash orchestrate.sh g1
```

`g1` プロファイル = G1 5 シード × 100 iter × num_envs=4096

**結果** (5 シード平均)

| 指標 | 値 |
|------|----|
| `mean_total_fps` | **71,460 ± 222** |
| `mean_collection_fps` | 76,891 ± 261 |
| `wall_clock_s` | 173.8 ± 2.5 |
| `app_launch_ms` (warm) | ~6,200 |

**GPU 使用率** (1Hz dmon)

| 指標 | 平均 | p95 | max |
|------|-----:|----:|----:|
| SM utilization | 48% | 65% | 67% |
| Memory bandwidth | 26% | 53% | 55% |
| Power draw | 146-156 W (TDP 350 W) | 178 W | 186 W |

**観察**

- GPU が遊んでいる (SM max 67%, MEM max 55%)。何かが律速。
- 当時の仮説: CPU シングルスレッド or PCIe x8 or ECC 帯域低下。

---

## 実験 2: g6e.4xlarge で同一条件 (CPU 仮説の検証)

**目的**: vCPU を 8 → 16 に倍増した場合に fps が改善するか確認。CPU 律速説の検証。

**インスタンス**: g6e.4xlarge (16 vCPU 同型 EPYC 7R13, GPU は同じ L40S 46 GB)

**コマンド** (実験 1 と同じプロファイル、`INSTANCE_ID` のみ差し替え)

```bash
$EDITOR ssm.env  # INSTANCE_ID を 4xl 用に変更
bash orchestrate.sh g1
```

**結果** (5 シード平均)

| 指標 | g6e.2xlarge | g6e.4xlarge | 差 |
|------|-------------|-------------|------|
| `mean_total_fps` | 71,460 ± 222 | **71,892 ± 290** | **+0.6%** |
| `wall_clock_s` | 173.8 | 170.4 | −2% |

**観察**

- vCPU を倍にしても **fps はほぼ変わらない**。
- **CPU 律速ではない** ことが実測で確定。
- 当初仮説 (CPU シングルスレッド性能差が支配) は **誤り**。
- 残る容疑者: GPU メモリ帯域 / PCIe x8 / Isaac Sim 内部同期。

---

## 実験 3: num_envs スイープ (本命の発見)

**目的**: GPU が遊んでいる原因が「並列度不足」なのかを検証。num_envs を増やして GPU 使用率と fps の関係を見る。

**インスタンス**: g6e.4xlarge

**コマンド**

```bash
bash orchestrate.sh g1-num-envs
```

`g1-num-envs` プロファイル = G1, 1 シード, 100 iter, **num_envs を {4096, 8192, 16384} でスイープ**

**結果**

| num_envs | mean_total_fps | スケール率 (vs 4096) | wall (s) | total_start (s) |
|---------:|---------------:|---------------------:|---------:|----------------:|
| 4,096 | 72,316 | 1.00× | 167 | 26 |
| 8,192 | **98,563** | **1.36×** | 247 | 40 |
| 16,384 | **112,503** | **1.56×** | 432 | 71 |

**GPU 使用率 (実走中の p95 / max)**

| num_envs | SM p95 | SM max | MEM p95 | MEM max | PWR max |
|---------:|-------:|-------:|--------:|--------:|--------:|
| 4,096 | 66% | 67% | (51%)  | 51% | 182 W |
| 8,192 | 78% | 78% | (71%) | 71% | 210 W |
| **16,384** | 84% | **93%** | (96%) | **96%** | **246 W** |

**観察**

- **num_envs を増やすと fps が伸びる**。線形ではないが 4× 並列度で 1.56× fps。
- num_envs=16384 で **SM max 93% / MEM max 96% / PWR 246 W** に到達 = **ほぼ完全飽和**。
- すなわち num_envs=4096 では **GPU の並列度を使い切れていなかった**。
- L40S の **メモリ帯域 864 GB/s が真のボトルネック**。これは当初の「ECC + GDDR6 帯域」仮説と一致。
- VRAM 余裕は不明 (要計測)。num_envs=16384 で死ぬ寸前か、まだ枠ありかは VRAM ログ要確認。

---

## 暫定的な結論と推奨

| 仮説 | 検証結果 |
|------|----------|
| CPU シングルスレッド/コア数が律速 | **却下** (vCPU 倍増で fps 変化なし) |
| GPU メモリ帯域が律速 | **支持** (num_envs=16384 で MEM 96% 飽和) |
| num_envs を上げると改善する | **支持** (4096→16384 で fps 1.56×) |

### 短期推奨
1. **num_envs を 8192 以上に上げる** — コード変更ほぼなし、効果絶大 (1.36×)
2. それでも足りなければ num_envs=16384 (リスク: VRAM 上限・OOM)

### 中期推奨
1. **g7e 系 (RTX PRO 6000 Blackwell, 1597 GB/s)** へスケール
   - L40S 比 1.85× 帯域、メモリ帯域律速のワークロードに直接効く
2. ~~g6e.4xlarge / 8xlarge へのスケールアップ~~ — CPU 増やしても効果薄い、スキップ可
3. マルチノード DDP (HyperPod / Training Jobs) — 仮説検証必要、まだ未実施

---

## 既知のバグメモ

- 旧 `orchestrate.sh` (commit `9a918cd` 時点) は SWEEP プロファイル時に各 num_envs の tarball を同じ展開先に flat に展開しており、`summary.csv` が最後の値で上書きされていた。今回の `g1-num-envs` 結果は orchestrate.sh の stdout から手動でサルベージし `sweep-summary.csv` として記録。
- 修正済み (次 commit): tarball を `<LOCAL_OUT>/<TAG>/` のサブディレクトリに展開するように変更。

---

## 未確認事項 (open questions)

- num_envs=16384 時の VRAM 使用量 (今回 dmon に fb メトリクス含めるべきだった)
- num_envs=24576 / 32768 で OOM になるか / fps 上限はどこか
- g7e.2xlarge での同一スイープ → メモリ帯域律速説の最終証明
- マルチノード時のスケール特性 (NCCL + EFA でどこまで落ちないか)
- Isaac Lab v2.3.2 / Isaac Sim 5.1 vs v2.2.0 / 5.1 で fps 差があるか

---

## 再現コマンドの早見

```bash
git clone git@github.com:littlemex/zenmai-lab.git
cd zenmai-lab/experiments/isaac-lab-bench

# DCV モード (EC2 上で直接)
cp bench.env.sample bench.env
source bench.env
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 g1-prod \
  "42 123 456 789 1337" 100 4096

# SSM モード (ローカルから)
cp ssm.env.sample ssm.env
$EDITOR ssm.env
bash orchestrate.sh g1            # 実験 1, 2 と同条件
bash orchestrate.sh g1-num-envs   # 実験 3 (num_envs スイープ)
```
