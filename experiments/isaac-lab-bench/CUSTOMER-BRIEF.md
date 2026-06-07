# Isaac Lab on EC2 — 最適化結果ブリーフ

実機 (EC2 g6e.2xlarge / g6e.4xlarge, NVIDIA L40S) で Isaac Lab v2.2 + Isaac Sim 5.1 を計測し、
**「30 時間学習を本当に短縮するレバー」** を順位付きで特定した結果。

タスク: `Isaac-Velocity-Rough-G1-v0` (Unitree G1 ヒューマノイドロコモーション、凹凸地形)
RL: rsl_rl PPO、1 iter = num_envs × 24 step

---

## 結論 1 行

**現行ワークロードに対して、`solver_position_iteration_count=8→4` と `solver_velocity_iteration_count=4→1` の 2 行変更で fps_total が +31.7% (実測 3 シード) 上がる。num_envs を 16384 に上げる併用で +56% (vs n=4096)。**

これだけで「30 時間 → 約 14-15 時間」の見込み。それ以上は通常クラス GPU の物理的天井 (L40S 864 GB/s) に当たる。

---

## 計測ベースの推奨レバー (効果順)

| 順 | レバー | 実測効果 | 場所 | リスク |
|---:|--------|----------|------|--------|
| **1** | **`solver_position_iteration_count=4, solver_velocity_iteration_count=1`** (G1 articulation) | **+31.7%** (n=16384, 3 seeds) | `G1_CFG.spawn.articulation_props` の override / `G1RoughEnvCfg.__post_init__` | 学習収束への影響を要 ablation。本ベンチでは max_rewards に有意な変化なし |
| **2** | **num_envs を 16384** (デフォルト 4096 から) | **+56%** (vs n=4096) | コマンドライン `--num_envs 16384` または env_cfg | VRAM 33% 利用、余裕あり (32GB 以上の GPU 推奨) |
| 3 | **G1-Rough → G1-Flat** (rough 不要なら) | **+18%** (n=4096) | タスク名 `Isaac-Velocity-Flat-G1-v0` | 学習要件次第。rough 地形が必須なら不可 |
| 4 | num_envs を 24576-32768 へ | **+5%** (vs n=16384) | 同上 | 帯域飽和で逓減、wall_clock も増 |
| 5 | `height_scanner.update_period × 2` | +1.2% | `__post_init__` | 観測周波数低下のトレードオフ |
| 5 | `height_scanner.pattern.resolution` 削減 | +2.3% | 同上 | 観測精度のトレードオフ |
| - | 上記 1+5 の組合せ (`combined`) | **+34.7%** (n=16384, 3 seeds) / **+112%** (vs default n=4096) | パッチ 3 個 | 加算的に効く。**デフォルトの 2.12 倍** |
| - | `contact_forces` の prim_path 絞り込み | +0.4% (誤差) | - | 効果なし |
| - | g6e.2xl → g6e.4xl (vCPU 倍増) | +0.6% (誤差) | - | CPU はボトルネックでない |

### コード片 (推奨 1: solver-iter-half)

```python
# G1RoughEnvCfg.__post_init__ 内
from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg
self.scene.robot = self.scene.robot.replace(
    spawn=self.scene.robot.spawn.replace(
        articulation_props=ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,  # 8→4
            solver_velocity_iteration_count=1,  # 4→1
        ),
    ),
)
```

---

## なぜこれが効くのか (nsys プロファイル)

1 ラン分の GPU タイムラインを `nsys profile --trace=cuda,nvtx` で取得し、kernel ごとの累積 GPU 時間を集計した。

### Top 10 GPU カーネル

| % | カーネル | カテゴリ |
|---:|---------|----------|
| **39.1** | `artiSolveInternalConstraintsTGS1T` | PhysX TGS solver (articulation 内部制約) |
| **9.0** | `stepArticulation1TTGS` | PhysX articulation step |
| 3.3 | `raycast_mesh_kernel` | Warp raycast (height_scan) |
| 3.2 | `at::native::index_kernel` | PyTorch indexing |
| 2.5 | `updateBodiesLaunch_Part2` | PhysX |
| 2.4 | `convexTrimeshNarrowphase` | PhysX 接触ナローフェーズ |
| 2.0 | `elu_backward_kernel` | PyTorch (NN backward) |

**PhysX articulation 系で計約 55%、PyTorch RL 計算で約 10%、raycast は 3.3% のみ**。

### 仮説駆動 vs データ駆動の差

検証に 1 日かけた結果、**「contact_sensor が多いから帯域を食ってるはず」→ +0.4% (誤差)**、**「nsys でホットスポット = articulation solver 48% → 反復削減」→ +31.7%** という対比になった。**プロファイル → ホットスポット → ピンポイント介入**の方が、直感的な仮説より圧倒的に効く。

---

## 物理的な上限 (これ以上は GPU を変えるしかない)

**L40S のメモリ帯域 864 GB/s が天井**。実測で num_envs=16384 のとき:
- SM 利用率 max **93%**
- メモリ帯域 max **96%**
- 消費電力 max **246 W** (TDP 350W の 70%)

これは「GPU は使い切られている」状態で、num_envs を増やしても fps はわずかしか伸びない (16384→32768 で +5%)。

**もう一段の高速化は GPU を変えるしかない**:

| GPU | メモリ帯域 | L40S 比 | EC2 instance |
|-----|----------:|--------:|--------------|
| L40S | 864 GB/s | 1.00× | g6e.* |
| RTX PRO 6000 Blackwell (Server Edition) | 1597 GB/s | 1.85× | g7e.* |
| H100 SXM | 3350 GB/s | 3.88× | p5.* (但し Isaac Sim 公式非対応 — RT Cores なし) |
| RTX 5090 (consumer Blackwell) | 1792 GB/s | 2.07× | (EC2 非提供、ローカル限定) |

⚠️ **Isaac Sim は RT Cores が必要**で、A100/H100/B200 等のデータセンタ GPU は公式非対応。p4d/p5/p6 系列ではフォールバック動作する場合があるが、訓練品質の保証外。

---

## 推奨アクション (優先度順)

### 即日適用できるもの (リスク低)

1. **solver-iter-half + num_envs=16384 + height-scan-halffreq の組合せ** → 実測 fps **151,512 (default 71k の 2.12 倍)、30h → 約 14h 見込み**。3 シード平均で std/mean < 0.4% と安定。
2. 学習収束への影響を実 task で 1 回確認 (max_iterations=200 程度の sanity ラン)

### 中期 (1-2 週間)

3. AWS の[公式マルチノード分散学習サンプル](https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/nvidia-isaac-lab) (Isaac Lab v2.3.2 + Isaac Sim 5.1.0、HyperPod EKS / SageMaker Training) でマルチ GPU DDP 検証
   - Isaac Lab 公式ベンチ: 4× L40 で fps_total が単 GPU 比 **4.35 倍**にスケール
   - 1 と組合せれば **30h → 3-4h** の可能性

### 長期 (本格運用)

4. g7e (RTX PRO 6000 Blackwell) で同一ベンチを取り、L40S vs Blackwell の実効差を確認
5. 単独 GPU 限界を超えた場合は HyperPod EKS or SageMaker Training の永続クラスタ化

---

## 顧客環境との突合せ

このブリーフは **AWS 側の参照環境** (g6e.4xlarge, ECC enabled, PCIe x8 link) で取った数値。
顧客環境との差分を 30 秒で把握するため、`scripts/diagnose.sh` を提供している:

```bash
git clone https://github.com/littlemex/zenmai-lab.git
cd zenmai-lab/experiments/isaac-lab-bench
bash scripts/diagnose.sh --task Isaac-Velocity-Rough-G1-v0
# 出力をペーストしてください
```

特に確認したい項目:
- TensorBoard の `Perf/collection_time` vs `Perf/learning_time` 比率
- 実際の num_envs と GPU VRAM 使用量
- 学習スクリプトが公式 train.py かカスタムか
- max_iterations と収束イテレーション数 (30h の中身)

---

## 再現コマンド

```bash
# 全部 EC2 上 (DCV) で
git clone https://github.com/littlemex/zenmai-lab.git
cd zenmai-lab/experiments/isaac-lab-bench
cp bench.env.sample bench.env && source bench.env

# (1) ベースライン
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 baseline "42 123 456 789 1337" 100 4096

# (2) num_envs を上げる (推奨 2)
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 n16k "42 123 456" 100 16384

# (3) solver-iter-half (推奨 1)
python3 scripts/ablations/apply_ablation.py --name solver-iter-half
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 solver "42 123 456" 100 16384

# (4) combined (推奨 1+5)
python3 scripts/ablations/apply_ablation.py --name combined
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 combined "42 123 456" 100 16384

# revert
python3 scripts/ablations/apply_ablation.py --name none

# プロファイルが見たいとき
bash scripts/profile_nsys.sh Isaac-Velocity-Rough-G1-v0 prof "42" 30 4096
```

詳細は `RESULTS-SUMMARY.md` `NOTES.md` `PERF-LEVERS.md` を参照。
