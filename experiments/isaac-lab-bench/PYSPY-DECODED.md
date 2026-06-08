# Python メインスレッド時間内訳 (py-spy 実機計測)

`scripts/profile_pyspy.sh` で取った `pyspy.svg` / `pyspy.speedscope.json` を `scripts/pyspy_tree.py` でツリー化し、**「nsys では見えなかった Python 側で何が CPU を食っているか」** を構造的に特定した結果。

`KERNELS-DECODED.md` (GPU カーネル側) と対になる、CPU 側のホットスポット詳細。

## 計測条件

| 項目 | 値 |
|------|------|
| インスタンス | g6e.4xlarge (16 vCPU, NVIDIA L40S) |
| タスク | `Isaac-Velocity-Rough-G1-v0` |
| num_envs | 16384 |
| サンプリング | py-spy 100 Hz, 60 秒, `--idle --subprocesses` |
| サンプル総数 | 5,859 |
| キャプチャ | シェーダウォームアップ後の steady-state |
| 入力 | `results/pyspy/pyspy-4xl-full.svg` (再現可) |

## 1. 上位ホットスポット (self-time top 15)

`self` = その関数自身で消費した CPU 時間 (子関数を呼び出している間は除く)。**leaf-only 集計と違い、親フレーム自身の処理も拾う**。

| self | % | 場所 | 何の処理か |
|---:|---:|---|---|
| **884** | **15.1%** | `_step (api/physics_context/physics_context.py:565)` | **GPU 物理 step 完了待ち** (Isaac Sim C 拡張内 blocking) |
| 381 | 6.5% | `update (rsl_rl/algorithms/ppo.py:394)` | PPO の epoch ループ (`for epoch in range(...)` の周回) |
| **243** | **4.1%** | `synchronize (warp/context.py:6129)` | **Warp の明示 GPU 同期** (raycast 後の cudaStreamSynchronize) |
| 215 | 3.7% | `update (rsl_rl/algorithms/ppo.py:292)` | PPO 収集データ generator の minibatch ループ |
| 208 | 3.6% | `_engine_run_backward (torch/autograd/graph.py:824)` | NN backward (PyTorch autograd) |
| 203 | 3.5% | `sample (torch/distributions/normal.py:74)` | アクション分布 (`Normal`) からのサンプリング |
| **122** | **2.1%** | **`compute (reward_manager.py:149)`** | **報酬集計の親自身** (各 reward 関数を呼ぶ for ループ) |
| 92  | 1.6% | `reset (reward_manager.py:118)` | エピソード終了時の reward 累積初期化 |
| 88  | 1.5% | `reset (reward_manager.py:121)` | 同上、続き |
| 80  | 1.4% | `compute (reward_manager.py:156)` | reward 関数群を呼んだ後の正規化 |
| 79  | 1.3% | `compute (actuators/actuator_pd.py:137)` | PD アクチュエータのトルク計算 |
| 72  | 1.2% | `joint_deviation_l1 (envs/mdp/rewards.py:178)` | reward: 関節偏差 L1 ノルム |
| 70  | 1.2% | `_apply_actuator_model (articulation.py:1812)` | アクチュエータモデル適用 (line 1812) |
| 67  | 1.1% | `_apply_actuator_model (articulation.py:1825)` | 同 line 1825 |
| 62  | 1.1% | `log (on_policy_runner.py:311)` | TensorBoard 等への logging |

### 重要な観察

1. **`_step` 15.1%** = GPU 物理待ち。Isaac Sim/PhysX の C 拡張で blocking なので **「これより速くするには GPU を強くする or 反復削減」** しかない。これが既知 ablation `solver-iter-half` (+31.7%) の作用先。
2. **`synchronize` 4.1%** = `RayCaster` が Warp カーネル投入後に明示的に GPU 同期している箇所。**height_scan ablation で削れる部分** の正体。
3. **`reward_manager.compute` の親 122 (2.1%) + 子 (joint_deviation_l1 など) で計 9.5%**。前回 leaf 集計 (80) では見えなかった。
4. **`_apply_actuator_model` が 1812-1835 行範囲に分散** して各々 1% 前後を消費 = **PD アクチュエータの計算を 1 行ずつ Python で回している**。G1 (37 関節) × num_envs がそのまま CPU 時間。

## 2. CPU 時間配分 (フェーズ別)

`learn` 関数を 4 つのフェーズに分けて集計したもの。1 イテレーション = ロールアウト 24 step + PPO 学習 1 回。

| フェーズ | サンプル | % | 中身 | 律速？ |
|---|---:|---:|---|:---:|
| **ロールアウト収集** (`learn:206`) | 4,435 | **75.7%** | env.step を 24 step × num_envs | **★** |
| 　 物理 (`step:190`) | 923 | 15.8% | うち `_step` 15.1% が GPU 待ち | |
| 　 リセット (`step:221`) | 869 | 14.8% | エピソード終了処理 (`event_manager.apply` 5.6%) | |
| 　 アクション書込 (`step:188`) | 856 | 14.6% | `articulation.write_data_to_sim` (PD 計算) | |
| 　 報酬計算 (`step:208`) | 691 | 11.8% | `reward_manager.compute` (各 reward 関数) | |
| 　 観測収集 (`step:240`) | 560 | 9.6% | `observation_manager.compute` (height_scan 7.5%) | |
| **PPO 学習** (`learn:262`) | 1,039 | **17.7%** | epoch × minibatch × backward | |
| ロールアウト act 詳細 | 234 | 4.0% | `actor_critic.act` / NN forward | |
| Logging | 113 | 1.9% | TensorBoard log | |

### 観察

- **ENV step の 5 大要素 (15-15-15-12-10%) がほぼ等価律速**。1 つを削っても残り 4 つが順に律速になり、効果が出にくい。これが `solver-iter-half` で +31.7% 取った後、頭打ちする構造的理由。
- **PPO 学習で 17.7%** = 1 イテレーション中の **学習側も無視できない大きさ**。`num_mini_batches × num_learning_epochs` を減らすと直接効くゾーン。

## 3. 主要枝の詳細ツリー

`scripts/pyspy_tree.py --max-depth 30 --max-children 5 --min-pct 1.0` の出力。**子関数を 5 つ表示、1.0% 未満は省略**。

```
- 5859 (100.0%) all
  - 5856 (99.9%) <module> (benchmark_rsl_rl.py:258)
    - 5856 (99.9%) wrapper (isaaclab_tasks/utils/hydra.py:104)
      - 5856 (99.9%) decorated_main (hydra/main.py:94)
        - ... (hydra bootstrap, 全 5856 通過)
        - 5856 (99.9%) main (benchmark_rsl_rl.py:216)
          - 4435 (75.7%) learn (rsl_rl/runners/on_policy_runner.py:206)  ← ロールアウト
          | - 4433 (75.7%) step (gymnasium core wrapper chain)
          |   - 923 (15.8%) ENV step → physics
          |   |   - 922 (15.7%) sim_context.step
          |   |     - 884 (15.1%) _step (physics_context:565)  [leaf, GPU 待ち]
          |   - 869 (14.8%) ENV step → reset_idx (エピソード終了)
          |   |   - 381 (6.5%) _reset_idx:364
          |   |   | - 326 (5.6%) event_manager.apply (domain randomization)
          |   |   |   - 61 (1.0%) apply_external_force_torque
          |   |   - 208 (3.6%) _reset_idx:377
          |   |   | - 92 (1.6%) reward_manager.reset:118  [leaf]
          |   |   | - 88 (1.5%) reward_manager.reset:121  [leaf]
          |   |   - 78  (1.3%) curriculum_manager.compute
          |   |   - 76  (1.3%) interactive_scene.reset
          |   |   - 72  (1.2%) command_manager.reset
          |   - 856 (14.6%) ENV step → write_data_to_sim (アクション → Sim)
          |   |   - 852 (14.5%) interactive_scene.write_data_to_sim
          |   |     - 779 (13.3%) articulation.write_data_to_sim
          |   |       - 186 (3.2%) _apply_actuator_model:1818
          |   |       | - 79 (1.3%) actuator_pd.compute:137  [leaf]
          |   |       | - 61 (1.0%) actuator_pd.compute:139
          |   |       - 83  (1.4%) _apply_actuator_model:1820
          |   |       - 70  (1.2%) _apply_actuator_model:1812  [leaf]
          |   |       - 67  (1.1%) _apply_actuator_model:1825  [leaf]
          |   |       - 61  (1.0%) _apply_actuator_model:1813  [leaf]
          |   - 691 (11.8%) ENV step → reward
          |   |   - 556 (9.5%) reward_manager.compute:149
          |   |     - 72 (1.2%) joint_deviation_l1  [leaf]
          |   |     - 61 (1.0%) feet_slide
          |   |   - 80  (1.4%) reward_manager.compute:156  [leaf]
          |   - 560 (9.6%) ENV step → observation
          |       - 559 (9.5%) observation_manager.compute
          |         - 518 (8.8%) compute_group
          |           - 442 (7.5%) height_scan
          |             - 442 (7.5%) ray_caster.data
          |               - 425 (7.3%) sensor_base._update_outdated_buffers
          |                 - 308 (5.3%) ray_caster._update_buffers_impl
          |                   - 248 (4.2%) raycast_mesh
          |                     - 243 (4.1%) warp.synchronize  [leaf, GPU 同期]
          - 1039 (17.7%) learn (on_policy_runner.py:262)  ← PPO 学習
          | - 381 (6.5%) update (ppo.py:394)  [leaf, epoch loop]
          | - 221 (3.8%) update (ppo.py:260)
          | | - 203 (3.5%) actor_critic.act
          | |   - 203 (3.5%) Normal.sample  [leaf]
          | - 215 (3.7%) update (ppo.py:292)  [leaf, minibatch loop]
          | - 208 (3.6%) update (ppo.py:375)
          |   - 208 (3.6%) Tensor.backward
          |     - 208 (3.6%) autograd.backward
          |       - 208 (3.6%) _engine_run_backward  [leaf]
          - 234 (4.0%) learn (on_policy_runner.py:204)  ← rollout 内 act 詳細
          | - 208 (3.6%) act (ppo.py:142)
          |   - 150 (2.6%) actor_critic.act:121
          |     - 76 (1.3%) update_distribution
          |       - 74 (1.3%) MLP forward (Linear x N + ELU x N)
          - 113 (1.9%) learn (on_policy_runner.py:270)  ← logging
            - 62 (1.1%) log (on_policy_runner.py:311)  [leaf]
```

完全版は `results/pyspy/pyspy-4xl-full.tree.md` を参照。

## 4. ボトルネックと打ち手の対応

### 構造的に削れない部分 (GPU 物理に固有)

| 項目 | % | 既存 ablation での削減 |
|---|---:|---|
| `_step` (GPU 物理待ち) | 15.1% | `solver-iter-half` で実測 +31.7% |
| `warp.synchronize` (raycast) | 4.1% | `height-scan-halffreq` で +1.2% |

### 構造的に削れる可能性がある部分 (Python 実装)

| 項目 | % | 攻め方 (実装変更要) |
|---|---:|---|
| `event_manager.apply` (domain randomization) | 5.6% | `EventCfg.interval` を伸ばす (毎ステップ → 数ステップ毎) |
| `_apply_actuator_model` 群 | 計 ~5% | アクチュエータモデルを CUDA カーネル化 (Isaac Lab 上流改修) |
| `reward_manager.compute` (報酬集計) | 9.5% | reward 関数を Python から JIT/CUDA に書換え |
| `height_scan` 全体 | 7.5% | `update_period` 倍化 / 解像度削減 (`height-scan-lowres` で +2.3%) |
| `_reset_idx` 各種 | 計 ~14.8% | エピソード長を伸ばす (`max_episode_length` 増) |

### PPO 側 (1 学習イテレーションで)

| 項目 | % | 攻め方 |
|---|---:|---|
| epoch loop (`update:394`) | 6.5% | `num_learning_epochs` 5 → 3 |
| minibatch loop (`update:292`) | 3.7% | `num_mini_batches` 4 → 2 |
| backward (`_engine_run_backward`) | 3.6% | NN サイズ縮小、AMP, `torch.compile` |

## 5. 顧客向け 1 行まとめ

> **GPU 待ち (`_step`) は CPU 時間の 15% だけ。残り 85% は Isaac Lab の Python メインスレッドで「アクション書込・報酬・観測・リセット」の 4 大処理が直列に走っており、これがハイクロックなコンシューマ CPU (i9 6 GHz) が EC2 サーバ CPU (EPYC 3.7 GHz) より 1.5 倍速い構造的理由。num_envs を増やしてもこの 5 大処理が比例して増えるだけで、単 GPU の天井は今の `combined +112%` 付近。本格的な高速化はマルチノード DDP (HyperPod) でプロセス数を増やす方向。**

## 6. 再現コマンド

```bash
# EC2 上で
cd ~/IsaacLab && conda activate env_isaaclab
cd ~/zenmai-lab/experiments/isaac-lab-bench

# プロファイル取得 (60 秒、SVG + speedscope JSON + tree.md 自動生成)
bash scripts/profile_pyspy.sh Isaac-Velocity-Rough-G1-v0 g1-pyspy "42" 100 16384

# 出力:
#   results/pyspy/g1-pyspy/run-seed42-*/pyspy.svg            ← ブラウザで開く
#   results/pyspy/g1-pyspy/run-seed42-*/pyspy.speedscope.json ← speedscope.app へ drop
#   results/pyspy/g1-pyspy/run-seed42-*/pyspy.tree.md        ← Markdown 解析 (これ)

# 任意の閾値で解析し直す
python3 scripts/pyspy_tree.py path/to/pyspy.svg --format md --min-pct 1.0 --max-depth 30 --max-children 5
```

## 7. 検証で確認できなかった点 (誠実性確保)

- 計測は **1 ラン 60 秒の py-spy サンプリング** であり、シードによる分散は計測していない。各関数の % はおおよそ ±1% の誤差を想定。
- `--idle` フラグを付けているため `_step` の GPU 待ちも CPU サンプルとして計上されている。**「CPU が忙しい」のではなく「メインスレッドが GPU を待ってブロックしている」状態を含む**。
- py-spy はメインスレッドのみサンプリングしている (`Sampling threads: 1`)。Isaac Sim の内部ヘルパースレッドは見えていない。
- 行番号は調査時点 (Isaac Lab v2.2 / Isaac Sim 5.1) のものであり、上流バージョンが上がるとずれる。
