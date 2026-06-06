# Isaac Lab パフォーマンス改善余地リスト

**前提**: g6e.2xl / 4xl の実測で「num_envs を上げると GPU メモリ帯域が飽和、CPU は遊ぶ」ことが分かった。インスタンスを変えずに試せるレバーを網羅。

**調査範囲**: 7 レンズ（Isaac Lab config / RL framework / PhysX / env design / system config / training loop / multi-GPU）× 146 提案 → 各案を adversarial verifier に通して 91 件採用。

このノートは「**何をどこで触れば fps が上がる可能性があるか**」のチェックリスト。具体的な数値は実測しないと分からない。

---

## 1. 即試せる Quick Wins（trivial 〜 small effort）

### 1.1 collect_time vs learn_time の比率を確認する

最初にこれを見る。これがないと以降のチューニング優先順位が決まらない。コストゼロ。

- **適用方法**: TensorBoard で `Perf/collection_time` と `Perf/learning_time`、または stdout の `Collection time:` / `Learning time:` を見る
- **判断**:
  - `collect_time >> learn_time` → 環境側 (PhysX / raycaster / contact sensor) を最適化
  - `learn_time` が全体の 20% 以上 → PPO ハイパラ (mini_batches, epochs) を見直す

### 1.2 GPU バッファオーバーフローのログ確認

`velocity_env_cfg.py` は 4096 env 向けにバッファサイズが調整されているが、num_envs を 16384 まで上げると **silent overflow** で nan / 異常な reward スパイクが起きる可能性。

- **確認**: `grep -r 'foundLostPairs\|PhysX error\|buffer overflow' /path/to/training_logs/`
- **対処**: ヒットしたら `LocomotionVelocityRoughEnvCfg.__post_init__` で以下を増やす
  ```python
  self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
  self.sim.physx.gpu_max_rigid_contact_count = 2**24
  ```

### 1.3 誤設定フラグの一括確認

これら 5 つはデフォルトが最適だが、誤って変更されていると **数十 % 落ちる**。確認コストはゼロ。

| フラグ | 期待値 | 誤設定時の影響 |
|---|---|---|
| `--enable_cameras` | 渡さない | RTX 拡張がロードされて大幅低下 |
| `SimulationCfg.use_fabric` | `True` | False で 20-40% 低下 (高 num_envs) |
| `physx.enable_enhanced_determinism` | `False` | True でスループット明示的に犠牲 |
| `physx.enable_ccd` | `False` | True で broad-phase 2 周、10-20% 低下 |
| `physx.enable_stabilization` | `False` | True で接触処理が増える |

### 1.4 TF32 の有効化（カスタムスクリプト使用時のみ）

公式 `scripts/reinforcement_learning/rsl_rl/train.py` には `torch.backends.cuda.matmul.allow_tf32 = True` がハードコード済み。**カスタムスクリプトを書いている場合は要確認**。

```python
# AppLauncher 初期化前
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 1.5 `empirical_normalization=True` への変更

G1 Rough のデフォルトは `False`。観測空間が多様なほど Welford オンライン正規化で学習収束が早くなる可能性。**fps への直接影響は無視できる**が時間短縮効果はあり得る。

- **編集**: `source/isaaclab_tasks/.../config/g1/agents/rsl_rl_ppo_cfg.py` の `G1RoughPPORunnerCfg`
  ```python
  empirical_normalization = True
  ```

### 1.6 地形キャッシュ `use_cache=True`

`ROUGH_TERRAINS_CFG` のデフォルトは `False`。再起動のたびに 200 タイル分のハイトフィールドメッシュを毎回生成している。

- **適用**:
  ```python
  from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
  self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.replace(use_cache=True)
  ```
- **効果**: 起動時間の短縮（学習中は影響なし）

---

## 2. 中効果 / 中労力

### 2.1 contact sensor の `prim_path` を必要なボディのみに絞る

現在の `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")` は G1 MINIMAL の **全 23 リンク**に PhysxContactReportAPI を適用している。実際に使うのは足首と torso のみ。

- **適用**: `config/g1/rough_env_cfg.py` の `__post_init__`
  ```python
  self.scene.contact_forces = ContactSensorCfg(
      prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",  # 足首だけ
      history_length=3,
      track_air_time=True,
  )
  ```
- **効果**: 接触処理コストが 23 体 → 3 体で激減。**メモリ帯域節約に直接効く**
- **要確認**: 顧客のカスタム reward が他のボディの contact_forces を参照していないか

### 2.2 height scan の `update_period` を 2 倍に伸ばす

すでに `decimation * sim.dt = 0.02s` (50 Hz) で動いているが、`0.04s` (25 Hz) に伸ばせば Warp raycast カーネルの実行頻度が半減する。

- **適用**: `config/g1/rough_env_cfg.py` の `__post_init__`
  ```python
  if self.scene.height_scanner is not None:
      self.scene.height_scanner.update_period = 2 * self.decimation * self.sim.dt
  ```
- **要確認**: 観測の遅延が学習に許容できるか

### 2.3 G1 articulation solver iteration の削減

`G1_CFG` は PhysX 推奨値の 2-4 倍に設定されている (`pos=8, vel=4`)。AnymalC（同種の脚式）は `pos=4, vel=0` で動作するので半減候補。

- **適用**: `config/g1/rough_env_cfg.py` の `__post_init__`
  ```python
  self.scene.robot = G1_MINIMAL_CFG.replace(
      prim_path="{ENV_REGEX_NS}/Robot",
      spawn=G1_MINIMAL_CFG.spawn.replace(
          articulation_props=ArticulationRootPropertiesCfg(
              solver_position_iteration_count=4,  # 8→4
              solver_velocity_iteration_count=1,  # 4→1
          ),
      ),
  )
  ```
- **効果**: solver スループット 10-20% 向上の可能性
- **要確認**: 学習収束（reward 曲線）が安定するか必ず ablation する

### 2.4 num_learning_epochs と num_mini_batches の調整

**`learn_time` が全体の 20% 以上を占める場合のみ**有効。

- **適用**: `config/g1/agents/rsl_rl_ppo_cfg.py`
  ```python
  num_mini_batches = 2  # 4→2 で mini-batch サイズが 2x、GPU の matmul が効率化
  ```
- **要確認**: PPO の clip 比率が安定しているか、KL divergence が暴れないか

### 2.5 height scan の解像度 / サイズを削減

`GridPattern(resolution=0.1, size=[1.6, 1.0])` は **160 光線/env**。16384 env だと **毎ステップ 262 万光線**。

- **削減案**: `resolution=0.15` または `size=[1.0, 0.6]` で 80 光線/env 程度に
- **適用**: `config/g1/rough_env_cfg.py`
  ```python
  from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
  if self.scene.height_scanner is not None:
      self.scene.height_scanner.pattern_cfg = GridPatternCfg(
          resolution=0.15, size=[1.0, 0.6]
      )
  ```
- **効果**: Warp raycast カーネルの計算量が約 1/2

### 2.6 height scan の完全削除（フラット地形）

`G1FlatEnvCfg` は既に `scene.height_scanner=None` + `terrain_type='plane'`。**rough 地形での汎化が不要**ならこれが最強のスループット最大化。

- **適用**: `G1FlatEnvCfg` を使うか、rough cfg で
  ```python
  self.scene.height_scanner = None
  self.scene.terrain.terrain_type = "plane"
  self.scene.terrain.terrain_generator = None
  ```
- **要確認**: 顧客のユースケース（実機で凹凸地形を歩かせる必要があるか）

### 2.7 `max_iterations` の早期停止

G1RoughPPORunnerCfg のデフォルトは `max_iterations=3000`。**TensorBoard の reward 曲線が 2000 で収束していたら、残り 1000 は無駄**（1/3 短縮）。

- **判定**: `Episode/rew_mean` の変化が直近 1000 iter で 5% 以内 → 収束
- **適用**: `--max_iterations 2000` または cfg の `max_iterations = 2000`

---

## 3. 顧客に確認したいこと

1. **TensorBoard の `Perf/collection_time` と `Perf/learning_time`** の値（または比率）を教えてください。最重要。これがないと優先順位が決まらない。
2. **学習スクリプト**は Isaac Lab 公式の `train.py` ですか、カスタムですか？カスタムなら `torch.backends.cuda.matmul.allow_tf32` が True になっていますか？
3. **contact sensor の用途**: `undesired_contacts` 報酬や他のカスタム報酬で「足首・torso 以外」のボディの contact_forces を参照していますか？
4. **rough 地形の汎化性能** は学習要件ですか？それともスループット最大化を優先できますか？
5. **現在の num_envs と GPU VRAM 使用量** （`nvidia-smi --query-gpu=memory.used --format=csv`）を教えてください。
6. **エラーログ**: `PhysX error:` や `foundLostPairsCapacity` などのメッセージ、NaN、reward の異常スパイクは出ていませんか？
7. **学習の収束イテレーション数**: TensorBoard の `Episode/rew_mean` がどのイテレーションあたりで頭打ちになっていますか？

---

## 4. 投機的（要追加検証）

- **`decimation=4→8`（policy 25 Hz）**: 物理ステップ数半減でメモリ帯域節約だが、G1 のような高 DOF 人型でバランス維持できるか不明。報酬設計の全面再調整が要る
- **g6e.12xlarge (4x L40S) へのスケールアウト**: 各 GPU が独立 batch を処理して帯域ボトルネックも分散。`--distributed` で公式サポート、実効 2-3x の見込み。コスト 4.7×
- **rsl_rl v5.x へのアップグレードと `torch_compile_mode`**: Isaac Lab v2.2.0 が使う rsl_rl v2.3.3 にはまだ無い。アップグレードできれば learn_time -10-20% の見込み
- **PhysX `solver_type=0` (PGS) + iteration 削減**: TGS より 1 反復あたりのコスト低い。G1 のような多 DOF で品質維持できるかは要検証
- **`contact_forces.history_length` を 3 → 1**: 削減量約 11MB、L40S 864 GB/s に対し 0.4% 未満で誤差レベル。`prim_path` 絞り込みの方が桁違いに効くので単体では低優先

---

## 5. 却下されたアイデア（理由付き）

- **`--headless` フラグ**: 顧客はすでに headless 実行中、適用余地なし
- **`render_interval` の調整**: `velocity_env_cfg.py` で既に `decimation` に合わせ済み + headless では render() が no-op、変更しても何も変わらない
- **`cudnn.benchmark`**: RSL-RL の ActorCritic は純 MLP (Linear のみ)、benchmark は Conv 専用機能で効果ゼロ
- **TF32 (公式 train.py)**: ハードコード済みで既に有効
- **`gpu_max_num_partitions`**: articulation 間の並列はパーティション数に依存しない（各ロボットが独立ツリー）。メモリ帯域への作用機序もなし
- **`dt=0.01, decimation=2` への変更**: G1 で 100 Hz 物理が維持できる根拠がない。sim-to-real ギャップ拡大リスク高
- **`contact_sensor.history_length` 削減単体**: 削減量が L40S 帯域の 0.4%、誤差レベル
- **`InteractiveSceneCfg.clone_in_fabric=True`**: v2.2.0 では PR #5437 / #5580 により no-op と確定
- **ネットワーク縮小 `[512,256,128] → [256,128,128]`**: NN パラメータ 1.5 MB は L2 キャッシュに収まる、メモリ帯域とは無関係。学習品質低下リスクの方が大きい
- **ECC / PCIe / GPU clock lock / SMT 無効化 / NUMA ピニング**: AWS EC2 では設定不可、または GPU 帯域ボトルネックと無関係
- **NCCL / EFA / マルチノード設定**: シングル GPU 環境では発生しない、適用条件を満たさない
- **`cudnn.deterministic`**: 公式 train.py にハードコード済み

---

## 進め方の提案

1. **まず顧客に §3 の質問 1-2 を投げる**（特に collect/learn 比率）
2. 答えに応じて優先順位を決め、§1 の Quick Wins を 30 分で全部チェック
3. 効果のありそうな §2 の項目を 1 つずつ ablation
4. 最後に余地が無ければ g6e.12xlarge / マルチノードを検討（§4）
