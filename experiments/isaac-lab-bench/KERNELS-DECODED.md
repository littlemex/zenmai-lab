# Top-10 GPU カーネルの正体 (一次ソース裏付け)

`scripts/profile_nsys.sh` で取った `kernels.csv` の上位 10 カーネルが**それぞれどのライブラリのどのコードに由来するか**を独立並列調査 + adversarial verify で裏取りした結果。

検証は次の 3 段階で実施した：

1. 各カーネルを別々のエージェントが Web 検索 + GitHub 一次ソースから調査
2. 別のエージェントが「URL は本当に存在するか」「機能説明は事実か」を独立に検証
3. 検証結果が食い違った点は「不確定事項」として明記

## EC2 上 grep で独立に確認した 3 件

これらは EC2 (i-03bfb5f08ce61d4b5) の実機ファイルで直接確認できた。

| 主張 | 確認方法 | 結果 |
|------|---------|------|
| `rsl_rl` の ActorCritic は ELU 活性化がデフォルト | `grep "activation" /home/ubuntu/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/rsl_rl/modules/actor_critic.py` | line 25: `activation="elu"` ← 確定 |
| `RayCaster` は Warp を使用 | `grep "import warp" /home/ubuntu/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/` | `ray_caster.py:16: import warp as wp` ← 確定 |
| `solver_position_iteration_count` の存在 | `grep -A 1 "solver_position_iteration_count" /home/ubuntu/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py` | line 28-29 で定義 + docstring ← 確定 |

## カーネル別調査結果

### 1. `artiSolveInternalConstraintsTGS1T` — 39.1%

| 項目 | 内容 |
|------|------|
| **起源** | NVIDIA PhysX (NVIDIA-Omniverse/PhysX) |
| **検証** | **確認済み (高信頼)** |
| **ソース** | [`physx/source/gpuarticulation/src/CUDA/internalConstraints2.cu` L2841](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpuarticulation/src/CUDA/internalConstraints2.cu) |
| **役割** | TGS (Temporal Gauss-Seidel) ソルバーの各サブステップで、GPU 上の全アーティキュレーション (ロボット) に対して **関節ドライブ・摩擦・位置制限・速度制限の拘束インパルス**を適用する。1 スレッド = 1 アーティキュレーションで運動連鎖を前向き・後向きに走査して空間インパルスをリンク全体に伝播。 |
| **呼ばれる場所** | `PxgArticulationCore::propagateRigidBodyImpulsesAndSolveInternalConstraints()` (`PxgArticulationCore.cpp` L656) → `PxgTGSCudaSolverCore::solveIsland()` の **位置反復ループと速度反復ループの両方**から呼ばれる |
| **頻度の支配パラメータ** | **`solver_position_iteration_count` + `solver_velocity_iteration_count`** (Isaac Lab default 4 + 0、G1 デフォルトは 8 + 4) × `num_envs` |
| **半減で +31.7% の根拠** | 物理サブステップあたりの呼び出し数が `(8+4) → (4+1)` で 5/12 ≈ 42% 減 → このカーネル単体で見た時間も同比例して減る → 全体時間 39.1% × 0.58 ≈ -23 ポイント減 → 残り時間ベースで +30% 程度の fps 改善は妥当 |

### 2. `stepArticulation1TTGS` — 8.9%

| 項目 | 内容 |
|------|------|
| **起源** | NVIDIA PhysX |
| **検証** | **確認済み (高信頼)** |
| **ソース** | [`physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu` L3873](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu) |
| **役割** | TGS ソルバーの各位置反復サブステップの末尾で、全アーティキュレーション・リンクのポーズを `stepDt = dt / numPositionIterations` で前進積分する。ルートリンクの SE(3) 積分後、子リンクへポーズを伝播。`mDeltaMotion` と `mPosMotionVelocity` アキュムレーターを更新。 |
| **頻度の支配パラメータ** | **`solver_position_iteration_count`** (`solver_velocity_iteration_count` には依存しない) × `num_envs` |
| **`1T` 接尾辞** | コード内コメント L1102: "running single threaded, hence the name 1T" — warp 内で 1 スレッドが 1 アーティキュレーションを順次処理する実装 |

### 3. `raycast_mesh_kernel_<hash>_cuda_kernel_forward` — 3.3%

| 項目 | 内容 |
|------|------|
| **起源** | **NVIDIA Warp JIT** (Isaac Lab で生成) |
| **検証** | **確認済み (高信頼)** |
| **ソース** | [`source/isaaclab/isaaclab/utils/warp/kernels.py` L13](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/warp/kernels.py) |
| **役割** | Isaac Lab の `RayCaster` センサーが毎物理ステップに Warp JIT で GPU 並列実行するレイキャストカーネル。地形メッシュの BVH に対して **Möller-Trumbore 三角形交差判定**を行い、ロボットの**高さマップ観測値** (`ray_hits_w`) を生成する。 |
| **`_forward` 接尾辞** | Warp の autograd 規約 (forward / backward 区別)。ここは推論経路 = 微分グラフ外。 |
| **頻度の支配パラメータ** | センサー更新周期 (`RayCasterCfg.update_period`) × `num_envs × レイ数 × BVH 深さ` |
| **`solver_iter` とは独立** | これが「solver-iter-half で削れない 3.3% 部分」の正体 |
| **EC2 上の確認** | `ray_caster.py:16: import warp as wp` で実機確認済 |

### 4. `at::native::index_elementwise_kernel` — 3.2%

| 項目 | 内容 |
|------|------|
| **起源** | PyTorch ATen |
| **検証** | **部分確認 (中信頼)** |
| **ソース候補** | [`pytorch/aten/src/ATen/native/cuda/IndexKernel.cu` L28](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/IndexKernel.cu) |
| **役割** | テンソルへの高度なインデックス操作 (`index_fill_`、`index_copy_`、`flip`、`take`/`put_` 等) を並列実行する CUDA カーネル。RL では **環境リセット時のバッファクリア**や **PPO ミニバッチサンプリング**で使われる可能性が高い。 |
| **頻度の支配パラメータ** | エピソード終了率 (環境リセットの頻度) と PPO の `num_mini_batches`。`solver_iter` とは**独立**。 |
| **不確定** | 単一インデックス・アライメント済みギャザーは別の `vectorized_gather_kernel_launch` (fast_gather パス) でバイパスされる場合がある。「全ての高度なインデックスがこれを経由する」とは断定できない。 |

### 5. `updateBodiesLaunch_Part2` — 2.4%

| 項目 | 内容 |
|------|------|
| **起源** | NVIDIA PhysX |
| **検証** | **確認済み (高信頼)** |
| **ソース** | [`physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu` L3077](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu) |
| **役割** | PhysX アーティキュレーションのソルバー完了後、各リンクの**ワールド姿勢 (`body2World`)・速度・加速度・関節力**を一括 GPU 書き戻しする第 2 パスカーネル。Isaac Lab が `net_contact_forces` / `incoming_joint_force` として公開するデータもここで生成。スリープ判定 (`sleepCheck1T`) も担当。 |
| **頻度の支配パラメータ** | **物理サブステップあたり 1 回**。ソルバー反復回数に依存しない。`num_envs` に比例。 |

### 6. `convexTrimeshNarrowphase` — 2.4%

| 項目 | 内容 |
|------|------|
| **起源** | NVIDIA PhysX |
| **検証** | **確認済み (高信頼)** |
| **ソース** | [`physx/source/gpunarrowphase/src/CUDA/convexMesh.cu` L623](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpunarrowphase/src/CUDA/convexMesh.cu) |
| **役割** | PhysX GPU ナローフェーズの**凸形状対トライアングルメッシュ衝突検出**カーネル。ブロードフェーズ後・ソルバー前に実行。1 ワープが 1 (凸形状, 三角形) ペアを処理。**ロボット足部対地形の接触生成**の中核。 |
| **頻度の支配パラメータ** | 物理サブステップあたり 1 回 (ソルバー反復回数に依存しない)。`num_envs × アクティブ接触ペア数` に比例。 |
| **`contact-scope-ankle` で +0.4% しか効かなかった理由** | このカーネル自体は接触ペアの内容を全件計算していて、`ContactSensor.prim_path` の絞り込みは**観測収集の集計対象を絞るだけ**で、本カーネルの実行には影響しないため。 |

### 7. `computeUnconstrainedSpatialInertiaLaunchPartial1T` — 2.4%

| 項目 | 内容 |
|------|------|
| **起源** | NVIDIA PhysX |
| **検証** | **部分確認 (中信頼)** |
| **ソース** | [`physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu` L1312](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu) |
| **役割** | PhysX GPU **Featherstone 順動力学**の第 1 空間慣性パスカーネル。各リンクの孤立空間慣性テンソル・コリオリ項・ダンピング力・バイアス力ベクトルを計算。後続の伝播カーネルへの入力を生成。 |
| **頻度の支配パラメータ** | 物理サブステップあたり 1 回 (ソルバー反復回数に依存しない)。`num_envs` および 1 アーティキュレーションあたりのリンク数に比例。 |
| **不確定** | 報告された行番号と実際にズレ。実装モデルの説明が一部不正確 (1 スレッドが 1 アーティキュレーション全体を順次処理するという形式)。コア機能 (ABA 第 1 パス、孤立慣性計算) は正確。 |

### 8. `at::native::index_put_kernel_impl` — 2.1%

| 項目 | 内容 |
|------|------|
| **起源** | PyTorch ATen |
| **検証** | **確認済み (高信頼、注記あり)** |
| **ソース** | [`pytorch/aten/src/ATen/native/cuda/IndexKernel.cu` L189](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/IndexKernel.cu) |
| **役割** | 整数インデックス配列によるテンソルへの**散乱書き込み** (`tensor[index_array] = values`) を実装するホストテンプレート関数。RL の環境リセット時のバッファ更新やアクション・観測テンソルの部分更新で使われる。 |
| **注記** | これ自体は `__global__` CUDA カーネルではなく C++ ホスト側テンプレート関数。実際に GPU で動くのは `index_elementwise_kernel`。nsys が「launcher」名でレポートしている。 |

### 9. `at::native::elu_backward_kernel` — 2.0%

| 項目 | 内容 |
|------|------|
| **起源** | PyTorch ATen |
| **検証** | **部分確認 (中信頼)** |
| **ソース** | [`pytorch/aten/src/ATen/native/cuda/ActivationEluKernel.cu` L48](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationEluKernel.cu) |
| **役割** | rsl_rl PPO 学習時に **ELU 活性化関数の逆伝播勾配**を GPU で要素ごとに計算。Actor / Critic MLP の学習バックパス中に各 ELU 層で呼び出される。 |
| **頻度の支配パラメータ** | PPO の `num_learning_epochs × num_mini_batches`。物理ソルバーと**独立**。`num_envs` はバッチサイズに影響するが起動回数には影響しない。 |
| **EC2 上の確認** | `actor_critic.py:25 activation="elu"` が rsl_rl のデフォルトであることを実機で確認済 |
| **不確定** | 行番号の引用が不正確 (報告 L53-76, 実際 L48-81)。機能説明は正確。 |

### 10. `computeUnconstrainedAccelerationsLaunch1T` — 1.9%

| 項目 | 内容 |
|------|------|
| **起源** | NVIDIA PhysX |
| **検証** | **確認済み (高信頼、行番号注記あり)** |
| **ソース** | [`physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu`](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu) |
| **役割** | PhysX GPU **Featherstone 順動力学**パイプラインの最終前処理カーネル。空間慣性伝播後・制約ソルバー前に `computeLinkAcceleration()` と `computeJointTransmittedFrictionForce()` を呼び出して**各リンクの自由加速度** (制約なしの場合の加速度) を計算する。制約ソルバーの出発点。 |
| **頻度の支配パラメータ** | 物理サブステップあたり 1 回 (ソルバー反復回数に依存しない)。`num_envs` に比例。 |

## カテゴリ別累積 %

| 起源 | カーネル群 | 累積 % |
|------|-----------|------:|
| **NVIDIA PhysX (articulation 系)** | 1, 2, 5, 7, 10 | **54.7%** |
| **NVIDIA PhysX (narrowphase)** | 6 | 2.4% |
| **NVIDIA Warp (Isaac Lab raycast)** | 3 | 3.3% |
| **PyTorch (RL update / indexing)** | 4, 8, 9 | **7.3%** |
| (top10 の合計) | | 67.7% |

残り 32.3% は kernels.csv の 11 位以下に分散。

## 検証で確認できなかった点 (誠実性確保のための明記)

- 各カーネルの行番号は調査時点の HEAD に対して報告されており、PhysX/Isaac Lab/PyTorch の最新バージョンでは数行ずれることがある。**機能の特定には影響しないが、行番号への直接 jump はソースタグを合わせる必要あり**。
- `index_elementwise_kernel` は本当に高度インデックスから呼ばれているか、または fast_gather でバイパスされているかは、一次ソースだけからは断定できない。RL 上の発生条件は推測。
- `index_put_kernel_impl` は名前だけ見ると `__global__` のように読めるが、実際は C++ ホスト関数。nsys が記録している名前と CUDA カーネル本体は別物 (実体は `index_elementwise_kernel`)。
- `at::native::elu_backward_kernel` の行番号引用は不正確だったため、原典に当たって裏取りを推奨。

## 顧客向け 1 行サマリ

> **GPU 時間の 55% を PhysX articulation の 5 カーネル群が占め、そのうち最大 (39%) は `solver_position_iteration_count` と `solver_velocity_iteration_count` に直接比例する `artiSolveInternalConstraintsTGS1T`。** これが `solver-iter-half` で +31.7% を取れた理由。残りは Warp raycast 3.3%、PyTorch RL update 7.3%、PhysX narrowphase 2.4%。
