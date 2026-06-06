# `isaac-lab-bench`

Isaac Lab 公式ベンチマーク (`scripts/benchmarks/benchmark_rsl_rl.py`) をリモートの EC2 上で複数シード走らせ、結果と GPU 使用率ログを手元に回収するための実験。

EC2 の用意は別作業 (CDK / IsaacAutomator / 手動など何でもよい)。本実験は **既に Isaac Lab v2.x が `~/IsaacLab` に入っていて conda env `env_isaaclab` が使える** 状態の EC2 を前提にする。

## 前提

- ローカル: AWS CLI v2 認証済み、`jq`、Python 3.10+
- 対象 EC2: SSM 接続可能 (Session Manager IAM 付与済み)、GPU + ドライバ OK、`nvidia-smi dmon` が動く、`/home/ubuntu/IsaacLab` あり

> 注: NVIDIA Marketplace の Isaac Sim AMI には `set -e` 下で `tabs(1)` が `TERM=unknown` で落ちる問題がある。`run_bench.sh` 側で `export TERM=xterm` を強制しているので素の SSM 実行でも壊れない。

## クイックスタート

```bash
# 1. clone
git clone git@github.com:littlemex/zenmai-lab.git
cd zenmai-lab/experiments/isaac-lab-bench

# 2. ターゲットを設定
cp push.env.sample push.env
$EDITOR push.env       # INSTANCE_ID と AWS_REGION を埋める

# 3. ベンチを走らせる (プロファイルから選ぶ)
bash orchestrate.sh g1            # G1 ロコモーション 5 シード × 100 iter
bash orchestrate.sh cartpole      # Cartpole 5 シード × 100 iter
bash orchestrate.sh g1-quick      # G1 1 シード × 50 iter (ハマり所確認用)
bash orchestrate.sh g1-num-envs   # num_envs スイープ {4096, 8192, 16384}
```

実行中、ローカルには 60 秒間隔でリモートの進捗が出力される。完了すると `results/<profile>-<timestamp>/` に成果が落ちる。

## 出力

```
results/g1-20260606-150000/
├── bench-g1-n4096-20260606-150000/
│   ├── summary.csv                                  ← seed × FPS の集計
│   ├── run-seed42-…/bench.log                       ← benchmark_rsl_rl.py の生 stdout
│   ├── run-seed42-…/nvidia-smi-dmon.log             ← 1Hz GPU サンプル
│   └── …(seed ごとに 1 ディレクトリ)
└── …
```

`orchestrate.sh` の最後に `dmon_stats.py --format md` の出力 (SM/MEM/PWR の avg/p95/max を Markdown 表化) が表示される。あとから集計しなおすには:

```bash
python3 remote/dmon_stats.py --format md \
  results/g1-20260606-150000/*/run-seed*/nvidia-smi-dmon.log
```

## ファイル構成

| パス | 役割 |
|---|---|
| `orchestrate.sh` | ローカルから SSM 経由でリモート実行を駆動 |
| `push.env.sample` | `INSTANCE_ID` 等の設定テンプレ |
| `remote/run_bench.sh` | EC2 上で動く本体。`benchmark_rsl_rl.py` をループ実行 |
| `remote/dmon_stats.py` | dmon ログ集計。ローカル側の最後の出力にも使われる |

`remote/` の中身は orchestrate.sh が SSM で逐次転送するため、EC2 に手動配置する必要はない。

## プロファイルを足す

`orchestrate.sh` 内の `case "$PROFILE"` に追加するだけで増やせる。`SWEEP=()` を複数値にすると同一実行内で `num_envs` を変えて連続計測できる (例: `g1-num-envs`)。

## トラブル

| 症状 | 対処 |
|---|---|
| `pgrep -c=0` のまま summary.csv が空 | `bench.log` を見る (`/home/ubuntu/<tag>.log`)。Isaac Sim の起動時シェーダーキャッシュ生成で初回 50 秒近くかかる |
| `'unknown': I need something more specific.` | `TERM` が伝わっていない。`run_bench.sh` 内で export しているので最新版に同期 |
| SSM が `InvalidInstanceId` | インスタンスが stopped か SSM Agent 停止。`aws ec2 describe-instances` で状態確認 |
| `summary.csv` の `mean_total_fps` が `NA` | `bench.log` に `Mean Total FPS:` 行が無い = ベンチが途中で死んだ。多くは OOM。`dmesg \| grep -i oom` |
