# `isaac-lab-bench`

Isaac Lab 公式ベンチマーク (`scripts/benchmarks/benchmark_rsl_rl.py`) を複数シードで走らせ、結果と GPU 使用率ログを集める実験。

EC2 の構築は別作業 (CDK / IsaacAutomator / 手動など何でもよい)。本実験は **`~/IsaacLab` と conda env `env_isaaclab` が使える状態の EC2** を前提にする。

## 2 つのモード

| | DCV モード | SSM モード |
|---|---|---|
| 操作場所 | EC2 上の DCV / SSH ターミナル | ローカル (Mac / Linux) |
| 入口 | `bash scripts/run_bench.sh ...` | `bash orchestrate.sh <profile>` |
| 必要な設定 | `bench.env` | `ssm.env` |
| 必要な依存 | python3, nvidia-smi (EC2 標準) | `aws` CLI v2 + `jq` + python3 |
| 結果の置き場 | EC2 上の `$DEST_BASE` | ローカル `results/<profile>-<stamp>/` |

迷ったら **DCV モード** が単純。SSM モードはラップトップから複数インスタンスを横並びで叩きたいときに使う。

## ファイル構成

```
isaac-lab-bench/
├── README.md                 ← 本ファイル
├── bench.env.sample          DCV モード用テンプレ (DEST_BASE 等)
├── ssm.env.sample            SSM モード用テンプレ (INSTANCE_ID + AWS_REGION)
├── orchestrate.sh            SSM モードの入口。aws/jq/python3 不在時は即エラー
├── scripts/                  どちらのモードからも呼ばれる本体
│   ├── run_bench.sh          benchmark_rsl_rl.py を seed × iter で回す
│   ├── dmon_stats.py         nvidia-smi dmon ログ集計 (text/csv/md)
│   └── check_local.sh        DCV モード用の pre-flight チェック
└── results/                  結果置き場 (gitignore)
```

`scripts/` 配下はモードに依存しない。`orchestrate.sh` は SSM 経由で `scripts/run_bench.sh` を base64 転送して走らせるだけ。

## DCV モード (EC2 上で直接実行)

```bash
git clone https://github.com/littlemex/zenmai-lab.git
cd zenmai-lab/experiments/isaac-lab-bench

# 初回のみ
cp bench.env.sample bench.env
$EDITOR bench.env             # DEST_BASE / ISAACLAB_DIR 等を確認
source bench.env

# (任意) 前提が揃っているか確認
bash scripts/check_local.sh

# ベンチ実行 (引数: <task> <tag> [seeds] [iters] [num_envs])
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 g1-quick "42" 50 4096
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 g1-prod "42 123 456 789 1337" 100 4096
bash scripts/run_bench.sh Isaac-Cartpole-v0          cp-prod "42 123 456 789 1337" 100 4096

# 集計
python3 scripts/dmon_stats.py --format md "$DEST_BASE"/g1-prod/run-seed*/nvidia-smi-dmon.log
```

`aws` CLI も `ssm.env` も不要。

## SSM モード (ラップトップから駆動)

```bash
git clone https://github.com/littlemex/zenmai-lab.git
cd zenmai-lab/experiments/isaac-lab-bench

# 初回のみ
cp ssm.env.sample ssm.env
$EDITOR ssm.env               # INSTANCE_ID, AWS_REGION

# 走らせる (プロファイルから選ぶ)
bash orchestrate.sh g1            # 5 seeds × 100 iter, ~15 min
bash orchestrate.sh g1-quick      # 1 seed × 50 iter, ~3-4 min
bash orchestrate.sh cartpole      # 5 seeds × 100 iter, ~5 min
bash orchestrate.sh g1-num-envs   # num_envs sweep {4096, 8192, 16384}, ~30 min
```

実行中は 60 秒間隔でリモートの進捗が出る。完了すると `results/<profile>-<timestamp>/` に成果物が落ち、最後に `dmon_stats.py --format md` の集計表が表示される。

プロファイルを増やす / 既存を変えるには `orchestrate.sh` の `case "$PROFILE"` を編集する。

## 出力レイアウト (両モード共通)

```
<DEST_BASE>/<tag>/                 ← DCV モード: EC2 上のパス
results/<profile>-<stamp>/<tag>/   ← SSM モード: ローカルに展開後のパス
├── summary.csv
├── run-seed42-…/
│   ├── bench.log                  benchmark_rsl_rl.py の生 stdout
│   └── nvidia-smi-dmon.log        1Hz GPU サンプル
└── …
```

`summary.csv` のスキーマ:

```
seed, iters, task, num_envs, app_launch_ms, total_start_ms,
mean_total_fps, mean_collection_fps, max_rewards, wall_clock_s
```

## トラブル

| 症状 | 確認場所 |
|---|---|
| `summary.csv` が空 | run-seed* 内の `bench.log`。Isaac Sim の初回シェーダーキャッシュ生成で 50 秒前後かかる |
| `'unknown': I need something more specific.` | `scripts/run_bench.sh` 内で `TERM=xterm` を export しているので最新版か確認 |
| SSM が `InvalidInstanceId` | インスタンスが stopped か SSM Agent 落ち。`aws ec2 describe-instances` |
| `mean_total_fps` が `NA` | `bench.log` に `Mean Total FPS:` 行が無い = ベンチが死んだ。多くは OOM。`dmesg | grep -i oom` |
| `aws: command not found` (DCV モード) | DCV モードでは aws CLI 不要。`scripts/run_bench.sh` を直接叩く |
