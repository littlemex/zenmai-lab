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
├── README.md                       ← 本ファイル
├── NOTES.md                        実験ノート (実施コマンド・条件・結果)
├── PERF-LEVERS.md                  91 件の確認済みチューニングレバー
├── PREP-PLAN.md                    顧客コール準備実行計画 + meeting playbook
├── RESULTS-SUMMARY.md              ベンチ結果サマリ
├── bench.env.sample                DCV モード用テンプレ (DEST_BASE 等)
├── ssm.env.sample                  SSM モード用テンプレ (INSTANCE_ID + AWS_REGION)
├── orchestrate.sh                  SSM モードの入口。aws/jq/python3 不在時は即エラー
├── scripts/                        どちらのモードからも呼ばれる本体
│   ├── run_bench.sh                benchmark_rsl_rl.py を seed × iter で回す
│   ├── dmon_stats.py               nvidia-smi dmon ログ集計 (text/csv/md)
│   ├── check_local.sh              DCV モード用の pre-flight チェック
│   ├── diagnose.sh                 顧客向け 30 秒セルフ診断 (環境差を取る)
│   ├── profile_nsys.sh             Nsight Systems で 1 ラン分のタイムラインを取る
│   ├── profile_pyspy.sh            py-spy で Python メインスレッドのフレームグラフを取る
│   ├── pyspy_tree.py               SVG フレームグラフを md/text ツリーに変換
│   └── ablations/
│       ├── apply_ablation.py       命名された ablation を冪等にパッチ
│       └── run_one.sh              SSM 経由 1 コマンドで ablation 実行
├── docs/
│   └── isaac-lab-bench-stack.drawio   レイヤー構造の draw.io 図
└── results/                        結果置き場 (gitignore、CSV のみ追跡)
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

## ablation・プロファイル・診断

### ablation
`scripts/ablations/apply_ablation.py` が `G1RoughEnvCfg.__post_init__` に名前付きパッチを冪等に注入する。`--name none` で revert。サポートされるパッチ:

| 名前 | 内容 |
|---|---|
| `contact-scope-ankle` | contact_forces を torso + ankle (3 リンク) のみに絞る |
| `solver-iter-half` | G1 articulation の solver 反復を半減 (pos 8→4, vel 4→1) |
| `height-scan-halffreq` | height_scanner の update_period を 2× |
| `height-scan-lowres` | height_scanner の解像度・サイズを削減 |
| `height-scan-none` | height_scanner=None + plane terrain (G1-Rough cfg では不整合、参考用) |
| `combined` | contact-scope-ankle + solver-iter-half + height-scan-halffreq |

```bash
# EC2 上で直接 (DCV)
python3 scripts/ablations/apply_ablation.py --isaaclab-dir ~/IsaacLab --name contact-scope-ankle
bash scripts/run_bench.sh Isaac-Velocity-Rough-G1-v0 abl-contact "42 123 456" 100 16384
python3 scripts/ablations/apply_ablation.py --name none   # revert
```

各 ablation の実測結果は `RESULTS-SUMMARY.md` を参照。

### Nsight Systems プロファイル
`scripts/profile_nsys.sh` が benchmark_rsl_rl.py を `nsys profile` でラップして実行し、1 ラン分の GPU タイムラインを `.nsys-rep` で保存、`nsys stats` で kernels.csv / nvtx_trace.csv / gpu-gaps.txt を生成する。

```bash
# EC2 上で。事前に nsys がインストール済みであること
# (Isaac Sim AMI 標準では未インストール: sudo apt install nsight-systems-2024.2.3)
bash scripts/profile_nsys.sh Isaac-Velocity-Rough-G1-v0 g1-prof "42" 30 4096
```

`.nsys-rep` は数百 MB になり gitignore 対象。CSV / TXT サマリのみコミットする。

### py-spy フレームグラフ
GPU 待ち中の Python メインスレッドが何で時間を使っているか可視化する。`scripts/profile_pyspy.sh` が benchmark_rsl_rl.py を py-spy でアタッチし、フレームグラフ SVG とテキストダンプを生成する。

```bash
# EC2 上で。sudo 権限が必要 (ptrace_scope=1 のため)
bash scripts/profile_pyspy.sh Isaac-Velocity-Rough-G1-v0 g1-pyspy "42" 100 16384
```

出力先: `results/pyspy/<tag>/run-seed*/pyspy.svg` と `pyspy-dump.txt`

SVG をローカルで解析する場合:

```bash
python3 scripts/pyspy_tree.py <path-to-svg> --format md --min-pct 1.0 --max-depth 8
```

### 顧客環境セルフ診断
`scripts/diagnose.sh` は EC2 上で実行する read-only の <30 秒スクリプト。GPU・CPU・IsaacLab・conda・PyTorch・指定タスクの主要 cfg をダンプする。顧客が自環境で実行 → 出力をペーストしてもらえれば、こちらの参照環境との差分が即分かる。

```bash
bash scripts/diagnose.sh                                            # default Isaac-Velocity-Rough-G1-v0
bash scripts/diagnose.sh --task Isaac-Velocity-Flat-G1-v0
```

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
