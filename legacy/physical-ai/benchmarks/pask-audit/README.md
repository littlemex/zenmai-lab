# PASK Audit (2026-06)

`aws-samples/sample-physical-ai-scaffolding-kit` (PASK) の監査ログと、その上で計測した Isaac Lab ベンチマーク結果。

機密情報 (アカウント ID、インスタンス ID、EIP、IP、パスワード、S3 バケット名) は `<placeholder>` で置換済み。

## 含まれるドキュメント

| ファイル | 内容 |
|---------|------|
| `PASK-AUDIT.md` | 6 レンズで PASK を多角レビュー。確認済み 90 finding。 |
| `PASK-PR-BODY.md` | 第 1 PR (TERM 修正・dpkg ロック・README 訂正) の本文。 |
| `ISAACLAB-ISSUE.md` | NVIDIA Isaac Lab 本家への bug report (英語)。`tabs 4` + `set -e` 問題。 |
| `results/cartpole-summary.csv` | Cartpole-v0, 5 シード, 100 iter の生 KPI。 |
| `results/g1-summary.csv` | Velocity-Rough-G1-v0, 5 シード, 100 iter の生 KPI。 |

## 主な発見

- L40S (g6e.2xlarge) の SM 利用率は Cartpole で 11%、G1 で 48% に留まる → GPU はボトルネックでない
- ボトルネックは CPU シングルスレッド性能 + PCIe x8 リンク制限 + ECC 有効化
- 推奨対処: g6e.4xlarge (vCPU 16) へのスケールアップ、または num_envs スケーリング

## 公開状態

このリポジトリは public です。RUNBOOK.md (実環境固有のコマンド付き手順書) は別途ローカル管理しています。
