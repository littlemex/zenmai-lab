# zenmai-lab vs PASK の設計差分

`aws-samples/sample-physical-ai-scaffolding-kit` (PASK) を 2026-06 に検証した結果を踏まえ、zenmai-lab で意識的に変えた設計判断のメモ。

## 1. 階層を浅く

PASK は `physai/cli/`, `physai/infra/`, `physai/examples/`, `samples/{gr00t,newton-rl,openpi-sample}/training/` 等で 4-5 階層。zenmai-lab は **3 階層以内**。

## 2. `ec2-push.sh` を 1 本に集約

PASK には `samples/{gr00t,newton-rl,openpi-sample}/training/build_and_push_ecr.sh` が 3 本ある。差分は (a) ビルドコンテキスト、(b) `--build-arg ACCEPT_EULA=Y` の有無、(c) `OPENPI_ROOT` の前提のみ。

zenmai-lab は `common/infra/ec2-push.sh` 1 本で吸収する:
- `BUILD_CONTEXT` 環境変数
- `BUILD_ARG_*` 接頭辞付き変数を自動展開

## 3. SSM を使わない (EC2 単台では)

PASK の `physai/infra/scripts/run-lifecycle.sh` は SSM 経由で sagemaker-cluster:host を扱う。EC2 単台のラボには過剰。zenmai-lab は素朴な `ssh-run.sh` のみ。HyperPod 用には別途 `slurm-submit.sh` を持つ。

## 4. `push.env` を実験単位で gitignore

PASK では各 sample が独自の環境変数 `.env` を持つ習慣が薄い。zenmai-lab は `_template/push.env.sample` で規律を強制し、`push.env` は必ず gitignore する。

## 5. 検証済みの PASK バグ

`benchmarks/pask-audit/PASK-AUDIT.md` 参照。
- `tabs 4` + `set -e` でクラウド非 TTY 環境では Isaac Lab 必ず壊れる (Critical, [Isaac Lab 本家 issue](ISAACLAB-ISSUE.md))
- `unattended-upgrades` と apt の dpkg ロック競合 (High)
- README の `cd cdk` ディレクトリ誤記 (High)
- 90 finding を `gh pr` 用にまとめたものが `PASK-PR-BODY.md`
