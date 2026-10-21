# Setup

## 前提

- AWS CLI v2 (認証済み)
- Docker (buildx)
- ssh
- (HyperPod 利用時のみ) physai CLI または直接 ssh で head node にアクセスできる状態

## 初期化

```bash
git clone git@github.com:littlemex/zenmai-lab.git
cd zenmai-lab
cp common/infra/env.sample common/infra/env
$EDITOR common/infra/env   # AWS_ACCOUNT_ID と S3_BUCKET を設定
```

`common/infra/env` は gitignore されている。

## 動作確認

```bash
source common/infra/env
aws sts get-caller-identity
echo "Account: ${AWS_ACCOUNT_ID}"
```

## 新実験

`docs/ADD-EXPERIMENT.md` を参照。
