# 新実験を追加する

## SOP (5 ステップ)

1. **テンプレートをコピー**
   ```bash
   cp -r experiments/_template experiments/<my-experiment>
   cd experiments/<my-experiment>
   ```

2. **`push.env` を設定**
   ```bash
   cp push.env.sample push.env
   $EDITOR push.env
   ```

3. **`Dockerfile` と `train.sh` を実験内容に合わせる**

   - PyTorch 系なら `FROM <account>.dkr.ecr.<region>.amazonaws.com/zenmai-base-pytorch:latest`
     (まず `common/containers/base-pytorch.Dockerfile` を `zenmai-base-pytorch` として ECR にビルド & push)
   - JAX 系なら `base-jax.Dockerfile` を使用

4. **動作確認**
   ```bash
   bash run.sh
   ```

5. **結果を集計**
   ```bash
   bash $(git rev-parse --show-toplevel)/common/scripts/collect-results.sh
   ```

## チェックリスト

- [ ] `push.env` は git add していない (`git status` で確認)
- [ ] `Dockerfile` は `requirements.txt` を内包しているか、または直接 `pip install` を書いた
- [ ] `train.sh` は `set -e` で書かれている
- [ ] `results/summary.csv` のヘッダは `task,seed,metric,value` である
- [ ] `README.md` に「結論」と「再現手順」が書かれている
- [ ] 大容量ファイル (動画、チェックポイント) は `common/scripts/sync-to-s3.sh` で S3 へ
- [ ] **機密情報** (実 IP、実アカウント ID、実パスワード) を repo に含めていない
