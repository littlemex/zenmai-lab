# `<experiment-name>`

> 1 行で実験の目的を書く

## 結論

実験完了後にここを更新する。

## 再現手順

```bash
# 1. push.env を設定
cp push.env.sample push.env
$EDITOR push.env

# 2. ビルド & プッシュ & 実行
bash run.sh
```

## 設定

`configs/` 配下のファイルで挙動を変更。

## 結果

`results/summary.csv` に集計。大容量ファイル (動画 / チェックポイント) は S3。

## 関連

- リンク先 (関連 issue / PR / モデルカード)
