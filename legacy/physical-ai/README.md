# zenmai-lab

Physical AI 実験ラボ。

## 構成

```
common/         共通インフラ・スクリプト・ベース Dockerfile
experiments/    各実験。_template/ を cp -r して開始
benchmarks/     横断結果集計と過去ベンチの永続記録
notebooks/      EC2 不要の軽量探索
docs/           SETUP / ADD-EXPERIMENT
```

## クイックスタート

```bash
cp common/infra/env.sample common/infra/env
$EDITOR common/infra/env

cp -r experiments/_template experiments/my-experiment
cd experiments/my-experiment
cp push.env.sample push.env
$EDITOR push.env
bash run.sh
```

詳細: [docs/SETUP.md](docs/SETUP.md), [docs/ADD-EXPERIMENT.md](docs/ADD-EXPERIMENT.md)
