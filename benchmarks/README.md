# benchmarks/

実験横断の結果集計と、過去の検証ジャーナル。

## サブディレクトリ

| パス | 役割 |
|------|------|
| `pask-audit/` | 2026-06 の PASK (aws-samples/sample-physical-ai-scaffolding-kit) 検証成果。永続記録。 |
| `cross-experiment/` | `common/scripts/collect-results.sh` が `experiments/*/results/summary.csv` を集約した結果を置く。 |

## 集計の更新

実験完了後、その実験ディレクトリで:

```bash
bash $(git rev-parse --show-toplevel)/common/scripts/collect-results.sh
```

`benchmarks/cross-experiment/summary.csv` に追記される。
