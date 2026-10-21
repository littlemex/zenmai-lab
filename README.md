# zenmai-lab

Experiments on models: small, self-contained studies of how a model or a model component behaves, each with the code, the configuration, the results and a README that reproduces them.

Cluster infrastructure (EKS clusters, GPU node pools, storage) comes from [littlemex/distributed-ai](https://github.com/littlemex/distributed-ai). An experiment here assumes a working Kubernetes cluster and brings only what it runs on it: its manifests and its code.

## Layout

```
experiments/
  YYYY-MM-DD-<name>/    one experiment: README, code, configs, manifests, results
legacy/physical-ai/     the earlier Physical AI lab layout (EC2 and Slurm helpers, templates, benchmarks)
```

## Experiments

| Directory | What it studies |
|---|---|
| [`experiments/2026-06-06-isaac-lab-bench`](experiments/2026-06-06-isaac-lab-bench) | Isaac Lab's RSL-RL benchmark on GPU instances: throughput, GPU use and PhysX ablations |
| [`experiments/2026-10-21-engram-nanogpt`](experiments/2026-10-21-engram-nanogpt) | A small GPT trained with and without an Engram layer that hashes like DeepSeek-V4.1-Flash |
| [`experiments/2026-10-21-engram-trace-viewer`](experiments/2026-10-21-engram-trace-viewer) | A static 3D viewer of Engram's n-gram lookups, with a recorder that uses the model's own hashing |

## Adding an experiment

1. Create `experiments/<today's date>-<short-name>/`. The date is the day the experiment starts, so the directory list reads in order.
2. Keep everything the experiment needs inside it, with a README that states what it measures, how to reproduce it from a fresh clone, and the results.
3. Run on a cluster built with distributed-ai; keep cluster-specific values (namespaces, account ids, addresses) out of the files and pass them as environment variables.
4. Track results that a README cites; large outputs such as checkpoints stay out of git.
