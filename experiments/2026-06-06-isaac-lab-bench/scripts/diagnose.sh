#!/usr/bin/env bash
# Self-diagnostic for an Isaac Lab + EC2 setup. Runs in <30 seconds and
# prints a JSON-ish blob the customer can paste back to us.
#
# Usage (on the EC2 box, no sudo needed):
#   bash diagnose.sh
#   bash diagnose.sh --task Isaac-Velocity-Rough-G1-v0
#
# What it captures:
#   - GPU model / driver / CUDA / ECC / PCIe link width / persistence mode
#   - CPU model / vCPU count / clock
#   - System memory total / available
#   - IsaacLab branch / commit / install version
#   - Conda env (env_isaaclab) status
#   - PhysX-side default flags (use_fabric / enable_ccd / enable_enhanced_determinism)
#   - For a given task: num_envs, decimation, dt, solver iter counts,
#     contact sensor prim_path, height scanner config
#
# It does NOT launch a benchmark. It only reads config — safe to run in seconds.

set -o pipefail

TASK="${1:-Isaac-Velocity-Rough-G1-v0}"
if [ "$1" = "--task" ]; then TASK="${2:-Isaac-Velocity-Rough-G1-v0}"; fi

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "=== zenmai-lab self-diagnostic ($(ts)) ==="
echo
echo "--- task ---"
echo "task=$TASK"
echo
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,pstate,persistence_mode,ecc.mode.current,memory.total,memory.used --format=csv 2>&1
  echo "(pcie link)"
  nvidia-smi -q | grep -i 'PCIe\|link width' | head -10
else
  echo "nvidia-smi not found"
fi
echo
echo "--- CPU ---"
lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Thread|CPU max MHz|CPU MHz)' | head -8
grep -E '^MemTotal|^MemAvailable' /proc/meminfo 2>/dev/null | head -2
echo
echo "--- IsaacLab repo ---"
if [ -d "$ISAACLAB_DIR/.git" ]; then
  ( cd "$ISAACLAB_DIR" && git log -1 --oneline 2>&1 && git rev-parse --abbrev-ref HEAD 2>&1 )
else
  echo "no git repo at $ISAACLAB_DIR"
fi
echo
echo "--- Isaac Sim version ---"
[ -f /opt/IsaacSim/VERSION ] && cat /opt/IsaacSim/VERSION || echo "missing /opt/IsaacSim/VERSION"
echo
echo "--- conda env ---"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH" 2>/dev/null
  conda env list 2>/dev/null | grep -E "^${CONDA_ENV}|^\*" || echo "$CONDA_ENV not present"
else
  echo "no conda init script at $CONDA_SH"
fi
echo
echo "--- python imports & versions ---"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH" && conda activate "$CONDA_ENV" 2>/dev/null
fi
python3 - <<'PYEOF' 2>&1 | head -30
import importlib
for pkg in ('torch', 'isaaclab', 'isaaclab_tasks', 'rsl_rl', 'rl_games', 'skrl', 'stable_baselines3'):
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', '?')
        print(f"{pkg}: {v}")
    except Exception as e:
        print(f"{pkg}: NOT AVAILABLE ({type(e).__name__})")
try:
    import torch
    print(f"torch.cuda: {torch.cuda.is_available()} dev={torch.cuda.device_count()}")
    print(f"torch.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"torch.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
except Exception as e:
    print(f"torch probe failed: {e}")
PYEOF
echo
echo "--- task config probe ---"
export TERM=xterm
python3 - <<PYEOF 2>&1 | head -60
import os
os.environ.setdefault("ISAAC_LAB_QUIET", "1")
try:
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry("$TASK", "env_cfg_entry_point")
    print(f"task: $TASK")
    print(f"num_envs: {getattr(env_cfg.scene, 'num_envs', '?')}")
    print(f"decimation: {getattr(env_cfg, 'decimation', '?')}")
    print(f"sim.dt: {getattr(env_cfg.sim, 'dt', '?')}")
    print(f"sim.use_fabric: {getattr(env_cfg.sim, 'use_fabric', '?')}")
    physx = getattr(env_cfg.sim, 'physx', None)
    if physx is not None:
        print(f"physx.solver_type: {getattr(physx, 'solver_type', '?')}")
        print(f"physx.enable_ccd: {getattr(physx, 'enable_ccd', '?')}")
        print(f"physx.enable_enhanced_determinism: {getattr(physx, 'enable_enhanced_determinism', '?')}")
        print(f"physx.gpu_max_rigid_contact_count: {getattr(physx, 'gpu_max_rigid_contact_count', '?')}")
        print(f"physx.gpu_max_rigid_patch_count: {getattr(physx, 'gpu_max_rigid_patch_count', '?')}")
        print(f"physx.gpu_found_lost_pairs_capacity: {getattr(physx, 'gpu_found_lost_pairs_capacity', '?')}")
    cf = getattr(env_cfg.scene, 'contact_forces', None)
    if cf is not None:
        print(f"contact_forces.prim_path: {getattr(cf, 'prim_path', '?')}")
        print(f"contact_forces.history_length: {getattr(cf, 'history_length', '?')}")
    hs = getattr(env_cfg.scene, 'height_scanner', None)
    if hs is None:
        print("height_scanner: None")
    else:
        pat = getattr(hs, 'pattern_cfg', None)
        print(f"height_scanner.update_period: {getattr(hs, 'update_period', '?')}")
        if pat is not None:
            print(f"height_scanner.pattern.resolution: {getattr(pat, 'resolution', '?')}")
            print(f"height_scanner.pattern.size: {getattr(pat, 'size', '?')}")
    robot = getattr(env_cfg.scene, 'robot', None)
    if robot is not None:
        spawn = getattr(robot, 'spawn', None)
        if spawn is not None:
            ap = getattr(spawn, 'articulation_props', None)
            if ap is not None:
                print(f"articulation.solver_position_iteration_count: {getattr(ap, 'solver_position_iteration_count', '?')}")
                print(f"articulation.solver_velocity_iteration_count: {getattr(ap, 'solver_velocity_iteration_count', '?')}")
except Exception as e:
    print(f"task probe failed: {type(e).__name__}: {e}")
PYEOF
echo
echo "=== diagnose done ==="
