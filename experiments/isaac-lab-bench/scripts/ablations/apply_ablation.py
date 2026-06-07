#!/usr/bin/env python3
"""Apply a named ablation to the customer's IsaacLab config.

The script edits `<ISAACLAB_DIR>/source/isaaclab_tasks/isaaclab_tasks/manager_based/
locomotion/velocity/config/g1/rough_env_cfg.py` by appending a known marker block
to the existing ``__post_init__`` of the ``G1RoughEnvCfg`` class.

Idempotent: running the same ablation twice is a no-op. Always reverts the
previous block before applying a new one. ``--name none`` reverts only.

Available ablations
-------------------
- ``none``                 No-op; just reverts any previous patch.
- ``contact-scope-ankle``  Replace the all-link contact sensor with one that
                            only watches ``.*_ankle_roll_link``.
- ``solver-iter-half``     Lower G1 articulation solver counters
                            (``solver_position_iteration_count`` 8 -> 4,
                            ``solver_velocity_iteration_count`` 4 -> 1).
- ``height-scan-none``     Disable the height scanner and switch the terrain
                            to a plane.
- ``height-scan-halffreq`` Double the height scanner ``update_period``.
- ``height-scan-lowres``   Reduce the height scanner ray density
                            (``resolution`` 0.10 -> 0.15, ``size`` 1.6x1.0 -> 1.0x0.6).
- ``combined``             Apply ``contact-scope-ankle`` + ``solver-iter-half`` +
                            ``height-scan-halffreq`` together.

Usage
-----
    python3 apply_ablation.py --isaaclab-dir ~/IsaacLab --name contact-scope-ankle
    python3 apply_ablation.py --name none      # revert

The patch block is delimited with ``# >>> zenmai-ablation BEGIN <<<`` /
``# >>> zenmai-ablation END <<<`` so it can always be located and removed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path

BEGIN = "# >>> zenmai-ablation BEGIN <<<"
END = "# >>> zenmai-ablation END <<<"

PATCHES = {
    "none": "",
    "contact-scope-ankle": textwrap.dedent("""
        from isaaclab.sensors.contact_sensor import ContactSensorCfg
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            history_length=3,
            track_air_time=True,
        )
    """),
    "solver-iter-half": textwrap.dedent("""
        from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg
        from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
        self.scene.robot = G1_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_MINIMAL_CFG.spawn.replace(
                articulation_props=ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
        )
    """),
    "height-scan-none": textwrap.dedent("""
        self.scene.height_scanner = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
    """),
    "height-scan-halffreq": textwrap.dedent("""
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = 2.0 * self.decimation * self.sim.dt
    """),
    "height-scan-lowres": textwrap.dedent("""
        from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.pattern_cfg = GridPatternCfg(
                resolution=0.15, size=(1.0, 0.6)
            )
    """),
    "combined": (
        textwrap.dedent("""
            from isaaclab.sensors.contact_sensor import ContactSensorCfg
            self.scene.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
                history_length=3,
                track_air_time=True,
            )
            from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg
            from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
            self.scene.robot = G1_MINIMAL_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=G1_MINIMAL_CFG.spawn.replace(
                    articulation_props=ArticulationRootPropertiesCfg(
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=1,
                    ),
                ),
            )
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = 2.0 * self.decimation * self.sim.dt
        """)
    ),
}

CONFIG_REL = (
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
    "locomotion/velocity/config/g1/rough_env_cfg.py"
)


def find_post_init_end(text: str) -> int:
    """Return the column-0 index just before the function after __post_init__,
    so we can insert the patch at the end of the method body."""
    # Locate the def __post_init__ line of G1RoughEnvCfg.
    m = re.search(r"\n    def __post_init__\(self\)[^\n]*:\n", text)
    if not m:
        raise ValueError("__post_init__ not found in rough_env_cfg.py")
    body_start = m.end()
    # Find next dedented line (4 spaces or fewer for class def boundary, or EOF).
    rest = text[body_start:]
    # Walk until we find a line that does not start with 8 spaces (i.e. exits method).
    i = 0
    while i < len(rest):
        nl = rest.find("\n", i)
        if nl == -1:
            return len(text)  # EOF
        line = rest[i:nl]
        if line.strip() == "":
            i = nl + 1
            continue
        if not line.startswith("        "):  # left the method
            return body_start + i
        i = nl + 1
    return len(text)


def revert(text: str) -> str:
    """Strip any existing zenmai-ablation block."""
    pattern = re.compile(
        r"\n? *" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?",
        re.DOTALL,
    )
    return pattern.sub("\n", text)


def apply_block(text: str, block_body: str) -> str:
    indent = "        "  # 8 spaces, inside method
    # Indent each non-empty line of block_body.
    indented = "\n".join(indent + l if l.strip() else "" for l in block_body.strip("\n").splitlines())
    block = f"\n{indent}{BEGIN}\n{indented}\n{indent}{END}\n"
    insert_at = find_post_init_end(text)
    return text[:insert_at].rstrip() + "\n" + block + text[insert_at:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-dir", default=os.environ.get("ISAACLAB_DIR", "/home/ubuntu/IsaacLab"))
    parser.add_argument("--name", required=True, choices=sorted(PATCHES.keys()))
    parser.add_argument("--show", action="store_true", help="print the resulting file fragment instead of writing")
    args = parser.parse_args()

    cfg_path = Path(args.isaaclab_dir) / CONFIG_REL
    if not cfg_path.is_file():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 1

    text = cfg_path.read_text()
    text = revert(text)

    if args.name == "none":
        new_text = text
        action = "reverted"
    else:
        body = PATCHES[args.name]
        if not body.strip():
            print(f"empty patch body for '{args.name}'", file=sys.stderr)
            return 1
        new_text = apply_block(text, body)
        action = f"applied '{args.name}'"

    if args.show:
        # show only the __post_init__ method for confirmation
        m = re.search(r"\n    def __post_init__\(self\)[^\n]*:\n", new_text)
        if m:
            tail = new_text[m.start():]
            # take up to 80 lines
            print("\n".join(tail.splitlines()[:80]))
        else:
            print(new_text[-2000:])
        return 0

    cfg_path.write_text(new_text)
    print(f"{action} on {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
