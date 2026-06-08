#!/usr/bin/env python3
"""py-spy / inferno SVG flamegraph and speedscope JSON -> Markdown tree viewer

SVG flamegraphs are hard to read without a browser, so this tool renders them
as a CLI Markdown tree that can be pasted directly into documents or chats.

Supported input formats (auto-detected by file extension):
  .svg              -- inferno inverted (icicle) flamegraph
  .json             -- speedscope JSON profile (including .speedscope.json)

Supported SVG format: inferno inverted=true (icicle) flamegraph.
Each frame is:
  <g><title>NAME (file:line) (N samples, X.YZ%)</title><rect ... fg:x="X" fg:w="W"/></g>

Speedscope JSON format reference:
  Top level: {"$schema":"...", "shared":{"frames":[FrameInfo,...]},
              "profiles":[Profile,...], "activeProfileIndex":int}
  FrameInfo: {"name":str, "file":Optional[str], "line":Optional[int]}
  Profile types:
    sampled: {"type":"sampled","samples":[[frame_idx,...],...],
              "weights":[num,...], ...}
    evented: {"type":"evented","events":[{"type":"O"|"C","frame":idx,"at":num},...]}

Output modes (--format):
  md   (default) : Markdown nested bullet tree + frames table
  text           : indented ASCII tree (no box-drawing characters)
  json           : raw tree as JSON (nested dict)

Usage examples
--------------
  python pyspy_tree.py results/pyspy/pyspy-4xl-full.svg
  python pyspy_tree.py results/pyspy/pyspy-4xl-full.svg --format md --min-pct 1.0 --max-depth 8 --max-children 6 --top-leaves 30
  python pyspy_tree.py results/pyspy/pyspy-4xl-full.svg --format json | python -m json.tool | head -60

  # Speedscope JSON
  python pyspy_tree.py profile.speedscope.json --format md --min-pct 1.0 --top-leaves 15
  python pyspy_tree.py profile.json --format text
"""

import argparse
import json
import os
import re
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Parse SVG
# ---------------------------------------------------------------------------

_FRAME_PATTERN = re.compile(
    r'<g><title>([^<]*)</title>'
    r'<rect[^>]*\by="(\d+)"[^>]*\bfg:x="(\d+)"[^>]*\bfg:w="(\d+)"'
)
_TITLE_PATTERN = re.compile(
    r'^(.*?)\s+\(([\d,]+)\s+samples,\s*([\d.]+)%\)$'
)


def _unescape_html(s: str) -> str:
    """Unescape HTML entities in a string."""
    return (
        s
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


def parse_svg(path: str) -> list[dict]:
    """Return frame list from SVG (x, w, y, name, location, samples, pct)."""
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    # NOTE: Match on the RAW content (before HTML entity unescaping).
    # Titles like <module>, <lambda>, <genexpr> are stored as &lt;...&gt; in SVG.
    # If we unescape the whole document first, the [^<]* in _FRAME_PATTERN can no
    # longer match those titles because the literal '<' terminates the character class.
    # Instead, unescape each captured title individually after extraction.

    frames = []
    for m in _FRAME_PATTERN.finditer(content):
        title_raw, y_str, fx_str, fw_str = m.groups()
        title_unescaped = _unescape_html(title_raw)
        tm = _TITLE_PATTERN.match(title_unescaped.strip())
        if not tm:
            continue
        name_loc = tm.group(1).strip()
        samples = int(tm.group(2).replace(",", ""))
        pct = float(tm.group(3))
        y = int(y_str)
        fx = int(fx_str)
        fw = int(fw_str)

        # Split "name (file:line)" into name + location
        loc_m = re.match(r'^(.*?)\s+\(([^)]+:\d+)\)$', name_loc)
        if loc_m:
            name = loc_m.group(1).strip()
            location = loc_m.group(2).strip()
        else:
            name = name_loc
            location = ""

        frames.append({
            "y": y,
            "x": fx,
            "w": fw,
            "name": name,
            "location": location,
            "samples": samples,
            "pct": pct,
        })

    return frames


# ---------------------------------------------------------------------------
# 2. Build tree (SVG path)
# ---------------------------------------------------------------------------

def annotate_self(node: dict) -> None:
    """Compute _self for every node: samples minus sum of direct children."""
    children_total = sum(c["samples"] for c in node["children"])
    node["_self"] = max(0, node["samples"] - children_total)
    for c in node["children"]:
        annotate_self(c)


def build_tree(frames: list[dict]) -> dict:
    """
    Parent detection: F2's parent = the frame with y < F2.y whose x-range
    contains [F2.x, F2.x+F2.w], with the largest y (closest ancestor).

    Returns root node dict with children list.
    """
    # Sort by y asc, x asc
    sorted_frames = sorted(frames, key=lambda f: (f["y"], f["x"]))

    # Add index and children list
    for i, f in enumerate(sorted_frames):
        f["_id"] = i
        f["children"] = []

    n = len(sorted_frames)
    parent_of = [-1] * n  # index into sorted_frames

    for i, frame in enumerate(sorted_frames):
        best_parent_idx = -1
        best_parent_y = -1
        fx, fw = frame["x"], frame["w"]
        for j in range(i - 1, -1, -1):
            candidate = sorted_frames[j]
            if candidate["y"] >= frame["y"]:
                continue
            cx, cw = candidate["x"], candidate["w"]
            # Check containment: [fx, fx+fw] subset of [cx, cx+cw]
            if cx <= fx and (fx + fw) <= (cx + cw):
                if candidate["y"] > best_parent_y:
                    best_parent_y = candidate["y"]
                    best_parent_idx = j
        parent_of[i] = best_parent_idx

    # Build children lists
    roots = []
    for i, frame in enumerate(sorted_frames):
        p = parent_of[i]
        if p == -1:
            roots.append(frame)
        else:
            sorted_frames[p]["children"].append(frame)

    # Sort children by samples desc at each node
    def sort_children(node: dict) -> None:
        node["children"].sort(key=lambda c: c["samples"], reverse=True)
        for child in node["children"]:
            sort_children(child)

    # Create virtual root if multiple roots
    if len(roots) == 1:
        root = roots[0]
    else:
        # Wrap in virtual root
        root = {
            "y": -1, "x": 0, "w": sum(r["w"] for r in roots),
            "name": "(all threads)",
            "location": "",
            "samples": max(r["samples"] for r in roots),
            "pct": max(r["pct"] for r in roots),
            "_id": -1,
            "children": sorted(roots, key=lambda r: r["samples"], reverse=True),
        }

    sort_children(root)
    annotate_self(root)
    return root


# ---------------------------------------------------------------------------
# 3. Parse speedscope JSON
# ---------------------------------------------------------------------------

def parse_speedscope(path: str) -> dict:
    """Parse a speedscope JSON profile and return a tree in the same shape as
    build_tree(parse_svg(...)):
      {"name": str, "location": str, "samples": num, "pct": float,
       "_self": num, "children": [...]}
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    frames_info = data.get("shared", {}).get("frames", [])
    profiles = data.get("profiles", [])
    if not profiles:
        raise ValueError("speedscope JSON has no profiles")

    active_idx = data.get("activeProfileIndex", 0) or 0
    profile = profiles[active_idx]

    ptype = profile.get("type", "sampled")

    def frame_name(idx: int) -> tuple[str, str]:
        """Return (name, location) for a frame index."""
        fi = frames_info[idx] if idx < len(frames_info) else {}
        name = fi.get("name", f"frame_{idx}")
        file_ = fi.get("file")
        line = fi.get("line")
        if file_ and line is not None:
            location = f"{file_}:{line}"
        elif file_:
            location = file_
        else:
            location = ""
        return name, location

    # Each node in our tree: {name, location, samples, children, _self}
    # We build it as a dict keyed by path tuple for fast lookup.

    if ptype == "sampled":
        samples_list = profile.get("samples", [])
        weights = profile.get("weights", [])
        total_weight = sum(weights) if weights else len(samples_list)

        # Build call tree incrementally
        # Node structure: {"name", "location", "samples", "children": {key: node}}
        # We use a dict-of-children keyed by frame_idx for O(1) lookup.
        root_node: dict = {
            "name": "all",
            "location": "",
            "samples": total_weight,
            "children": {},
        }

        for i, stack in enumerate(samples_list):
            w = weights[i] if i < len(weights) else 1
            current = root_node
            for frame_idx in stack:  # root -> leaf order
                key = frame_idx
                if key not in current["children"]:
                    n2, loc = frame_name(frame_idx)
                    current["children"][key] = {
                        "name": n2,
                        "location": loc,
                        "samples": 0,
                        "children": {},
                    }
                child_node = current["children"][key]
                child_node["samples"] += w
                current = child_node

    elif ptype == "evented":
        events = profile.get("events", [])
        start_val = profile.get("startValue", 0)
        end_val = profile.get("endValue", 0)
        total_weight = end_val - start_val

        root_node = {
            "name": "all",
            "location": "",
            "samples": total_weight,
            "children": {},
        }

        # Stack of (frame_idx, node, enter_time)
        stack: list[tuple[int, dict, float]] = []
        current_node = root_node

        for ev in events:
            ev_type = ev.get("type")
            fidx = ev.get("frame", 0)
            at = ev.get("at", 0)

            if ev_type == "O":
                key = fidx
                if key not in current_node["children"]:
                    n2, loc = frame_name(fidx)
                    current_node["children"][key] = {
                        "name": n2,
                        "location": loc,
                        "samples": 0,
                        "children": {},
                    }
                child_node = current_node["children"][key]
                stack.append((fidx, current_node, at))
                current_node = child_node

            elif ev_type == "C":
                if stack:
                    fidx_open, parent_node, enter_at = stack.pop()
                    duration = at - enter_at
                    current_node["samples"] += duration
                    current_node = parent_node
    else:
        raise ValueError(f"Unknown speedscope profile type: {ptype!r}")

    def convert(node: dict) -> dict:
        """Recursively convert internal tree format to final output format."""
        children = [convert(v) for v in node["children"].values()]
        children.sort(key=lambda c: c["samples"], reverse=True)
        total_samples = node["samples"]
        root_s = root_node["samples"] if root_node["samples"] else 1
        return {
            "name": node["name"],
            "location": node["location"],
            "samples": total_samples,
            "pct": 100.0 * total_samples / root_s,
            "_self": 0,  # will be filled by annotate_self
            "children": children,
        }

    root = convert(root_node)
    root["pct"] = 100.0
    annotate_self(root)
    return root


# ---------------------------------------------------------------------------
# 4. Dispatch: auto-detect input format
# ---------------------------------------------------------------------------

def parse_input(path: str) -> dict:
    """Auto-detect input format by extension and return root tree node."""
    if path.lower().endswith(".json"):
        return parse_speedscope(path)
    # Default: SVG
    frames = parse_svg(path)
    if not frames:
        print("ERROR: no frames parsed from SVG", file=sys.stderr)
        sys.exit(1)
    return build_tree(frames)


# ---------------------------------------------------------------------------
# 5. Collect all nodes for self-time ranking
# ---------------------------------------------------------------------------

def collect_all_nodes(node: dict) -> list[dict]:
    """Return a flat list of all nodes in the tree."""
    result = [node]
    for child in node["children"]:
        result.extend(collect_all_nodes(child))
    return result


# ---------------------------------------------------------------------------
# 6. Formatters
# ---------------------------------------------------------------------------

def _fmt_node_label(node: dict, root_samples: int) -> str:
    pct = 100.0 * node["samples"] / root_samples if root_samples else 0.0
    loc = f" ({node['location']})" if node["location"] else ""
    return f"{node['samples']} ({pct:.1f}%) {node['name']}{loc}"


def render_md_tree(
    node: dict,
    root_samples: int,
    min_pct: float,
    max_depth: int,
    max_children: int,
    depth: int = 0,
) -> list[str]:
    """Markdown nested bullet list (no box-drawing chars)."""
    lines = []
    indent = "  " * depth
    pct = 100.0 * node["samples"] / root_samples if root_samples else 0.0
    loc = f" ({node['location']})" if node["location"] else ""
    is_leaf = not node["children"]
    leaf_marker = "  *[leaf]*" if is_leaf else ""
    lines.append(
        f"{indent}- **{node['samples']} ({pct:.1f}%)**"
        f" {node['name']}{loc}{leaf_marker}"
    )

    if max_depth > 0 and depth >= max_depth - 1:
        if node["children"]:
            lines.append(f"{indent}  - *(depth limit)*")
        return lines

    visible = [
        c for c in node["children"]
        if 100.0 * c["samples"] / root_samples >= min_pct
    ]
    hidden = len(node["children"]) - len(visible)

    if max_children > 0:
        trimmed = visible[max_children:]
        visible = visible[:max_children]
        hidden += len(trimmed)

    for child in visible:
        lines.extend(
            render_md_tree(child, root_samples, min_pct, max_depth, max_children, depth + 1)
        )

    if hidden > 0:
        lines.append(f"{indent}  - *({hidden} more children below threshold)*")

    return lines


def render_text_tree(
    node: dict,
    root_samples: int,
    min_pct: float,
    max_depth: int,
    max_children: int,
    depth: int = 0,
) -> list[str]:
    """Plain ASCII indented tree, no box-drawing characters."""
    lines = []
    indent = "  " * depth
    pct = 100.0 * node["samples"] / root_samples if root_samples else 0.0
    loc = f" ({node['location']})" if node["location"] else ""
    is_leaf = not node["children"]
    leaf_marker = "  [leaf]" if is_leaf else ""
    lines.append(
        f"{indent}{node['samples']} ({pct:.1f}%) {node['name']}{loc}{leaf_marker}"
    )

    if max_depth > 0 and depth >= max_depth - 1:
        if node["children"]:
            lines.append(f"{indent}  (depth limit)")
        return lines

    visible = [
        c for c in node["children"]
        if 100.0 * c["samples"] / root_samples >= min_pct
    ]
    hidden = len(node["children"]) - len(visible)

    if max_children > 0:
        trimmed = visible[max_children:]
        visible = visible[:max_children]
        hidden += len(trimmed)

    for child in visible:
        lines.extend(
            render_text_tree(child, root_samples, min_pct, max_depth, max_children, depth + 1)
        )

    if hidden > 0:
        lines.append(f"{indent}  ({hidden} more children below threshold)")

    return lines


def node_to_dict(node: dict) -> dict:
    """Serialize tree node to plain dict for JSON output (strip internal fields)."""
    return {
        "name": node["name"],
        "location": node["location"],
        "samples": node["samples"],
        "pct": node["pct"],
        "_self": node.get("_self", 0),
        "children": [node_to_dict(c) for c in node["children"]],
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="py-spy SVG flamegraph or speedscope JSON -> Markdown/text/JSON tree"
    )
    parser.add_argument(
        "path",
        help="Path to py-spy/inferno SVG flamegraph or speedscope JSON profile"
    )
    parser.add_argument(
        "--format", choices=["md", "text", "json"], default="md",
        help="Output format (default: md)"
    )
    parser.add_argument(
        "--min-pct", type=float, default=1.0,
        help="Hide branches below this %% of root (default: 1.0)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Max tree depth to display (0 = unlimited, default: 10)"
    )
    parser.add_argument(
        "--max-children", type=int, default=8,
        help="Max children per node (0 = unlimited, default: 8)"
    )
    parser.add_argument(
        "--top-leaves", type=int, default=30,
        help="Number of top frames by self-time to show in table (default: 30)"
    )
    args = parser.parse_args()

    root = parse_input(args.path)
    root_samples = root["samples"]

    # Collect all nodes for self-time ranking (not just leaves)
    all_nodes = collect_all_nodes(root)
    frames_by_self = sorted(
        (n for n in all_nodes if n.get("_self", 0) > 0),
        key=lambda n: n["_self"],
        reverse=True,
    )

    # For legacy stats: count how many frames were parsed (SVG path populates
    # the flat frame list; for JSON we count all nodes instead)
    total_frame_count = len(all_nodes)

    basename = os.path.basename(args.path)

    if args.format == "json":
        print(json.dumps(node_to_dict(root), indent=2))
        return

    if args.format == "text":
        print(f"# py-spy flamegraph: {basename}")
        print(f"  Total samples : {root_samples:,}")
        print(f"  Frames        : {total_frame_count:,}")
        print()
        print("## Top frames by self-time")
        top_n = args.top_leaves
        for node in frames_by_self[:top_n]:
            self_samples = node.get("_self", 0)
            pct = 100.0 * self_samples / root_samples if root_samples else 0.0
            loc = f"  ({node['location']})" if node["location"] else ""
            print(f"  {self_samples:>6} ({pct:5.1f}%)  {node['name']}{loc}")
        print()
        print("## Call tree (top branches)")
        lines = render_text_tree(
            root, root_samples,
            args.min_pct, args.max_depth, args.max_children,
        )
        print("\n".join(lines))
        return

    # --- Markdown ---
    lines = []
    lines.append(f"## py-spy flamegraph: {basename}")
    lines.append("")
    lines.append(f"- Total samples: {root_samples:,}")
    lines.append(f"- Frames: {total_frame_count:,}")
    lines.append("")

    lines.append("### Top frames by self-time")
    lines.append("")
    lines.append("| self | % of root | name | location |")
    lines.append("|---:|---:|---|---|")
    top_n = args.top_leaves
    for node in frames_by_self[:top_n]:
        self_samples = node.get("_self", 0)
        pct = 100.0 * self_samples / root_samples if root_samples else 0.0
        loc = node["location"] if node["location"] else "-"
        lines.append(f"| {self_samples:,} | {pct:.1f}% | {node['name']} | {loc} |")
    lines.append("")

    lines.append("### Call tree (top branches, depth limited)")
    lines.append("")
    tree_lines = render_md_tree(
        root, root_samples,
        args.min_pct, args.max_depth, args.max_children,
    )
    lines.extend(tree_lines)
    lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
