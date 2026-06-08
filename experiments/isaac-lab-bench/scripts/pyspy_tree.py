#!/usr/bin/env python3
"""py-spy / inferno SVG flamegraph -> Markdown tree ビューア

SVG フレームグラフはブラウザなしでは読めないため、CLI で Markdown ツリーとして
出力し、そのままドキュメントやチャットに貼り付けられるようにする。

対応 SVG 形式: inferno が生成する inverted=true (icicle) フレームグラフ。
各フレームは <g><title>NAME (file:line) (N samples, X.YZ%)</title><rect ... fg:x="X" fg:w="W"/></g>
の形式。

出力モード (--format):
  md   (デフォルト) : Markdown ネスト箇条書きツリー + リーフテーブル
  text : インデント ASCII ツリー (ボックス描画文字なし)
  json : 生ツリーを JSON で出力 (ネスト dict)

使用例
------
  python pyspy_tree.py results/pyspy/pyspy-4xl-full.svg
  python pyspy_tree.py results/pyspy/pyspy-4xl-full.svg --format md --min-pct 1.0 --max-depth 8 --max-children 6 --top-leaves 30
  python pyspy_tree.py results/pyspy/pyspy-4xl-full.svg --format json | python -m json.tool | head -60
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
    """SVG ファイルからフレームリストを返す (x, w, y, name, location, samples, pct)."""
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
# 2. Build tree
# ---------------------------------------------------------------------------

def build_tree(frames: list[dict]) -> dict:
    """
    parent 検出: F2 の親 = y < F2.y かつ [F2.x, F2.x+F2.w] が親の範囲内に含まれる
    フレームのうち y が最大のもの (= 最も近い祖先)。

    returns root node dict with children list.
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
    return root


# ---------------------------------------------------------------------------
# 3. Collect leaves
# ---------------------------------------------------------------------------

def collect_leaves(node: dict) -> list[dict]:
    if not node["children"]:
        return [node]
    leaves = []
    for child in node["children"]:
        leaves.extend(collect_leaves(child))
    return leaves


# ---------------------------------------------------------------------------
# 4. Formatters
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
        "children": [node_to_dict(c) for c in node["children"]],
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="py-spy / inferno SVG flamegraph -> Markdown/text/JSON tree"
    )
    parser.add_argument("svg", help="Path to py-spy/inferno SVG flamegraph")
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
        help="Number of top leaf frames to show in table (default: 30)"
    )
    args = parser.parse_args()

    frames = parse_svg(args.svg)
    if not frames:
        print("ERROR: no frames parsed from SVG", file=sys.stderr)
        sys.exit(1)

    root = build_tree(frames)
    root_samples = root["samples"]
    leaves = collect_leaves(root)
    leaves_sorted = sorted(leaves, key=lambda l: l["samples"], reverse=True)
    leaf_sum = sum(l["samples"] for l in leaves)
    # Approximate thread count: if leaf_sum > root_samples, multi-threaded
    # (py-spy aggregates all threads)
    thread_hint = max(1, round(leaf_sum / root_samples)) if root_samples else 1

    basename = os.path.basename(args.svg)

    if args.format == "json":
        print(json.dumps(node_to_dict(root), indent=2))
        return

    if args.format == "text":
        print(f"# py-spy flamegraph: {basename}")
        print(f"  Total samples : {root_samples:,}")
        print(f"  Frames        : {len(frames):,}")
        print(f"  Approx threads: {thread_hint}")
        print()
        print("## Top leaves by self-time")
        top_n = args.top_leaves
        for leaf in leaves_sorted[:top_n]:
            pct = 100.0 * leaf["samples"] / root_samples if root_samples else 0.0
            loc = f"  ({leaf['location']})" if leaf["location"] else ""
            print(f"  {leaf['samples']:>6} ({pct:5.1f}%)  {leaf['name']}{loc}")
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
    lines.append(f"- Frames: {len(frames):,}")
    multi_label = " (multi-threaded)" if thread_hint > 1 else ""
    lines.append(f"- Sampling threads: {thread_hint}{multi_label}")
    lines.append("")

    lines.append("### Top leaves by self-time")
    lines.append("")
    lines.append("| samples | % of root | name | location |")
    lines.append("|---:|---:|---|---|")
    top_n = args.top_leaves
    for leaf in leaves_sorted[:top_n]:
        pct = 100.0 * leaf["samples"] / root_samples if root_samples else 0.0
        loc = leaf["location"] if leaf["location"] else "-"
        lines.append(f"| {leaf['samples']:,} | {pct:.1f}% | {leaf['name']} | {loc} |")
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
