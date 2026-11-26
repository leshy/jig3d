"""Standalone `jig` CLI for building, inspecting, rendering, and measuring procedural 3D models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from . import glb as glb_mod
from . import metrics as metrics_mod
from . import render as render_mod
from . import words as words_mod
from .materials import DEFAULT_BY_NAME, THEMES
from .scene import load_script, run_script, script_variants

_TEMPLATE = '''"""Short description of what this model is, its size, and how it works.

Front +Z, up +Y, metres.
"""

from jig import *  # noqa: F403

VARIANTS = {{"{name}": {{}}}}
LAID = True


def model(p):
    # 1. Primary silhouette (avoid bare boxes: use cut corners, drafts, or orthographic body())
    base = p.put(extrude(rect(0.42, 0.36, cut=0.05), 0.14, draft=0.02).chamfer(0.012).move(0, 0.18, 0), "body2")

    # 2. Secondary form: recessed panel + bezel on front face
    front = base.face("front")
    bezel = p.put(extrude(rect(0.28, 0.20, cut=0.03), 0.018).chamfer(0.005, "out").on(front, sink=0.005), "trim")
    screen = p.inset(bezel.face("out"), 0.022, 0.014, role="screen", mouth=0.006)
    p.mark(screen, rect(0.18, 0.010).at(0, 0.02), "glow")

    # 3. Functional details: corner fasteners + named attachment slot
    p.put(bolt(0.018, 0.005).on(front.corners(inset=0.035)), "panel")
    p.slot("screen", screen)
'''


def _parse_vec3(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected X,Y,Z, got {s!r}")
    return (parts[0], parts[1], parts[2])


def cmd_build(args: argparse.Namespace) -> int:
    t0 = time.time()
    script = Path(args.script)
    mod = load_script(script)
    variants = script_variants(mod, script.stem)
    targets = list(variants) if args.all else [args.variant or next(iter(variants))]
    for vname in targets:
        m, resolved, _ = run_script(script, variant=vname, palette=args.palette, laid=(False if args.welded else None))
        pos, nor, pnt, tri = m.build_mesh()
        data = glb_mod.pack_glb(
            pos,
            nor,
            pnt,
            tri,
            m.palette,
            name=resolved,
            slots=m.named,
            digest=m.digest(),
            vertex_colors=args.vertex_colors,
        )
        if args.out and len(targets) == 1:
            out_path = Path(args.out)
        else:
            out_dir = Path(args.out) if args.out else Path("out")
            out_path = out_dir / f"{resolved}.glb"
        nbytes = glb_mod.write_glb(out_path, data)
        nums = metrics_mod.measure(pos[tri])
        warns = m.warnings()
        print(
            f"ok  {out_path}  ({nbytes / 1024:.1f} KB, {len(tri):,} tris, {time.time() - t0:.2f}s)  "
            f"{metrics_mod.format_line(nums)}"
        )
        for (_, ln), text in warns:
            print(f"  warning :{ln} {text}")
        if m.triangles and len(tri) > m.triangles:
            print(f"  budget exceeded: {len(tri):,} > {m.triangles:,} triangles")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    pal_list = args.palettes.split(",") if args.palettes else None
    _out, summary = render_mod.render_view(
        args.script,
        out_path=args.out,
        variant=args.variant,
        palette=args.palette,
        palettes=pal_list,
        themes=args.themes,
        ats=args.at or None,
        beside=args.beside or None,
        scale=args.scale,
        wireframe=args.wireframe,
        jobs=args.jobs,
    )
    for ln in summary:
        print(ln)
    return 0


def cmd_look(args: argparse.Namespace) -> int:
    views = args.views.split(",") if args.views else None
    out = render_mod.render_look(
        args.models,
        out_path=args.out,
        views=views,
        az=args.az,
        el=args.el,
        ortho=args.ortho,
        size=args.size,
        slots=args.slots,
        scale=args.scale,
        palette=args.palette,
    )
    print(out)
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    m, vname, _ = run_script(args.script, variant=args.variant, palette=args.palette)
    print(f"=== {vname} ===")
    for ln in m.describe_lines(line=args.line):
        print(ln)
    pos, _, _, tri = m.build_mesh()
    nums = metrics_mod.measure(pos[tri])
    print(metrics_mod.format_line(nums))
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    m, vname, _ = run_script(args.script, variant=args.variant)
    items = sorted(m.cost().items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in items)
    print(f"{vname}: {total:,} triangles")
    src_cache: dict[str, list[str]] = {}
    for (f, ln), v in items:
        if f not in src_cache:
            src_cache[f] = Path(f).read_text().splitlines() if Path(f).is_file() else []
        code = src_cache[f][ln - 1].strip()[:80] if 0 < ln <= len(src_cache[f]) else ""
        pct = 100.0 * v / max(total, 1)
        print(f"  {v:6d} ({pct:4.1f}%)  :{ln:<4d} {code}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    for p in args.files:
        fp = Path(p)
        if fp.suffix == ".py":
            m, vname, _ = run_script(fp)
            pos, _, _, tri = m.build_mesh()
            nums = metrics_mod.measure(pos[tri])
            label = vname
        else:
            nums = metrics_mod.measure_file(fp)
            label = fp.name
        w, h, d = nums["size"]
        print(f"{label}: {w:.3f} x {h:.3f} x {d:.3f} m  {nums['triangles']:,} tris  {metrics_mod.format_line(nums)}")
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    labels = args.labels.split(",") if args.labels else None
    out = render_mod.join_sheet(
        args.pics,
        out_path=args.out,
        cols=args.cols,
        labels=labels,
        crop=args.crop,
        width=args.width,
        lift=args.lift,
    )
    print(out)
    return 0


def cmd_words(args: argparse.Namespace) -> int:
    print(words_mod.page(args.filter))
    return 0


def cmd_materials(_args: argparse.Namespace) -> int:
    print("Built-in PBR materials (or pass '#rrggbb', (r, g, b), or Material(...)):")
    for name, mat in DEFAULT_BY_NAME.items():
        r, g, b = mat.rgb
        flags = []
        if mat.emissive:
            flags.append("emissive")
        if mat.alpha < 1.0:
            flags.append(f"alpha={mat.alpha:.2f}")
        extra = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {name:<14} #{r:02x}{g:02x}{b:02x}  metal={mat.metallic:.2f}  rough={mat.roughness:.2f}{extra}")
    print("\nBuilt-in colour themes (use with --theme NAME or --themes):")
    for tname, overrides in THEMES.items():
        summary = "built-in defaults" if not overrides else " ".join(f"{k}={v}" for k, v in list(overrides.items())[:5]) + " ..."
        print(f"  {tname:<14} {summary}")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    dest = Path(args.path)
    if dest.suffix != ".py":
        dest = dest.with_suffix(".py")
    if dest.exists() and not args.force:
        sys.exit(f"refusing to overwrite existing {dest} (pass --force)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_TEMPLATE.format(name=dest.stem))
    print(f"created {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jig",
        description="Agent-native procedural 3D solid & SubD modelling workshop (.py -> .glb + f3d inspection)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="Compile a .py model script into deterministic .glb file(s)")
    p.add_argument("script", help="Path to model .py script")
    p.add_argument("-o", "--out", help="Output .glb path (or directory with --all)")
    p.add_argument("--variant", help="Variant name from VARIANTS/PIECES")
    p.add_argument("--all", action="store_true", help="Build all variants defined in the script")
    p.add_argument("--theme", "--palette", dest="palette", help="Built-in theme name (default, light, dark, industrial, warm) or JSON palette override file")
    p.add_argument("--welded", action="store_true", help="Force welded CSG instead of laid shells")
    p.add_argument("--vertex-colors", action="store_true", help="Emit single-primitive COLOR_0 glTF instead of PBR materials")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("view", help="Render a multi-angle workshop contact sheet via f3d")
    p.add_argument("script", help="Path to model .py script")
    p.add_argument("-o", "--out", help="Output PNG path (default out/<variant>.png)")
    p.add_argument("--variant", help="Variant name")
    p.add_argument("--theme", "--palette", dest="palette", help="Built-in theme name (default, light, dark, industrial, warm) or JSON palette file")
    p.add_argument("--themes", action="store_true", help="Add a bottom comparison strip across all built-in themes")
    p.add_argument("--palettes", help="Comma-separated theme names and/or JSON palette files to compare in an extra bottom strip")
    p.add_argument("--at", action="append", type=_parse_vec3, help="Close-up focus X,Y,Z (repeatable, up to 2)")
    p.add_argument("--beside", action="append", help="Reference/family .py or .glb model(s) to show alongside (repeatable)")
    p.add_argument("--scale", action="store_true", help="Show a 1.75 m human silhouette and 1 m bar for real-world scale")
    p.add_argument("--wireframe", action="store_true", help="Include a topology wireframe panel")
    p.add_argument("--jobs", type=int, default=2, help="Parallel f3d workers (default 2)")
    p.set_defaults(fn=cmd_view)

    p = sub.add_parser("look", help="Render one or more .glb/.gltf/.py models side by side at true scale")
    p.add_argument("models", nargs="+", help="Model file(s) (.glb, .gltf, or .py)")
    p.add_argument("--views", help="Comma-separated presets (3q,front,side,back,top,low,eye,under) or AZ:EL")
    p.add_argument("--az", type=float, help="Custom azimuth in degrees (0=+Z, +90=+X)")
    p.add_argument("--el", type=float, help="Custom elevation in degrees (-90..90)")
    p.add_argument("--ortho", action="store_true", help="Use orthographic camera")
    p.add_argument("--size", type=int, default=900, help="Tile pixel size (default 900)")
    p.add_argument("--slots", action="store_true", help="Draw declared slot:<name> markers")
    p.add_argument("--scale", action="store_true", help="Include a 1.75 m human silhouette and 1 m bar for real-world scale")
    p.add_argument("--theme", "--palette", dest="palette", help="Built-in theme name or JSON palette override for .py inputs")
    p.add_argument("-o", "--out", help="Output PNG path")
    p.set_defaults(fn=cmd_look)

    p = sub.add_parser("describe", help="Print size, triangles, material breakdown, warnings, slots, and part faces")
    p.add_argument("script", help="Path to model .py script")
    p.add_argument("line", nargs="?", type=int, help="Optional source line number to inspect individual face dimensions")
    p.add_argument("--variant", help="Variant name")
    p.add_argument("--theme", "--palette", dest="palette", help="Built-in theme name or JSON palette override file")
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("cost", help="Show triangle cost per source line, sorted descending")
    p.add_argument("script", help="Path to model .py script")
    p.add_argument("--variant", help="Variant name")
    p.set_defaults(fn=cmd_cost)

    p = sub.add_parser("metrics", help="Measure soft/deep/plain form quality metrics on .glb/.gltf/.py files")
    p.add_argument("files", nargs="+", help="File(s) to measure")
    p.set_defaults(fn=cmd_metrics)

    p = sub.add_parser("sheet", help="Join multiple PNG images into one labelled contact sheet")
    p.add_argument("pics", nargs="+", help="Image paths (optionally PATH@WxH+X+Y)")
    p.add_argument("--cols", type=int, help="Columns per row")
    p.add_argument("--labels", help="Comma-separated labels (use '-' to keep filename)")
    p.add_argument("--crop", help="Default crop WxH+X+Y")
    p.add_argument("--width", type=int, help="Resize each tile to this width")
    p.add_argument("--lift", type=float, default=1.0, help="Shadow gamma lift")
    p.add_argument("-o", "--out", default="out/sheet.png", help="Output PNG path")
    p.set_defaults(fn=cmd_sheet)

    p = sub.add_parser("words", help="Print the live modelling vocabulary reference with measured triangle counts")
    p.add_argument("filter", nargs="?", help="Optional substring filter")
    p.set_defaults(fn=cmd_words)

    p = sub.add_parser("materials", help="List built-in PBR materials")
    p.set_defaults(fn=cmd_materials)

    p = sub.add_parser("new", help="Create a starter model script")
    p.add_argument("path", help="Destination .py path")
    p.add_argument("--force", action="store_true", help="Overwrite if file exists")
    p.set_defaults(fn=cmd_new)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as exc:
        print(f"jig error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
