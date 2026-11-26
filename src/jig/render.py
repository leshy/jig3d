"""Fast headless 3D rendering, multi-view workshop sheets, side-by-side `look`, and contact `sheet` via f3d + Pillow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from . import glb as glb_mod
from . import metrics as metrics_mod
from .materials import THEMES, Palette
from .scene import Model, run_script
from .words import block, lathe

VIEWS: dict[str, tuple[float, float, float]] = {
    "3q": (1.0, 0.55, 1.2),
    "back": (-1.0, 0.55, -1.2),
    "side": (1.0, 0.15, 0.0),
    "front": (0.0, 0.15, 1.0),
    "top": (0.0, 1.0, 0.08),
    "low": (1.0, 0.08, 1.2),
    "eye": (0.0, 0.15, 1.0),
    "under": (0.7, -0.45, 1.2),
}
EYE_HEIGHT = 1.7
DEFAULT_BG = "0.2,0.2,0.22"
WORKSHOP_BG = "0.78,0.78,0.80"


def _require_f3d() -> str:
    exe = shutil.which("f3d")
    if not exe:
        raise RuntimeError("f3d not found on PATH; install f3d (https://f3d.app) to render previews")
    return exe


def _font(size: int):
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if os.path.isfile(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default(size=size)


def _camera_up(d):
    n = math.hypot(*d) or 1.0
    if abs(d[1] / n) > 0.999:
        return (0.0, 0.0, -1.0) if d[1] > 0 else (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


def angle_dir(az: float, el: float) -> tuple[float, float, float]:
    """Unit direction from focal point toward camera for azimuth/elevation in degrees (`az=0` is `+Z`)."""
    if not (-90.0 <= el <= 90.0):
        raise ValueError(f"elevation {el:g} must be in -90..90")
    ar, er = math.radians(az), math.radians(el)
    return (math.cos(er) * math.sin(ar), math.sin(er), math.cos(er) * math.cos(ar))


def parse_view(spec: str) -> tuple[str, tuple[float, float, float], str]:
    if spec in VIEWS:
        return (spec, VIEWS[spec], "eye" if spec == "eye" else "preset")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) == 2:
            az, el = float(parts[0]), float(parts[1])
            return (f"az {az:g} el {el:g}", angle_dir(az, el), "angle")
    raise ValueError(f"unknown view {spec!r} (valid: {', '.join(VIEWS)} or AZ:EL)")


def _f3d_render(
    scene_path: Path,
    out_png: Path,
    w: int,
    h: int,
    extra_args: list[str],
    *,
    bg: str = DEFAULT_BG,
    grid: bool = False,
):
    f3d = _require_f3d()
    cmd = [
        f3d,
        str(scene_path),
        f"--output={out_png}",
        f"--resolution={w},{h}",
        f"--background-color={bg}",
        "--axis=0",
        "--filename=0",
        "--light-intensity=2.4",
        "--anti-aliasing",
        "--verbose=quiet",
    ]
    if grid:
        cmd += ["--grid", "--grid-absolute", "--grid-unit=1", "--grid-reflection=0"]
    else:
        cmd += ["--grid=0"]
    cmd += extra_args
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not out_png.is_file():
        raise RuntimeError(f"f3d failed ({res.returncode}): {(res.stdout + res.stderr)[-400:]}")


def _glowing(pic: Image.Image) -> Image.Image:
    """Add a subtle 2D bloom halo around bright emissive pixels."""
    a = np.asarray(pic).astype(np.int16)
    bright = ((a.max(2) >= 235) & ((a.max(2) - a.min(2)) > 40)) | (a.min(2) >= 248)
    glow = Image.fromarray(np.where(bright[..., None], a, 0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(7))
    return ImageChops.add(pic, glow.point(lambda v: int(v * 0.75)))


def _cam_args(pos, at, angle=30.0) -> list[str]:
    return [
        f"--camera-position={','.join(f'{c:.3f}' for c in pos)}",
        f"--camera-focal-point={','.join(f'{c:.3f}' for c in at)}",
        f"--camera-view-angle={angle}",
        "--camera-view-up=0,1,0",
    ]


def _upright_frame(eye, at, angle, w, h):
    """Tilt-shift camera framing so vertical lines stay upright when pitching down/up."""
    d = np.subtract(at, eye)
    flat = float(np.hypot(d[0], d[2]))
    pitch, half = np.arctan2(-d[1], flat), np.radians(angle) / 2
    beta = abs(pitch) + half
    if abs(pitch) < np.radians(5) or beta > np.radians(75) or flat < 1e-6:
        return w, h, _cam_args(eye, at, angle), None
    f = (h / 2) / np.tan(half)
    y0, y1 = f * np.tan(pitch - half), f * np.tan(pitch + half)
    hc = y1 - y0
    w2, h2 = int(round(w * hc / h)), int(round(2 * f * np.tan(beta))) + 2
    level = np.add(eye, (d[0], 0.0, d[2]))
    opts = _cam_args(eye, level, float(np.degrees(2 * np.arctan((h2 / 2) / f))))
    return w2, h2, opts, (h2 / 2 + y0, hc, w, h)


def _scale_marks_glb(x: float, z: float) -> tuple[dict, list[bytes]]:
    """Build a 1.75 m human reference silhouette + 1.0 m striped scale bar at `(x, 0, z)`."""
    p = Model(laid=True)
    # 1.75 m human figure silhouette (legs, torso, head)
    for sx in (-0.09, 0.09):
        p.put(block(0.11, 0.85, 0.13).move(x + sx, 0.425, z - 0.065), "body2")
    p.put(block(0.40, 0.62, 0.20).chamfer(0.03).move(x, 1.16, z - 0.10), "body2")
    p.put(lathe([(0.0, 0.10), (0.22, 0.10)], 8).turn("x", -90).move(x, 1.53, z), "body2")
    # 1.0 m scale bar in 10 cm alternating dark/light bands
    bx = x + 0.42
    for k in range(10):
        p.put(block(0.05, 0.10, 0.05).move(bx, k * 0.10 + 0.05, z - 0.025), "trim" if k % 2 == 0 else "panel")
    pos, nor, pnt, tri = p.build_mesh()
    raw = glb_mod.pack_glb(pos, nor, pnt, tri, p.palette, name="scale_marks")
    doc, bin_ = glb_mod.read_glb_parts(raw)
    return doc, [bin_]


def _slot_markers_glb(slots: list[tuple]) -> tuple[dict, list[bytes]] | None:
    """Build small green axis/normal arrows for declared slots."""
    if not slots:
        return None
    p = Model(laid=True)
    for _, at, normal, sz in slots:
        n = np.asarray(normal, float)
        ln = float(np.linalg.norm(n))
        n = n / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
        # Build small marker pin along normal
        from ._places import frame

        fr = frame(at=at, z=n)
        p.put(lathe([(0.0, 0.012), (0.08, 0.004)], 6).on(fr), "led_green")
        w, h = sz if len(sz) == 2 else (0.0, 0.0)
        if w > 0.01 and h > 0.01:
            from .words import extrude, rect

            p.put(extrude(rect(w, h).band(min(0.008, min(w, h) * 0.15)), 0.004).on(fr), "led_green")
    pos, nor, pnt, tri = p.build_mesh()
    raw = glb_mod.pack_glb(pos, nor, pnt, tri, p.palette, name="slots")
    doc, bin_ = glb_mod.read_glb_parts(raw)
    return doc, [bin_]


def render_view(
    script_path: str | os.PathLike,
    out_path: str | os.PathLike | None = None,
    *,
    variant: str | None = None,
    palette: str | os.PathLike | dict | None = None,
    palettes: list[str | os.PathLike] | None = None,
    themes: bool = False,
    ats: list[tuple[float, float, float]] | None = None,
    beside: list[str | os.PathLike] | None = None,
    scale: bool = False,
    wireframe: bool = False,
    jobs: int = 2,
) -> tuple[Path, list[str]]:
    """Render the workshop inspection sheet for a model `.py` script.

    Default 3-row sheet:
      - Row 1: 4x 3/4 angled views (front-left, back-left, back-right, front-right)
      - Row 2: Front (+Z) & Back (-Z) ortho elevations | Perspective hero view (with glow halo)
      - Row 3: Left (-X), Right (+X), Top (+Y), and Bottom (-Y) orthographic views
    Optional toggles: `--scale` (1.75 m human + 1 m bar), `--wireframe` (topology edges panel),
    `--themes` (built-in theme comparison strip), `--palettes` (custom theme/JSON comparison strip),
    `--at`/`CLOSE` (macro close-ups), `--beside` (reference comparison).
    Returns `(png_path, summary)`.
    """
    t0 = time.time()
    m, vname, mod = run_script(script_path, variant=variant, palette=palette)
    pos, nor, pnt, tri = m.build_mesh()
    t_build = time.time() - t0

    lo, hi = pos.min(0).astype(float), pos.max(0).astype(float)
    size, mid = hi - lo, (lo + hi) / 2
    compact = float(size.max()) < 0.5
    nums = metrics_mod.measure(pos[tri])
    metrics_str = metrics_mod.format_line(nums)

    # Material breakdown
    by_mat: dict[str, int] = {}
    for q in pnt[tri[:, 0]]:
        nm = m.palette.name_of(int(q))
        by_mat[nm] = by_mat.get(nm, 0) + 1
    costly = "  ".join(f"{k} {v:,}" for k, v in sorted(by_mat.items(), key=lambda kv: -kv[1])[:8])

    main_bytes = glb_mod.pack_glb(pos, nor, pnt, tri, m.palette, name=vname, slots=m.named, digest=m.digest())
    main_doc, main_bin = glb_mod.read_glb_parts(main_bytes)

    out = Path(out_path) if out_path else Path("out") / f"{vname}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jig-view-") as tmp:
        tmp_p = Path(tmp)

        # 1. Four 3/4 corners side-by-side (orthographic, full-frame without scale figure)
        span = float(np.hypot(size[0], size[2])) + max(0.06, float(size.max()) * 0.35)
        right = np.array([1.0, 0.0, 1.0]) / math.sqrt(2)
        turns_items = []
        for k, yaw in enumerate((0, 90, 180, 270)):
            shift = k * span * right
            turns_items.append((main_doc, [main_bin], (-mid[0] + shift[0], -lo[1], -mid[2] + shift[2]), yaw))
        turns_glb = tmp_p / "turns.glb"
        turns_glb.write_bytes(glb_mod.merge_glbs(turns_items))

        # 2. Front (+Z) and Back (-Z) orthographic elevations (optional 1.75 m scale figure)
        elev_gap = size[0] + (max(0.25, float(size[0]) * 0.45) if scale else max(0.06, float(size[0]) * 0.35))
        elev_items = [
            (main_doc, [main_bin], (-mid[0], -lo[1], -mid[2]), 0.0),
            (main_doc, [main_bin], (-mid[0] + elev_gap, -lo[1], -mid[2]), 180.0),
        ]
        if scale:
            sm_doc, sm_bufs = _scale_marks_glb(0.0, 0.0)
            elev_items.append((sm_doc, sm_bufs, (elev_gap + size[0] / 2 + 0.55, 0.0, 0.0), 0.0))
        elev_glb = tmp_p / "elev.glb"
        elev_glb.write_bytes(glb_mod.merge_glbs(elev_items))

        # 3. Perspective hero view (framed cleanly around object + optional 1.75 m scale figure)
        eye_items = [(main_doc, [main_bin], (-mid[0], -lo[1], -mid[2]), 0.0)]
        if scale:
            sm_doc, sm_bufs = _scale_marks_glb(float(size[0]) / 2 + 0.55, -0.15)
            eye_items.append((sm_doc, sm_bufs, (0.0, 0.0, 0.0), 0.0))
        eye_bytes = glb_mod.merge_glbs(eye_items)
        eye_glb = tmp_p / "eye.glb"
        eye_glb.write_bytes(eye_bytes)
        eye_doc, _ = glb_mod.read_glb_parts(eye_bytes)
        s_lo, s_hi = glb_mod.bounds_of_gltf(eye_doc)
        s_mid = (s_lo + s_hi) / 2.0
        s_sz = s_hi - s_lo
        eye_w, eye_h = (450 if wireframe else 900), 450
        eye_fov = 38.0
        half_v = math.radians(eye_fov / 2.0)
        aspect = eye_w / eye_h
        dist_v = max(float(s_sz[1]) * 0.55, 0.05) / math.tan(half_v)
        dist_h = max(math.hypot(float(s_sz[0]), float(s_sz[2])) * 0.52, 0.05) / (math.tan(half_v) * aspect)
        eye_dist = max(dist_v, dist_h) * 1.28 + float(s_sz[2]) * 0.45
        eye_y = float(getattr(mod, "EYE", EYE_HEIGHT if scale else float(s_mid[1]) + float(s_sz[1]) * 0.28))
        eye_pos = (float(s_mid[0]) + eye_dist * 0.36, eye_y, float(s_mid[2]) + eye_dist * 0.93)
        ew, eh, eopts, ecrop = _upright_frame(eye_pos, s_mid, eye_fov, eye_w, eye_h)

        # 4. Single centred model for ortho Left / Right / Top / Bottom, optional wireframe & close-ups
        plain_glb = tmp_p / "plain.glb"
        plain_glb.write_bytes(glb_mod.merge_glbs([(main_doc, [main_bin], (0.0, -lo[1], 0.0), 0.0)]))

        render_jobs = [
            ("turns", turns_glb, 1800, 450, ["--camera-direction=1,-0.55,-1", "--camera-orthographic", "--camera-zoom-factor=1.55"], WORKSHOP_BG, False, None, False),
            ("elev", elev_glb, 900, 450, ["--camera-direction=0,0,-1", "--camera-orthographic", "--camera-zoom-factor=0.86"], WORKSHOP_BG, False, None, False),
            ("eye", eye_glb, ew, eh, eopts, WORKSHOP_BG, False, ecrop, True),
            ("left", plain_glb, 450, 450, ["--camera-direction=1,0,0", "--camera-view-up=0,1,0", "--camera-orthographic", "--camera-zoom-factor=0.82"], WORKSHOP_BG, False, None, False),
            ("right", plain_glb, 450, 450, ["--camera-direction=-1,0,0", "--camera-view-up=0,1,0", "--camera-orthographic", "--camera-zoom-factor=0.82"], WORKSHOP_BG, False, None, False),
            ("top", plain_glb, 450, 450, ["--camera-direction=0,-1,-0.001", "--camera-view-up=0,0,-1", "--camera-orthographic", "--camera-zoom-factor=0.82"], WORKSHOP_BG, False, None, False),
            ("bottom", plain_glb, 450, 450, ["--camera-direction=0,1,-0.001", "--camera-view-up=0,0,1", "--camera-orthographic", "--camera-zoom-factor=0.82"], WORKSHOP_BG, False, None, False),
        ]
        if wireframe:
            render_jobs.append(
                ("wire", plain_glb, 450, 450, ["--camera-direction=-1,-0.55,-1.2", "--camera-orthographic", "--edges"], WORKSHOP_BG, False, None, False)
            )

        # Optional custom close-ups (only when --at or CLOSE is explicitly specified)
        close_pts = ats or getattr(mod, "CLOSE", None) or []
        if close_pts:
            close_d = min(1.4, max(0.16, float(size.max()) * 0.78))
            c_dirs = [np.array([0.28, 0.18, 1.0]), np.array([1.0, 0.22, 0.38])]
            c_dirs = [v / np.linalg.norm(v) for v in c_dirs]
            for ci, pt in enumerate(close_pts[:2]):
                cd = c_dirs[ci % len(c_dirs)]
                render_jobs.append(
                    (f"close{ci}", plain_glb, 450, 450, _cam_args(np.asarray(pt, float) + close_d * cd, pt, 35.0), WORKSHOP_BG, False, None, False)
                )

        # Optional `--beside` comparison panel
        beside_names: list[str] = []
        if beside:
            row_items = [(main_doc, [main_bin], (-mid[0], -lo[1], -mid[2]), 0.0)]
            cursor_x = float(size[0] / 2)
            for bp in beside:
                bp_path = Path(bp)
                if bp_path.suffix == ".py":
                    bm, bname, _ = run_script(bp_path, palette=palette)
                    bpos, bnor, bpnt, btri = bm.build_mesh()
                    braw = glb_mod.pack_glb(bpos, bnor, bpnt, btri, bm.palette, name=bname)
                    bdoc, bbin = glb_mod.read_glb_parts(braw)
                    bbufs = [bbin]
                else:
                    bdoc, bbufs = glb_mod.load_gltf_doc(bp_path)
                    bname = bp_path.stem
                blo, bhi = glb_mod.bounds_of_gltf(bdoc)
                bmid = (blo + bhi) / 2
                bw = float(bhi[0] - blo[0])
                cursor_x += 0.35 + bw / 2
                row_items.append((bdoc, bbufs, (cursor_x - float(bmid[0]), -float(blo[1]), -float(bmid[2])), 0.0))
                cursor_x += bw / 2
                beside_names.append(bname)
            among_glb = tmp_p / "among.glb"
            among_glb.write_bytes(glb_mod.merge_glbs(row_items))
            render_jobs.append(
                ("among", among_glb, 900, 450, ["--camera-direction=0.35,-0.45,-1", "--camera-orthographic"], WORKSHOP_BG, False, None, False)
            )

        # Optional theme / palette comparison strip (`--themes` or `--palettes`)
        pal_entries: list[str | os.PathLike] = list(palettes) if palettes else (list(THEMES.keys()) if themes else [])
        pal_labels: list[str] = []
        if pal_entries:
            n_pal = len(pal_entries)
            pal_tile_w = 1800 // n_pal
            for idx, pf in enumerate(pal_entries):
                pf_str = str(pf)
                label = pf_str if pf_str in THEMES else Path(pf).stem
                tpal = Palette(overrides=pf, extra=getattr(mod, "MATERIALS", None))
                tb = glb_mod.pack_glb(pos, nor, pnt, tri, tpal, name=f"{vname}_{label}")
                tdoc, tbin = glb_mod.read_glb_parts(tb)
                tile_glb = tmp_p / f"pal_{idx}.glb"
                tile_glb.write_bytes(glb_mod.merge_glbs([(tdoc, [tbin], (-mid[0], -lo[1], -mid[2]), 0.0)]))
                tw = (1800 - pal_tile_w * (n_pal - 1)) if idx == n_pal - 1 else pal_tile_w
                render_jobs.append(
                    (f"pal_{idx}", tile_glb, tw, 344, ["--camera-direction=0.45,-0.30,-1", "--camera-orthographic", "--camera-zoom-factor=0.84"], WORKSHOP_BG, False, None, False)
                )
                pal_labels.append(label)

        def run_one(job):
            tag, gpath, w, h, opts, bg, grid, crop, do_halo = job
            png = tmp_p / f"{tag}.png"
            _f3d_render(gpath, png, w, h, opts, bg=bg, grid=grid)
            im = Image.open(png).convert("RGB")
            if crop:
                top, tall, w0, h0 = crop
                im = im.crop((0, round(top), im.width, round(top + tall))).resize((int(w0), int(h0)))
            if do_halo:
                im = _glowing(im)
            return tag, im

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            pics = dict(pool.map(run_one, render_jobs))

    t_render = time.time() - t0 - t_build

    # Assemble 1800px workshop sheet (visual panels only — diagnostics go to stdout)
    W = 1800
    extra_row = bool(close_pts or beside)
    H = 64 + 450 * (4 if extra_row else 3) + (380 if pal_entries else 0)
    sheet = Image.new("RGB", (W, H), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    f_big, f_sm = _font(20), _font(15)

    budget_txt = f" / {m.triangles:,}" if m.triangles else ""
    d.text(
        (12, 8),
        f"{vname}   {size[0]:.2f} x {size[2]:.2f} m, {size[1]:.2f} m tall   {len(tri):,}{budget_txt} triangles   {metrics_str}",
        font=f_big,
        fill=(240, 240, 242),
    )
    d.text(
        (12, 36),
        f"materials: {costly}    front +Z  up +Y    built {t_build:.2f}s, drawn {t_render:.1f}s",
        font=f_sm,
        fill=(175, 178, 185),
    )

    # Row 1: 4x 3/4 angled views
    sheet.paste(pics["turns"], (0, 64))
    for x, lab in zip((0, 450, 900, 1350), ("front-left 3/4", "back-left 3/4", "back-right 3/4", "front-right 3/4")):
        d.text((x + 8, 70), lab, font=f_sm, fill=(20, 20, 22))

    # Row 2: Front (+Z) & Back (-Z) ortho (900) | Perspective hero view (900, or 450 + wireframe 450)
    sheet.paste(pics["elev"], (0, 514))
    elev_lab = "front (+Z), back (-Z), and 1.75 m scale" if scale else "front (+Z) and back (-Z), orthographic"
    d.text((8, 520), elev_lab, font=f_sm, fill=(20, 20, 22))
    sheet.paste(pics["eye"], (900, 514))
    eye_lab = f"perspective view (with 1.75 m scale, y={eye_y:.2f}m)" if scale else "perspective view"
    d.text((908, 520), eye_lab, font=f_sm, fill=(20, 20, 22))
    if wireframe:
        sheet.paste(pics["wire"], (1350, 514))
        d.text((1358, 520), "topology / wireframe", font=f_sm, fill=(20, 20, 22))

    # Row 3: Left (-X) | Right (+X) | Top (+Y) | Bottom (-Y) orthographic views
    for x, key, lab in (
        (0, "left", "left (-X), orthographic"),
        (450, "right", "right (+X), orthographic"),
        (900, "top", "top (+Y), orthographic"),
        (1350, "bottom", "bottom (-Y), orthographic"),
    ):
        sheet.paste(pics[key], (x, 964))
        d.text((x + 8, 970), lab, font=f_sm, fill=(20, 20, 22))

    y_next = 1414
    if extra_row:
        for ci, pt in enumerate(close_pts[:2]):
            sheet.paste(pics[f"close{ci}"], (ci * 450, y_next))
            d.text((ci * 450 + 8, y_next + 6), f"close-up {ci + 1} at ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})", font=f_sm, fill=(20, 20, 22))
        if beside:
            x_b = 900 if close_pts else 0
            sheet.paste(pics["among"], (x_b, y_next))
            d.text((x_b + 8, y_next + 6), "beside: " + ", ".join(beside_names), font=f_sm, fill=(20, 20, 22))
        y_next += 450

    if pal_entries:
        bg_rgb = tuple(round(float(c) * 255) for c in WORKSHOP_BG.split(","))
        d.rectangle((0, y_next, W - 1, y_next + 379), fill=bg_rgb)
        x_cur = 0
        for idx, label in enumerate(pal_labels):
            tile = pics[f"pal_{idx}"]
            sheet.paste(tile, (x_cur, y_next + 4))
            d.text((x_cur + tile.width // 2, y_next + 352), label, font=f_sm, fill=(20, 20, 22), anchor="mt")
            x_cur += tile.width
        d.text((8, y_next + 6), "themes", font=f_sm, fill=(20, 20, 22))

    sheet.save(out)
    summary = [
        f"{out}  ({time.time() - t0:.1f}s: build {t_build:.2f}s, draw {t_render:.1f}s)  {len(tri):,} tris  {metrics_str}",
        *[f"  warning :{ln} {text}" for (_, ln), text in m.warnings()],
    ]
    return out, summary


def render_look(
    model_paths: list[str | os.PathLike],
    out_path: str | os.PathLike | None = None,
    *,
    views: list[str] | None = None,
    az: float | None = None,
    el: float | None = None,
    ortho: bool = False,
    size: int = 900,
    slots: bool = False,
    scale: bool = False,
    palette: str | os.PathLike | dict | None = None,
) -> Path:
    """Render one or more `.glb`/`.gltf`/`.py` models side by side at true scale on a 1 m grid."""
    if not model_paths:
        raise ValueError("look requires at least one model path")
    if views and (az is not None or el is not None):
        raise ValueError("cannot combine --views with --az/--el")

    view_specs: list[tuple[str, tuple[float, float, float], str]]
    if az is not None or el is not None:
        a = 40.0 if az is None else az
        e = 20.0 if el is None else el
        view_specs = [(f"az {a:g} el {e:g}", angle_dir(a, e), "angle")]
    else:
        view_specs = [parse_view(v) for v in (views or ["3q"])]

    loaded = []
    labels = []
    notes = []
    boxes = []
    slot_items = []

    with tempfile.TemporaryDirectory(prefix="jig-look-") as tmp:
        tmp_p = Path(tmp)
        for idx, mp in enumerate(model_paths):
            p = Path(mp)
            if p.suffix == ".py":
                m, vname, _ = run_script(p, palette=palette)
                pos, nor, pnt, tri = m.build_mesh()
                raw = glb_mod.pack_glb(pos, nor, pnt, tri, m.palette, name=vname, slots=m.named)
                doc, bin_ = glb_mod.read_glb_parts(raw)
                bufs = [bin_]
                nums = metrics_mod.measure(pos[tri])
                lbl = vname
                m_slots = m.named
            else:
                doc, bufs = glb_mod.load_gltf_doc(p)
                nums = metrics_mod.measure_file(p)
                lbl = p.stem
                m_slots = glb_mod.extract_slots(doc)
            lo, hi = glb_mod.bounds_of_gltf(doc)
            loaded.append((doc, bufs))
            boxes.append((lo, hi))
            labels.append(lbl)
            notes.append(f"{nums['triangles']:,} tris  {metrics_mod.format_line(nums)}")
            slot_items.append(m_slots)

        tiles = []
        gap = 0.4
        for vi, (vlabel, vdir, vkind) in enumerate(view_specs):
            right = np.array([vdir[2], 0.0, -vdir[0]], dtype=float)
            rn = float(np.hypot(right[0], right[2]))
            right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])

            moves = []
            edge = None
            first_mid = None
            for lo, hi in boxes:
                mid = (lo + hi) / 2
                half = abs(right[0]) * (hi[0] - lo[0]) / 2 + abs(right[2]) * (hi[2] - lo[2]) / 2
                if edge is None:
                    target, edge, first_mid = mid.copy(), half, mid.copy()
                else:
                    target = np.array([first_mid[0] + right[0] * (edge + gap + half), 0.0, first_mid[2] + right[2] * (edge + gap + half)])
                    edge += gap + 2 * half
                moves.append((float(target[0] - mid[0]), float(-lo[1]), float(target[2] - mid[2])))

            merge_list = [(doc, bufs, mv, 0.0) for (doc, bufs), mv in zip(loaded, moves)]
            if scale:
                sm_doc, sm_bufs = _scale_marks_glb(0.0, 0.0)
                sm_lo, sm_hi = glb_mod.bounds_of_gltf(sm_doc)
                sm_mid = (sm_lo + sm_hi) / 2
                sm_half = abs(right[0]) * (sm_hi[0] - sm_lo[0]) / 2 + abs(right[2]) * (sm_hi[2] - sm_lo[2]) / 2
                sm_target = np.array(
                    [first_mid[0] + right[0] * (edge + gap + sm_half), 0.0, first_mid[2] + right[2] * (edge + gap + sm_half)]
                )
                sm_mv = (float(sm_target[0] - sm_mid[0]), float(-sm_lo[1]), float(sm_target[2] - sm_mid[2]))
                merge_list.append((sm_doc, sm_bufs, sm_mv, 0.0))
            if slots:
                for m_slots, mv in zip(slot_items, moves):
                    mk = _slot_markers_glb(m_slots)
                    if mk is not None:
                        merge_list.append((mk[0], mk[1], mv, 0.0))

            scene_glb = tmp_p / f"scene_{vi}.glb"
            scene_glb.write_bytes(glb_mod.merge_glbs(merge_list))
            tile_png = tmp_p / f"tile_{vi}.png"

            up = _camera_up(vdir)
            up_arg = f"--camera-view-up={','.join(f'{c:g}' for c in up)}"
            if vkind == "eye":
                w_lo = np.min([lo + np.array(mv) for (lo, _), mv in zip(boxes, moves)], axis=0)
                w_hi = np.max([hi + np.array(mv) for (_, hi), mv in zip(boxes, moves)], axis=0)
                w_mid = (w_lo + w_hi) / 2
                dist = max(3.0, 1.9 * float(w_hi[0] - w_lo[0]))
                aim = _cam_args((w_mid[0], EYE_HEIGHT, w_hi[2] + dist), w_mid, 30.0)
            else:
                aim = [f"--camera-direction={-vdir[0]},{-vdir[1]},{-vdir[2]}", up_arg]
            if ortho:
                aim.append("--camera-orthographic")
            _f3d_render(scene_glb, tile_png, size, size, aim, bg=DEFAULT_BG, grid=(vdir[1] > 0))
            tiles.append((vlabel, Image.open(tile_png).convert("RGB")))

    # Compose multi-view image with header
    cols = min(len(tiles), 4)
    rows_n = math.ceil(len(tiles) / cols)
    f_hdr, f_sm = _font(18), _font(15)
    hdr_lines = [
        f"{lbl}  {hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} x {hi[2] - lo[2]:.2f} m   ({note})"
        for lbl, (lo, hi), note in zip(labels, boxes, notes)
    ]
    hdr_h = 16 + 24 * len(hdr_lines)
    page = Image.new("RGB", (cols * size + (cols - 1) * 2, hdr_h + rows_n * (size + 26)), (20, 20, 22))
    d = ImageDraw.Draw(page)
    for k, hl in enumerate(hdr_lines):
        d.text((10, 8 + k * 24), hl, font=f_hdr, fill=(238, 238, 240))
    for i, (vlab, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        x0, y0 = c * (size + 2), hdr_h + r * (size + 26)
        d.rectangle((x0, y0, x0 + size - 1, y0 + 25), fill=(12, 12, 14))
        d.text((x0 + 8, y0 + 4), vlab, font=f_sm, fill=(255, 220, 80))
        page.paste(im, (x0, y0 + 26))

    out = Path(out_path) if out_path else Path("out") / f"look-{'-'.join(labels[:3])}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    return out


_CROP_RE = re.compile(r"\d+x\d+\+\d+\+\d+")


def join_sheet(
    pics: list[str | os.PathLike],
    out_path: str | os.PathLike = "out/sheet.png",
    *,
    cols: int | None = None,
    labels: list[str] | None = None,
    crop: str | None = None,
    width: int | None = None,
    lift: float = 1.0,
) -> Path:
    """Join images into one labelled grid sheet."""
    if not pics:
        raise ValueError("sheet requires at least one image")
    ims = []
    names = []
    for raw in pics:
        s = str(raw)
        path_s, _, own = s.rpartition("@")
        if path_s and _CROP_RE.fullmatch(own):
            p, box = Path(path_s), own
        else:
            p, box = Path(s), crop
        im = Image.open(p).convert("RGB")
        if lift != 1.0:
            im = im.point([round(255 * (v / 255) ** (1 / lift)) for v in range(256)] * 3)
        if box:
            wh, x, y = box.split("+")
            w_, h_ = map(int, wh.split("x"))
            im = im.crop((int(x), int(y), int(x) + w_, int(y) + h_))
        ims.append(im)
        names.append(p.stem)

    if labels:
        if len(labels) != len(pics):
            raise ValueError(f"{len(labels)} labels for {len(pics)} pictures")
        names = [old if new == "-" else new.strip() for old, new in zip(names, labels)]

    w = width or ims[0].width
    h = round(ims[0].height * w / ims[0].width)
    cols = cols or (len(ims) if len(ims) <= 4 else 4)
    rows = math.ceil(len(ims) / cols)
    fsize = max(14, min(24, w // 45))
    bar = fsize + 10
    font = _font(fsize)
    page = Image.new("RGB", (cols * (w + 2) - 2, rows * (h + bar + 2) - 2), (20, 20, 20))
    d = ImageDraw.Draw(page)
    for i, (im, t) in enumerate(zip(ims, names)):
        k = min(w / im.width, h / im.height)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
        x, y = (i % cols) * (w + 2), (i // cols) * (h + bar + 2)
        d.rectangle((x, y, x + w - 1, y + bar - 1), fill=(12, 12, 14))
        d.text((x + 6, y + 4), t, fill=(255, 220, 0), font=font)
        page.paste(im, (x + (w - im.width) // 2, y + bar + (h - im.height) // 2))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    return out
