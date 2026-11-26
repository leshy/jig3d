"""Unified Model builder combining hard-surface CAD (`Solid`) and organic SubD (`QuadCage`)."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys

import numpy as np

from . import _lay, _places, _plane
from ._features import Features, painted
from .materials import Palette
from .sculpt import QuadCage
from .solid import moved as _moved_parts

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _caller():
    f = sys._getframe(1)
    while f.f_back and os.path.dirname(os.path.abspath(f.f_code.co_filename)) == _PKG_DIR:
        f = f.f_back
    return f


def _rotation(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _rigid(parts, R, t):
    out = []
    for part in parts:
        q = []
        for n, d, st, sm in part:
            rn = R @ np.asarray(n, float)
            sm = sm if sm is None else (R @ np.asarray(sm[0], float) + t, R @ np.asarray(sm[1], float))
            q.append((rn, float(d + rn @ t), st, sm))
        out.append(q)
    return out


def _about_y(at, turn):
    at_v = np.asarray(at, float)
    if turn % 360 in (0, 180):
        half = turn % 360 == 180
        R = np.diag([-1.0, 1.0, -1.0]) if half else np.eye(3)
        return (lambda parts: _moved_parts(parts, at_v, half)), (lambda v: R @ np.asarray(v, float) + at_v), R, at_v
    R = _rotation(np.array([0.0, 1.0, 0.0]), math.radians(turn))
    return (lambda parts: _rigid(parts, R, at_v)), (lambda v: R @ np.asarray(v, float) + at_v), R, at_v


class Model(Features):
    """Stateful procedural model builder passed as `p` into `def model(p, **knobs):` (or `def shape(p):`)."""

    def __init__(self, laid: bool = True, palette: Palette | None = None, overrides: str | Path | dict | None = None):
        self.laid = laid
        self.palette = palette if palette is not None else Palette(overrides=overrides)
        self.statements: list[tuple] = []  # (kind, parts, paint, options, (file, line))
        self.sculpts: list[tuple] = []  # (QuadCage, material, smooth, (file, line))
        self.points: list[tuple] = []
        self.pinned: tuple | None = None
        self.triangles: int | None = None  # optional budget
        self.hinges: dict = {}
        self._poses: list[tuple] = []  # (name, part_xf, pt_xf, R, t)
        self._features_init()

    def _state(self, kind, solid, paint=None, **options):
        parts = solid.parts if hasattr(solid, "parts") else solid
        for _, move, *_ in reversed(self._poses):
            parts = move(parts)
        f = _caller()
        line = getattr(solid, "line", None) or (f.f_code.co_filename, f.f_lineno)
        options["poses"] = tuple(name for name, *_ in self._poses)
        self.statements.append((kind, parts, paint, options, line))

    def put(self, solid_or_cage, material, smooth: bool = False):
        """Draw a hard-surface `Solid` (or list of Solids) OR an organic `QuadCage` in `material`."""
        if isinstance(solid_or_cage, QuadCage):
            return self.sculpt(solid_or_cage, material, smooth=smooth)
        return self._draw("put", solid_or_cage, material)

    def sculpt(self, cage: QuadCage, material, smooth: bool = False) -> QuadCage:
        """Emit an organic `QuadCage` into the model, honoring active `with p.on`, `with p.mirror`, and `with p.pose`."""
        f = _caller()
        line = (f.f_code.co_filename, f.f_lineno)
        c = cage
        if self._frames:
            c = c._xf(*self._frames[-1])
        copies = [c]
        for R, t in self._mirrors:
            copies += [x._xf(R, t) for x in copies]
        for _, _, _, R, t in reversed(self._poses):
            copies = [x._xf(R, t) for x in copies]
        for x in copies:
            self.sculpts.append((x, material, smooth, line))
        return copies[0]

    def cut(self, solid, role=None, mouth=0.0, of=None):
        """Boolean-subtract `solid` from earlier puts (or only `of=` Parts); `mouth` adds a 45° bevel rim."""
        return self._cut(solid, role, mouth, of=of)

    def detail(self, solid, material):
        """Alias for `put` (compatibility with scripts distinguishing thin surface details)."""
        return self.put(solid, material)

    def place(self, sub, at=(0, 0, 0), turn=0, **knobs):
        """Place a reusable sub-model function `sub(p, **knobs)` at `at`, turned `turn` degrees about Y."""
        child = Model(laid=self.laid, palette=self.palette)
        sub(child, **knobs)
        move, spot, R_y, t_y = _about_y(at, turn)
        outer = tuple(name for name, *_ in self._poses)
        for kind, parts, paint, options, line in child.statements:
            parts = move(parts)
            for _, m, *_ in reversed(self._poses):
                parts = m(parts)
            self.statements.append((kind, parts, paint, {**options, "poses": outer + options["poses"]}, line))
        for cage, mat, sm, line in child.sculpts:
            c = cage._xf(R_y, t_y)
            for _, _, _, R_p, t_p in reversed(self._poses):
                c = c._xf(R_p, t_p)
            self.sculpts.append((c, mat, sm, line))
        self.hinges.update(child.hinges)
        for name, v, fits, on in child.points:
            self._point(name, spot(v), fits, on)
        for nm, a, m, sz in child.named:
            na, nm_vec = spot(a), R_y @ np.asarray(m, float)
            for _, _, sp, R_p, _ in reversed(self._poses):
                na, nm_vec = sp(na), R_p @ nm_vec
            self.named.append((nm, na, nm_vec, sz))

    @contextmanager
    def pose(self, name, hinge=None, turn=(0, 0), at=0.0, slide=None):
        """Articulate everything inside around a hinge line `(a, b)` or along a `slide` vector at parameter `at` in 0..1."""
        self.hinges[name] = None if slide is not None else tuple(np.asarray(v, float) for v in hinge)
        if slide is not None:
            t = np.asarray(slide, float) * at
            R = np.eye(3)
        else:
            a, b = (np.asarray(v, float) for v in hinge)
            R = _rotation(b - a, math.radians(turn[0] + (turn[1] - turn[0]) * at))
            t = a - R @ a
        self._poses.append((name, lambda parts: _rigid(parts, R, t), lambda v: R @ np.asarray(v, float) + t, R, t))
        try:
            yield
        finally:
            self._poses.pop()

    def point(self, name, at, fits=None, on=None):
        v = np.asarray(at, float)
        if self._frames:
            R, t = self._frames[-1]
            v = R @ v + t
        self._point(name, v, fits, on)

    def _point(self, name, v, fits, on):
        for _, _, spot, *_ in reversed(self._poses):
            v = spot(v)
        self.points.append((name, v, fits, on))

    def budget(self, n: int):
        """Declare a target triangle budget for this model."""
        self.triangles = int(n)

    def box(self, lo, hi):
        """Pin the CSG bounding box to `lo..hi`."""
        self.pinned = (tuple(lo), tuple(hi))

    def _csg_mesh_obj(self):
        if not self.statements:
            return None
        if self.pinned:
            lo, hi = self.pinned
        else:
            far = [(tuple(float(e == a) * s for e in range(3)), 50.0, True, None) for a in range(3) for s in (-1.0, 1.0)]
            pts = np.array(
                [
                    q
                    for kind, parts, paint, _, _ in self.statements
                    if paint is not None
                    for part in parts
                    for _, f in _plane._faces(
                        _plane._dedup([(tuple(float(c) for c in n), float(d), st, None) for n, d, st, *_ in part] + far)
                    )
                    for q in f
                ]
            )
            if not len(pts):
                return None
            lo, hi = pts.min(0), pts.max(0)
        m = (_lay.Laid if self.laid else _lay.Mesh)(lo, hi, palette=self.palette)
        by_part: dict = {}
        cut_of: dict = {}
        for _, parts, paint, options, line in self.statements:
            idx = len(m.puts)
            part = options.get("part")
            if part is not None:
                by_part.setdefault(part, []).append(idx)
            if options.get("of") is not None:
                cut_of[idx] = options["of"]
            m.puts.append((parts, paint))
            m.lines.append(line)
        if cut_of:
            missing = [t for targets in cut_of.values() for t in targets if t not in by_part]
            if missing:
                raise ValueError(f"cut(of=...) target was not drawn in this model: {missing[0]!r}")
            m.cut_of = {c: frozenset(i for p in targets for i in by_part[p]) for c, targets in cut_of.items()}
        return painted(m, self.painted)

    def build_mesh(self):
        """Compile all CSG solids and QuadCage sculpts into `(pos f32, nor f32, mat_ids u16, tri u32)`."""
        pos_list, nor_list, pnt_list, tri_list = [], [], [], []
        base = 0
        csg = self._csg_mesh_obj()
        if csg is not None:
            p, n, q, t = csg.mesh()
            if len(t):
                pos_list.append(p)
                nor_list.append(n)
                pnt_list.append(q)
                tri_list.append(t)
                base += len(p)
        for cage, mat, sm, _ in self.sculpts:
            p, n, q, t = cage.to_mesh(mat, self.palette, smooth=sm)
            if len(t):
                pos_list.append(p)
                nor_list.append(n)
                pnt_list.append(q)
                tri_list.append(t + base)
                base += len(p)
        if not pos_list:
            raise ValueError("empty model: no geometry was emitted")
        return (
            np.concatenate(pos_list).astype(np.float32),
            np.concatenate(nor_list).astype(np.float32),
            np.concatenate(pnt_list).astype(np.uint16),
            np.concatenate(tri_list).astype(np.uint32),
        )

    def digest(self) -> str:
        """Deterministic content hash of the compiled geometry & materials."""
        pos, nor, pnt, tri = self.build_mesh()
        h = hashlib.sha1(b"jig-v1")
        h.update(np.round(pos, 5).tobytes())
        h.update(np.round(nor, 5).tobytes())
        h.update(pnt.tobytes())
        h.update(tri.tobytes())
        for nm, a, m, sz in self.named:
            h.update(nm.encode() + np.round(a, 5).tobytes() + np.round(m, 5).tobytes() + np.round(sz, 5).tobytes())
        return h.hexdigest()[:16]

    def cost(self) -> dict[tuple[str, int], int]:
        """Per-source-line triangle counts across both CSG puts and QuadCage sculpts."""
        out: dict[tuple[str, int], int] = defaultdict(int)
        csg = self._csg_mesh_obj()
        if csg is not None and hasattr(csg, "cost"):
            for k, v in csg.cost().items():
                out[k] += v
        elif csg is not None:
            _, _, _, t = csg.mesh()
            out[("<welded>", 0)] += len(t)
        for cage, mat, sm, line in self.sculpts:
            _, _, _, t = cage.to_mesh(mat, self.palette, smooth=sm)
            out[line] += len(t)
        return dict(out)

    def warnings(self) -> list[tuple[tuple[str, int], str]]:
        csg = self._csg_mesh_obj()
        if csg is not None and hasattr(csg, "warnings"):
            return csg.warnings()
        return []

    def describe_lines(self, line: int | None = None) -> list[str]:
        """Human/agent-readable breakdown of size, triangles, materials, slots, warnings, and faces."""
        pos, _, pnt, tri = self.build_mesh()
        lo, hi = pos.min(0).astype(float), pos.max(0).astype(float)
        size = hi - lo
        if line is not None:
            rows = []
            for (_, ln), part in self.drawn_parts:
                if ln == line and part.role is not None:
                    faces = [
                        (f.name, round(f.w, 3), round(f.h, 3), tuple(round(float(c), 3) for c in f.origin))
                        for f in part.faces()
                        if f.name != "chamfer"
                    ]
                    rows.append(f":{ln} {part.role}: {len(faces)} faces (chamfers omitted)")
                    rows += [f"  {name:<18} {w:.3f} x {h:.3f}  at {at}" for name, w, h, at in faces]
            return rows or [f"no part is stated on line {line}"]

        roles: dict[str, int] = defaultdict(int)
        for q in pnt[tri[:, 0]]:
            roles[self.palette.name_of(int(q))] += 1
        budget_str = f" / budget {self.triangles:,}" if self.triangles else ""
        rows = [
            f"size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m   bounds [{lo[0]:.3f}..{hi[0]:.3f}, {lo[1]:.3f}..{hi[1]:.3f}, {lo[2]:.3f}..{hi[2]:.3f}]",
            f"triangles {len(tri):,}{budget_str} ({'laid' if self.laid else 'welded'})   digest {self.digest()}",
            "by material:  " + "  ".join(f"{k} {v:,}" for k, v in sorted(roles.items(), key=lambda kv: -kv[1])),
        ]
        cost_items = sorted(self.cost().items(), key=lambda kv: -kv[1])
        src_cache: dict[str, list[str]] = {}
        for (f, ln), v in cost_items[:12]:
            if f not in src_cache:
                src_cache[f] = open(f).read().split("\n") if os.path.isfile(f) else []
            txt = src_cache[f][ln - 1].strip()[:76] if 0 < ln <= len(src_cache[f]) else ""
            rows.append(f"  {v:5d}  :{ln:<4d} {txt}")
        if len(cost_items) > 12:
            rows.append(f"  {sum(v for _, v in cost_items[12:]):5d}  ({len(cost_items) - 12} more lines)")
        for (_, ln), text in self.warnings():
            rows.append(f"warning :{ln} {text}")
        if self.named:
            rows.append("slots:")
            for nm, a, m, sz in self.named:
                rows.append(
                    f"  {nm:<14} at ({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})  normal ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f})  size {sz[0]:.3f} x {sz[1]:.3f}"
                )
        if self.drawn_parts:
            rows.append("parts by line (pass LINE to inspect face sizes):")
            last = None
            for (_, ln), part in self.drawn_parts:
                if part.role is None:
                    continue
                words = ", ".join(sorted({f.name for f in part.faces() if f.name != "chamfer"}))
                if last and last[0] == (ln, str(part.role), words):
                    last[1] += 1
                    rows[-1] = f"  :{ln} {part.role} x{last[1]}: {words}"
                    continue
                last = [(ln, str(part.role), words), 1]
                rows.append(f"  :{ln} {part.role}: {words}")
        return rows


# --- Script loading helpers ---


def load_script(path: str | os.PathLike):
    """Load a `.py` model script fresh as a Python module."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"model script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"jig_model_{p.stem}", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Allow local sibling imports cleanly
    parent = str(p.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(mod)
    finally:
        if added and parent in sys.path:
            sys.path.remove(parent)
    return mod


def script_variants(mod, default_name: str) -> dict[str, dict]:
    """Extract `{variant_name: kwargs}` from `VARIANTS` or `PIECES`, defaulting to `{stem: {}}`."""
    return getattr(mod, "VARIANTS", None) or getattr(mod, "PIECES", None) or {default_name: {}}


def run_script(
    path: str | os.PathLike,
    variant: str | None = None,
    palette: str | Path | dict | None = None,
    laid: bool | None = None,
) -> tuple[Model, str, object]:
    """Execute a model script's `model(p, **knobs)` (or `shape(p, **knobs)`) and return `(Model, variant_name, mod)`."""
    p = Path(path).resolve()
    mod = load_script(p)
    fn = getattr(mod, "model", None) or getattr(mod, "shape", None)
    if fn is None:
        raise AttributeError(f"{p.name}: must define `def model(p, ...):` (or `def shape(p, ...):`)")
    variants = script_variants(mod, p.stem)
    if variant is None:
        vname = next(iter(variants))
    else:
        if variant not in variants:
            raise KeyError(f"{p.name}: unknown variant {variant!r}; available: {', '.join(variants)}")
        vname = variant
    knobs = variants[vname]
    use_laid = laid if laid is not None else getattr(mod, "LAID", True)
    use_overrides = palette if palette is not None else getattr(mod, "PALETTE", None)
    extra_mats = getattr(mod, "MATERIALS", None)
    pal = Palette(overrides=use_overrides, extra=extra_mats)
    m = Model(laid=use_laid, palette=pal)
    fn(m, **knobs)
    return m, vname, mod
