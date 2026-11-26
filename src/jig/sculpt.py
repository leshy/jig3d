"""Deterministic organic & curved low-poly sculpting via creased Catmull-Clark `QuadCage`.

Pure NumPy control-cage engine:
- `QuadCage`: immutable quad/n-gon cage with constructors (`ellipsoid`, `box_grid`, `loft_grid`, `uv_patch`),
  `.mirror_x()`, `.sculpt()`, `.move()`, `.turn()`, `.scale()`, `.tag()`, `.keep()`, `.drop()`,
  `.crease()`, creased Catmull-Clark `.subdivide(steps=1, pin_corners=False)`, `.ensure_outward(c)`,
  watertight beveled `.thicken(t, inner_material=..., rim_material=..., bevel=..., tuck_in=...)`.
- `project_cage(target_tris, cage, offset=0.0, relief_fn=None)`: snap a cage onto any triangle mesh
  (e.g. a hard-surface Solid or another QuadCage) along smooth normals for exact surface conformity.
"""

from __future__ import annotations

import inspect
import math

import numpy as np


def _arity(fn) -> int:
    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 1


def _eval_face_fn(fn, arity, ctr, fv, tag):
    return fn(ctr) if arity <= 1 else (fn(ctr, tag) if arity == 2 else fn(ctr, fv, tag))


def _ce(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _face_normal(pts) -> np.ndarray:
    n = np.zeros(3, dtype=np.float64)
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        n += (
            (a[1] - b[1]) * (a[2] + b[2]),
            (a[2] - b[2]) * (a[0] + b[0]),
            (a[0] - b[0]) * (a[1] + b[1]),
        )
    return n


def _vertex_normals(verts: np.ndarray, faces) -> np.ndarray:
    acc = np.zeros_like(verts)
    for f in faces:
        fn = _face_normal([verts[i] for i in f])
        acc[list(f)] += fn
    lens = np.linalg.norm(acc, axis=1, keepdims=True)
    return np.where(lens > 1e-12, acc / np.maximum(lens, 1e-12), np.array([0.0, 1.0, 0.0]))


class QuadCage:
    """Immutable low-poly quad/n-gon control cage with face tags, sharp creases, and per-face materials."""

    def __init__(self, verts, quads, tags=None, creases=None, face_materials=None):
        self.verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3).copy()
        self.quads = tuple(tuple(int(i) for i in f) for f in quads)
        nf = len(self.quads)
        self.tags = ("",) * nf if tags is None else tuple(str(t) for t in tags)
        if len(self.tags) != nf:
            raise ValueError(f"tags length {len(self.tags)} != quads length {nf}")
        self.creases = frozenset(_ce(int(u), int(v)) for u, v in (creases or ()))
        self._fmats = (None,) * nf if face_materials is None else tuple(face_materials)
        if len(self._fmats) != nf:
            raise ValueError(f"face_materials length {len(self._fmats)} != quads length {nf}")

    @classmethod
    def ellipsoid(cls, rx, ry, rz, center=(0.0, 0.0, 0.0), nu=8, nv=6) -> QuadCage:
        """Watertight oblong ellipsoid (`nu >= 4` around Y, `nv >= 3` latitude bands)."""
        nu, nv = max(4, int(nu)), max(3, int(nv))
        cx, cy, cz = (float(c) for c in center)
        rx, ry, rz = float(rx), float(ry), float(rz)
        verts = [[cx, cy - ry, cz]]
        for j in range(1, nv):
            phi = math.pi * (1.0 - j / nv)
            sp, cp = math.sin(phi), math.cos(phi)
            verts.extend(
                [cx + rx * sp * math.sin(2.0 * math.pi * i / nu), cy + ry * cp, cz + rz * sp * math.cos(2.0 * math.pi * i / nu)]
                for i in range(nu)
            )
        verts.append([cx, cy + ry, cz])
        north, top = len(verts) - 1, 1 + (nv - 2) * nu
        quads = (
            [(0, 1 + (i + 1) % nu, 1 + i) for i in range(nu)]
            + [
                (1 + j * nu + i, 1 + j * nu + (i + 1) % nu, 1 + (j + 1) * nu + (i + 1) % nu, 1 + (j + 1) * nu + i)
                for j in range(nv - 2)
                for i in range(nu)
            ]
            + [(top + i, top + (i + 1) % nu, north) for i in range(nu)]
        )
        return cls(verts, quads)

    @classmethod
    def box_grid(cls, center, size, segs=(2, 2, 2)) -> QuadCage:
        """Watertight subdivided quad box with `segs=(sx, sy, sz)` per axis."""
        nx, ny, nz = (max(1, int(s)) for s in segs)
        cx, cy, cz = (float(c) for c in center)
        sx, sy, sz = (float(s) for s in size)
        vmap: dict[tuple[int, int, int], int] = {}
        verts: list[list[float]] = []

        def vid(ix, iy, iz):
            key = (ix, iy, iz)
            if key not in vmap:
                vmap[key] = len(verts)
                verts.append([cx + (ix / nx - 0.5) * sx, cy + (iy / ny - 0.5) * sy, cz + (iz / nz - 0.5) * sz])
            return vmap[key]

        quads = (
            [(vid(x, y, nz), vid(x + 1, y, nz), vid(x + 1, y + 1, nz), vid(x, y + 1, nz)) for x in range(nx) for y in range(ny)]
            + [(vid(x + 1, y, 0), vid(x, y, 0), vid(x, y + 1, 0), vid(x + 1, y + 1, 0)) for x in range(nx) for y in range(ny)]
            + [(vid(nx, y, z + 1), vid(nx, y, z), vid(nx, y + 1, z), vid(nx, y + 1, z + 1)) for z in range(nz) for y in range(ny)]
            + [(vid(0, y, z), vid(0, y, z + 1), vid(0, y + 1, z + 1), vid(0, y + 1, z)) for z in range(nz) for y in range(ny)]
            + [(vid(x, ny, z + 1), vid(x + 1, ny, z + 1), vid(x + 1, ny, z), vid(x, ny, z)) for x in range(nx) for z in range(nz)]
            + [(vid(x, 0, z), vid(x + 1, 0, z), vid(x + 1, 0, z + 1), vid(x, 0, z + 1)) for x in range(nx) for z in range(nz)]
        )
        return cls(verts, quads)

    @classmethod
    def loft_grid(cls, rows, closed_u=True, cap_start=False, cap_end=False) -> QuadCage:
        """Loft an `(R, C, 3)` grid of cross-section rows into an open or closed-U quad cage."""
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ValueError(f"loft_grid expects shape (R>=2, C>=2, 3), got {arr.shape}")
        nr, nc = arr.shape[:2]
        quads = [
            (r * nc + c, r * nc + (c + 1) % nc, (r + 1) * nc + (c + 1) % nc, (r + 1) * nc + c)
            for r in range(nr - 1)
            for c in range(nc if closed_u else nc - 1)
        ]
        if closed_u and cap_start:
            quads.append(tuple(range(nc - 1, -1, -1)))
        if closed_u and cap_end:
            quads.append(tuple((nr - 1) * nc + c for c in range(nc)))
        return cls(arr.reshape(-1, 3), quads)

    @classmethod
    def uv_patch(cls, pts_grid) -> QuadCage:
        """Open 2D `(R, C, 3)` quad patch for curved panels, visors, fairings, or shells."""
        return cls.loft_grid(pts_grid, closed_u=False)

    def _compact(self, keep_idx) -> QuadCage:
        keep_idx = list(keep_idx)
        used, vmap = [], {}
        for fi in keep_idx:
            for vi in self.quads[fi]:
                if vi not in vmap:
                    vmap[vi] = len(used)
                    used.append(vi)
        return QuadCage(
            self.verts[used] if used else np.zeros((0, 3), dtype=np.float64),
            [tuple(vmap[vi] for vi in self.quads[fi]) for fi in keep_idx],
            [self.tags[fi] for fi in keep_idx],
            {_ce(vmap[u], vmap[v]) for u, v in self.creases if u in vmap and v in vmap},
            [self._fmats[fi] for fi in keep_idx],
        )

    def _xf(self, R: np.ndarray, t: np.ndarray) -> QuadCage:
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        new_v = self.verts @ R.T + t
        quads = tuple(tuple(reversed(f)) for f in self.quads) if np.linalg.det(R) < 0 else self.quads
        return QuadCage(new_v, quads, self.tags, self.creases, self._fmats)

    def move(self, x=0.0, y=0.0, z=0.0) -> QuadCage:
        """Translate cage by `(x, y, z)`."""
        return self._xf(np.eye(3), np.array([x, y, z], dtype=np.float64))

    def turn(self, axis: str, deg: float, about=(0.0, 0.0, 0.0)) -> QuadCage:
        """Rotate `deg` degrees (right-handed) about `axis` (`'x'`, `'y'`, `'z'`) through `about`."""
        c, s, k = math.cos(math.radians(deg)), math.sin(math.radians(deg)), "xyz".index(axis)
        R = np.eye(3)
        i, j = (k + 1) % 3, (k + 2) % 3
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        about_v = np.asarray(about, dtype=np.float64)
        return self._xf(R, about_v - R @ about_v)

    def scale(self, sx, sy=None, sz=None, about=(0.0, 0.0, 0.0)) -> QuadCage:
        """Scale cage uniformly or per axis around `about`."""
        if sy is None:
            sy = sz = sx
        elif sz is None:
            sz = sx
        about_v = np.asarray(about, dtype=np.float64)
        s = np.array([float(sx), float(sy), float(sz)], dtype=np.float64)
        new_v = (self.verts - about_v) * s + about_v
        quads = tuple(tuple(reversed(f)) for f in self.quads) if (s[0] * s[1] * s[2]) < 0 else self.quads
        return QuadCage(new_v, quads, self.tags, self.creases, self._fmats)

    def mirror_x(self, tol=1e-5) -> QuadCage:
        """Mirror `+X` half across `X = 0`, welding `|x| < tol` seam vertices with outward winding."""
        half = self._compact(i for i, f in enumerate(self.quads) if float(self.verts[list(f), 0].mean()) >= -tol)
        verts, pmap, nmap = [], {}, {}
        for i, p in enumerate(half.verts):
            if abs(float(p[0])) < tol:
                pmap[i] = nmap[i] = len(verts)
                verts.append(np.array([0.0, p[1], p[2]], dtype=np.float64))
            else:
                pmap[i], nmap[i] = len(verts), len(verts) + 1
                verts.extend((p.copy(), np.array([-p[0], p[1], p[2]], dtype=np.float64)))
        quads = [q for f in half.quads for q in (tuple(pmap[v] for v in f), tuple(nmap[v] for v in reversed(f)))]
        creases = {_ce(pmap[u], pmap[v]) for u, v in half.creases} | {_ce(nmap[u], nmap[v]) for u, v in half.creases}
        return QuadCage(
            verts,
            quads,
            [t for t in half.tags for _ in (0, 1)],
            creases,
            [m for m in half._fmats for _ in (0, 1)],
        )

    def sculpt(self, fn, vectorized=None) -> QuadCage:
        """Apply vertex deformer `fn(pt) -> new_pt` (or vectorized `fn(verts) -> new_verts`)."""
        if len(self.verts) == 0:
            return self
        if vectorized is True:
            new_v = np.asarray(fn(self.verts.copy()), dtype=np.float64)
        elif vectorized is False:
            new_v = np.asarray([fn(p.copy()) for p in self.verts], dtype=np.float64)
        else:
            try:
                p0 = np.asarray(fn(self.verts[0].copy()), dtype=np.float64)
                new_v = (
                    np.asarray([p0, *(fn(p.copy()) for p in self.verts[1:])], dtype=np.float64)
                    if p0.shape == (3,)
                    else np.asarray(fn(self.verts.copy()), dtype=np.float64)
                )
            except (IndexError, TypeError, ValueError):
                new_v = np.asarray(fn(self.verts.copy()), dtype=np.float64)
        if new_v.shape != self.verts.shape:
            raise ValueError(f"sculpt returned shape {new_v.shape}, expected {self.verts.shape}")
        return QuadCage(new_v, self.quads, self.tags, self.creases, self._fmats)

    def _match_faces(self, spec) -> set[int]:
        if callable(spec):
            ar = _arity(spec)
            return {
                fi
                for fi, f in enumerate(self.quads)
                if _eval_face_fn(spec, ar, self.verts[list(f)].mean(axis=0), self.verts[list(f)], self.tags[fi])
            }
        names = {spec} if isinstance(spec, str) else set(spec)
        return {fi for fi, t in enumerate(self.tags) if t in names}

    def tag(self, pred, name: str) -> QuadCage:
        """Tag faces matching `pred` (centroid, `(centroid, tag)`, or `(centroid, verts, tag)`) as `name`."""
        hits = self._match_faces(pred)
        return QuadCage(
            self.verts,
            self.quads,
            [str(name) if i in hits else t for i, t in enumerate(self.tags)],
            self.creases,
            self._fmats,
        )

    def paint_faces(self, pred, material) -> QuadCage:
        """Assign an explicit per-face `material` to faces matching `pred` or tag(s)."""
        hits = self._match_faces(pred)
        return QuadCage(
            self.verts,
            self.quads,
            self.tags,
            self.creases,
            [material if i in hits else m for i, m in enumerate(self._fmats)],
        )

    def keep(self, tags_or_pred) -> QuadCage:
        """Keep only faces matching `tags_or_pred`."""
        hits = self._match_faces(tags_or_pred)
        return self._compact(i for i in range(len(self.quads)) if i in hits)

    def drop(self, tags_or_pred) -> QuadCage:
        """Drop faces matching `tags_or_pred` (clean cutouts for visors, vents, openings)."""
        hits = self._match_faces(tags_or_pred)
        return self._compact(i for i in range(len(self.quads)) if i not in hits)

    def crease(self, edge_pred_or_tag_boundary=None) -> QuadCage:
        """Mark sharp edges (`None`, `'tag_boundary'`, `'boundary'`, `'all'`, tag(s), or callable)."""
        ef: dict[tuple[int, int], list[int]] = {}
        for fi, f in enumerate(self.quads):
            for i in range(len(f)):
                ef.setdefault(_ce(f[i], f[(i + 1) % len(f)]), []).append(fi)
        spec = edge_pred_or_tag_boundary
        if callable(spec):
            ar = _arity(spec)
            added = {
                (u, v)
                for (u, v), fl in ef.items()
                if (
                    spec(0.5 * (self.verts[u] + self.verts[v]))
                    if ar <= 1
                    else (
                        spec(self.verts[u], self.verts[v])
                        if ar == 2
                        else spec(self.verts[u], self.verts[v], tuple(self.tags[i] for i in fl))
                    )
                )
            }
        elif spec in (None, "all", "boundary", "tag_boundary"):
            added = {
                e
                for e, fl in ef.items()
                if (
                    spec == "all"
                    or (spec == "boundary" and len(fl) == 1)
                    or (spec == "tag_boundary" and len(fl) > 1 and len({self.tags[i] for i in fl}) > 1)
                    or (spec is None and (len(fl) == 1 or len({self.tags[i] for i in fl}) > 1))
                )
            }
        else:
            names = {spec} if isinstance(spec, str) else set(spec)
            added = {
                e
                for e, fl in ef.items()
                if ({self.tags[i] for i in fl} & names) and (len(fl) == 1 or len({self.tags[i] for i in fl}) > 1)
            }
        return QuadCage(self.verts, self.quads, self.tags, self.creases | added, self._fmats)

    def ensure_outward(self, center) -> QuadCage:
        """Flip all face windings if average face normal points toward `center` instead of outward."""
        if not self.quads:
            return self
        c = np.asarray(center, dtype=np.float64)
        score = sum(
            float(np.dot(_face_normal([self.verts[k] for k in f]), self.verts[list(f)].mean(axis=0) - c))
            for f in self.quads
        )
        if score >= 0.0:
            return self
        return QuadCage(self.verts, [tuple(reversed(f)) for f in self.quads], self.tags, self.creases, self._fmats)

    def _subdivide_once(self, pin_corners=False) -> QuadCage:
        nv = len(self.verts)
        ef: dict[tuple[int, int], list[int]] = {}
        vf: list[list[int]] = [[] for _ in range(nv)]
        for fi, f in enumerate(self.quads):
            for i in range(len(f)):
                vf[f[i]].append(fi)
                ef.setdefault(_ce(f[i], f[(i + 1) % len(f)]), []).append(fi)
        edges = sorted(ef)
        e_idx = {e: nv + i for i, e in enumerate(edges)}
        sharp = {e for e in edges if e in self.creases or len(ef[e]) != 2}
        ve: list[list[int]] = [[] for _ in range(nv)]
        vsh: list[list[int]] = [[] for _ in range(nv)]
        for u, v in edges:
            ve[u].append(v)
            ve[v].append(u)
            if (u, v) in sharp:
                vsh[u].append(v)
                vsh[v].append(u)
        f_pos = np.array([self.verts[list(f)].mean(axis=0) for f in self.quads], dtype=np.float64)
        e_pos = np.empty((len(edges), 3), dtype=np.float64)
        for ei, (u, v) in enumerate(edges):
            if (u, v) in sharp:
                e_pos[ei] = 0.5 * (self.verts[u] + self.verts[v])
            else:
                f0, f1 = ef[(u, v)]
                e_pos[ei] = 0.25 * (self.verts[u] + self.verts[v] + f_pos[f0] + f_pos[f1])
        v_new = np.empty((nv, 3), dtype=np.float64)
        cos_cut = -math.cos(math.pi / 4.0)
        for v in range(nv):
            sh, nb = vsh[v], ve[v]
            is_corner = False
            if pin_corners and len(sh) == 2:
                d0, d1 = self.verts[sh[0]] - self.verts[v], self.verts[sh[1]] - self.verts[v]
                l0, l1 = float(np.linalg.norm(d0)), float(np.linalg.norm(d1))
                is_corner = l0 > 1e-9 and l1 > 1e-9 and float(np.dot(d0, d1)) / (l0 * l1) > cos_cut
            if is_corner or len(sh) >= 3 or (len(sh) <= 1 and (len(nb) < 3 or not vf[v])):
                v_new[v] = self.verts[v]
            elif len(sh) == 2:
                v_new[v] = 0.75 * self.verts[v] + 0.125 * (self.verts[sh[0]] + self.verts[sh[1]])
            else:
                n = len(nb)
                v_new[v] = (
                    f_pos[vf[v]].mean(axis=0) + self.verts[v] + self.verts[nb].mean(axis=0) + (n - 3.0) * self.verts[v]
                ) / n
        f_off = nv + len(edges)
        nquads = [
            (f[i], e_idx[_ce(f[i], f[(i + 1) % len(f)])], f_off + fi, e_idx[_ce(f[(i - 1) % len(f)], f[i])])
            for fi, f in enumerate(self.quads)
            for i in range(len(f))
        ]
        ncreases = {c for e in self.creases if e in e_idx for c in (_ce(e[0], e_idx[e]), _ce(e[1], e_idx[e]))}
        return QuadCage(
            np.vstack([v_new, e_pos, f_pos]),
            nquads,
            [self.tags[fi] for fi, f in enumerate(self.quads) for _ in f],
            ncreases,
            [self._fmats[fi] for fi, f in enumerate(self.quads) for _ in f],
        )

    def subdivide(self, steps=1, pin_corners=False) -> QuadCage:
        """Deterministic creased Catmull-Clark subdivision (`pin_corners=True` pins sharp >45° corners)."""
        cur = self
        for _ in range(int(steps)):
            cur = cur._subdivide_once(pin_corners=pin_corners)
        return cur

    def thicken(
        self,
        t,
        inner_material="trim",
        rim_material=None,
        bevel=0.0,
        tuck_in=0.0,
    ) -> QuadCage:
        """Turn an open/closed shell into a watertight solid (`t>0` outward, `t<0` inward) or open `tuck_in` lip."""
        nv = len(self.verts)
        if nv == 0 or not self.quads:
            return self
        nrms = _vertex_normals(self.verts, self.quads)
        directed = {(f[i], f[(i + 1) % len(f)]): fi for fi, f in enumerate(self.quads) for i in range(len(f))}
        boundary = [(u, v, fi) for (u, v), fi in directed.items() if (v, u) not in directed]
        b_verts = sorted({u for u, _, _ in boundary} | {v for _, v, _ in boundary})
        t_val, tuck = float(t), float(tuck_in)
        emit_inner = tuck <= 0.0
        thick_eff = abs(t_val) if emit_inner else tuck
        p_out = self.verts + nrms * (thick_eff if emit_inner and t_val >= 0.0 else (0.0 if emit_inner else t_val))
        p_in = p_out - nrms * thick_eff
        bv = min(max(0.0, float(bevel)), thick_eff * 0.48) if boundary else 0.0
        p_outer, sh_pts, sh_map = p_out.copy(), [], {}
        if bv > 1e-6 and b_verts:
            inw: dict[int, list[np.ndarray]] = {u: [] for u in b_verts}
            min_el = {u: float("inf") for u in b_verts}
            for u, v, _ in boundary:
                ed = p_out[v] - p_out[u]
                el = float(np.linalg.norm(ed))
                if el >= 1e-12:
                    min_el[u] = min(min_el[u], el)
                    min_el[v] = min(min_el[v], el)
                    for w in (u, v):
                        ti = np.cross(nrms[w], ed / el)
                        lt = float(np.linalg.norm(ti))
                        if lt > 1e-9:
                            inw[w].append(ti / lt)
            base_sh = 2 * nv if emit_inner else nv
            for u in b_verts:
                if inw[u]:
                    s = np.sum(inw[u], axis=0)
                    ls = float(np.linalg.norm(s))
                    if ls > 1e-9:
                        d_in = s / ls
                        p_outer[u] = p_out[u] + d_in * (
                            min(bv, 0.36 * min_el[u]) * min(1.5, 1.0 / max(float(np.dot(d_in, inw[u][0])), 0.5))
                        )
                sh_map[u] = base_sh + len(sh_pts)
                sh_pts.append(p_out[u] - nrms[u] * bv)

        blocks = [p_outer]
        if emit_inner:
            blocks.append(p_in)
            in_idx = {u: nv + u for u in range(nv)}
        if sh_pts:
            blocks.append(np.asarray(sh_pts, dtype=np.float64))
        if not emit_inner:
            in_idx = {u: nv + len(sh_pts) + i for i, u in enumerate(b_verts)}
            if b_verts:
                blocks.append(np.asarray([p_in[u] for u in b_verts], dtype=np.float64))

        quads, tags, fmats = list(self.quads), list(self.tags), list(self._fmats)
        if emit_inner:
            quads.extend(tuple(in_idx[v] for v in reversed(f)) for f in self.quads)
            tags.extend(self.tags)
            fmats.extend([inner_material] * len(self.quads))
        for u, v, fi in boundary:
            mat = rim_material if rim_material is not None else self._fmats[fi]
            tg = self.tags[fi]
            if bv > 1e-6:
                rim = [(u, sh_map[u], sh_map[v], v)]
                if thick_eff - bv > 1e-5:
                    rim.append((sh_map[u], in_idx[u], in_idx[v], sh_map[v]))
            else:
                rim = [(u, in_idx[u], in_idx[v], v)]
            quads.extend(rim)
            tags.extend([tg] * len(rim))
            fmats.extend([mat] * len(rim))
        return QuadCage(np.vstack(blocks), quads, tags, (), fmats)

    def is_watertight(self) -> tuple[bool, float]:
        """Return `(watertight, signed_volume)` for this cage."""
        edges = {(f[i], f[(i + 1) % len(f)]) for f in self.quads for i in range(len(f))}
        if len(edges) != sum(len(f) for f in self.quads):
            return False, 0.0
        if any((v, u) not in edges for u, v in edges):
            return False, 0.0
        vol = sum(
            float(np.dot(self.verts[f[0]], np.cross(self.verts[f[i]], self.verts[f[i + 1]]))) / 6.0
            for f in self.quads
            for i in range(1, len(f) - 1)
        )
        return vol > 0.0, vol

    def to_mesh(self, material, palette, smooth: bool = False):
        """Convert cage to `(pos f32, nor f32, mat_ids u16, tri u32)` with flat or smooth shading."""
        if not self.quads:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.uint16),
                np.zeros((0, 3), np.uint32),
            )
        mat_ar = _arity(material) if callable(material) else 0
        vnrms = _vertex_normals(self.verts, self.quads) if smooth else None
        pos, nor, pnt = [], [], []

        def emit_tri(ia, ib, ic, mat_id):
            a, b, c = self.verts[ia], self.verts[ib], self.verts[ic]
            fn = np.cross(b - a, c - a)
            ln = float(np.linalg.norm(fn))
            if ln < 1e-11:
                return
            fn = fn / ln
            pos.extend((a, b, c))
            if smooth and vnrms is not None:
                nor.extend((vnrms[ia], vnrms[ib], vnrms[ic]))
            else:
                nor.extend((fn, fn, fn))
            pnt.extend((mat_id, mat_id, mat_id))

        for fi, f in enumerate(self.quads):
            fv = self.verts[list(f)]
            ctr, tag = fv.mean(axis=0), self.tags[fi]
            spec = material if self._fmats[fi] is None else self._fmats[fi]
            mat = _eval_face_fn(spec, mat_ar if spec is material else _arity(spec), ctr, fv, tag) if callable(spec) else spec
            mat_id = palette.id_of(mat)
            if len(f) == 3:
                emit_tri(f[0], f[1], f[2], mat_id)
            elif len(f) == 4:
                cx = float(fv[:, 0].mean())
                if cx < -1e-6:
                    emit_tri(f[0], f[1], f[3], mat_id)
                    emit_tri(f[1], f[2], f[3], mat_id)
                else:
                    emit_tri(f[0], f[1], f[2], mat_id)
                    emit_tri(f[0], f[2], f[3], mat_id)
            else:
                for k in range(1, len(f) - 1):
                    emit_tri(f[0], f[k], f[k + 1], mat_id)

        n_verts = len(pos)
        if not n_verts:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.uint16),
                np.zeros((0, 3), np.uint32),
            )
        tri = np.arange(n_verts, dtype=np.uint32).reshape(-1, 3)
        return (
            np.asarray(pos, dtype=np.float32),
            np.asarray(nor, dtype=np.float32),
            np.asarray(pnt, dtype=np.uint16),
            tri,
        )


def _closest_points_on_triangles(points: np.ndarray, tris: np.ndarray):
    """Exact closest points from `points (M, 3)` onto `tris (N, 3, 3)` (Ericson barycentric projection)."""
    a = tris[:, 0]
    ab = tris[:, 1] - a
    ac = tris[:, 2] - a
    ab_ab = np.einsum("ij,ij->i", ab, ab)
    ab_ac = np.einsum("ij,ij->i", ab, ac)
    ac_ac = np.einsum("ij,ij->i", ac, ac)

    m = len(points)
    out_pts = np.empty((m, 3), dtype=np.float64)
    out_tid = np.empty((m,), dtype=np.int64)
    out_bary = np.empty((m, 3), dtype=np.float64)
    out_dist = np.empty((m,), dtype=np.float64)

    for i, p in enumerate(points):
        ap = p - a
        d1 = np.einsum("ij,ij->i", ab, ap)
        d2 = np.einsum("ij,ij->i", ac, ap)
        bp = p - tris[:, 1]
        d3 = np.einsum("ij,ij->i", ab, bp)
        d4 = np.einsum("ij,ij->i", ac, bp)
        cp = p - tris[:, 2]
        d5 = np.einsum("ij,ij->i", ab, cp)
        d6 = np.einsum("ij,ij->i", ac, cp)

        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4

        # Default: interior projection
        denom = np.where((va + vb + vc) > 1e-20, va + vb + vc, 1.0)
        v = vb / denom
        w = vc / denom
        u = 1.0 - v - w

        # Vertex / edge regions
        m_a = (d1 <= 0.0) & (d2 <= 0.0)
        m_b = (d3 >= 0.0) & (d4 <= d3)
        m_c = (d6 >= 0.0) & (d5 <= d6)
        m_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        m_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        m_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

        t_ab = d1 / np.where(np.abs(d1 - d3) > 1e-20, d1 - d3, 1.0)
        t_ac = d2 / np.where(np.abs(d2 - d6) > 1e-20, d2 - d6, 1.0)
        t_bc = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-20, (d4 - d3) + (d5 - d6), 1.0)

        u = np.where(m_bc, 0.0, u)
        v = np.where(m_bc, 1.0 - t_bc, v)
        w = np.where(m_bc, t_bc, w)

        u = np.where(m_ac, 1.0 - t_ac, u)
        v = np.where(m_ac, 0.0, v)
        w = np.where(m_ac, t_ac, w)

        u = np.where(m_ab, 1.0 - t_ab, u)
        v = np.where(m_ab, t_ab, v)
        w = np.where(m_ab, 0.0, w)

        u = np.where(m_c, 0.0, u)
        v = np.where(m_c, 0.0, v)
        w = np.where(m_c, 1.0, w)

        u = np.where(m_b, 0.0, u)
        v = np.where(m_b, 1.0, v)
        w = np.where(m_b, 0.0, w)

        u = np.where(m_a, 1.0, u)
        v = np.where(m_a, 0.0, v)
        w = np.where(m_a, 0.0, w)

        closest = a + ab * v[:, None] + ac * w[:, None]
        diff = p - closest
        d2_all = np.einsum("ij,ij->i", diff, diff)
        best = int(np.argmin(d2_all))
        out_pts[i] = closest[best]
        out_tid[i] = best
        out_bary[i] = (u[best], v[best], w[best])
        out_dist[i] = math.sqrt(float(d2_all[best]))
    return out_pts, out_tid, out_bary, out_dist


def project_cage(target, cage: QuadCage, offset=0.0, relief_fn=None) -> QuadCage:
    """Snap `cage` vertices onto `target` (a `QuadCage`, `Solid`, or `(N, 3, 3)` triangles), offset along smooth normals."""
    from . import _plane
    from .words import Solid

    if len(cage.verts) == 0:
        return cage
    if isinstance(target, QuadCage):
        pos, _, _, tri = target.to_mesh("body", _DummyPal())
        tris = pos[tri].astype(np.float64)
    elif isinstance(target, Solid):
        pos, _, _, tri = _plane.mesh([(target.parts, 1)], (-_plane.BIG / 2,) * 3, (_plane.BIG / 2,) * 3)
        tris = pos[tri].astype(np.float64)
    else:
        tris = np.asarray(target, dtype=np.float64).reshape(-1, 3, 3)

    if len(tris) == 0:
        return cage

    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    # Weld vertices at 1e-5 m for crease-aware smooth normals (average only faces within 60°)
    keys = np.round(tris.reshape(-1, 3) * 1e5).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv_tri = inv.reshape(-1, 3)
    v_faces: dict[int, list[int]] = {}
    for fi in range(len(tris)):
        for vi in inv_tri[fi]:
            v_faces.setdefault(int(vi), []).append(fi)

    c_pts, tid, bary, _ = _closest_points_on_triangles(cage.verts, tris)
    nrms = np.empty_like(c_pts)
    for i in range(len(c_pts)):
        ti = int(tid[i])
        n0 = fn[ti]
        corner_n = np.empty((3, 3), dtype=np.float64)
        for k in range(3):
            vi = int(inv_tri[ti, k])
            nbrs = [fn[fj] for fj in v_faces[vi] if float(np.dot(fn[fj], n0)) > 0.5]
            s = np.sum(nbrs, axis=0) if nbrs else n0
            ls = float(np.linalg.norm(s))
            corner_n[k] = s / ls if ls > 1e-12 else n0
        ni = bary[i, 0] * corner_n[0] + bary[i, 1] * corner_n[1] + bary[i, 2] * corner_n[2]
        ln = float(np.linalg.norm(ni))
        nrms[i] = ni / ln if ln > 1e-12 else n0
    off = float(offset)
    new_v = np.array(
        [c_pts[i] + nrms[i] * (off + (float(relief_fn(c_pts[i])) if relief_fn else 0.0)) for i in range(len(c_pts))],
        dtype=np.float64,
    )
    quads = cage.quads
    if quads and sum(float(np.dot(_face_normal([new_v[k] for k in f]), nrms[list(f)].mean(axis=0))) for f in quads) < 0.0:
        quads = tuple(tuple(reversed(f)) for f in quads)
    return QuadCage(new_v, quads, cage.tags, cage.creases, cage._fmats)


class _DummyPal:
    def id_of(self, _):
        return 1
