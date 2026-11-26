"""Places for the body words: frames, faces picked by semantic words, and grids/rings on them.

A frame is an origin and three orthonormal axes (x right, y up, z out); a part's seat is its
own z = 0 and `.on(frame)` stands it there. A face's frame has its origin at the middle of its
bounds, z = outward normal, y = up the face (on a level face, away from front: -z), x = right.
"""

from __future__ import annotations

import math

import numpy as np

from . import _plane

LEVEL = 0.95  # |normal . y| past this: a level face, its y away from front
NM = 9
SLIVER = 1e-4  # m^2: faces smaller than 1 cm^2 are ignored by picks
WORLD = {"front": (2, 1), "back": (2, -1), "top": (1, 1), "bottom": (1, -1), "right": (0, 1), "left": (0, -1)}
COS45 = math.cos(math.radians(45)) + 1e-6
FLAT = 0.99  # cos(~8.1 deg): strict world-axis alignment for faces(..., flat=True)


def axes(n):
    """`(x, y)` of a face or frame whose `z` is `n`."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    if abs(n[1]) > LEVEL:
        y = np.array([0.0, 0.0, -1.0])
        y = y - (y @ n) * n
    else:
        y = np.array([0.0, 1.0, 0.0]) - n[1] * n
    y /= np.linalg.norm(y)
    return np.cross(y, n), y


def xf_parts(parts, R, t):
    """Transform convex parts by `p -> R p + t` (`R` rotation or reflection), rounded to `NM` decimals."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    out = []
    for q in parts:
        planes = []
        for n, d, s, sm in q:
            rn = R @ np.asarray(n, float)
            sm = sm if sm is None else (tuple(R @ np.asarray(sm[0], float) + t), tuple(R @ np.asarray(sm[1], float)))
            planes.append((tuple(round(float(c), NM) + 0.0 for c in rn), round(float(d + rn @ t), NM) + 0.0, s, sm))
        out.append(planes)
    return out


class Frame:
    """An origin and axes x, y, z (columns of R); `world` if read off a drawn part, `role` from host."""

    def __init__(self, origin, R, world=False, role=None):
        self.origin = np.asarray(origin, float)
        self.R = np.asarray(R, float)
        self.world = world
        self.role = role

    x = property(lambda s: s.R[:, 0])
    y = property(lambda s: s.R[:, 1])
    z = property(lambda s: s.R[:, 2])

    def at(self, u=0.0, v=0.0, turn=0.0):
        """Frame shifted by `u` along x and `v` along y, turned `turn` degrees about z."""
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        R = self.R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return Frame(self.origin + u * self.x + v * self.y, R, self.world, self.role)

    def lift(self, d):
        """Frame shifted `d` along z (out of a face; negative sinks into it)."""
        return Frame(self.origin + d * self.z, self.R, self.world, self.role)

    def around(self, n, r=0.0, start=0.0):
        """`n` frames on a circle of radius `r` about z, starting at `start` degrees from x."""
        out = []
        for k in range(n):
            a = start + 360.0 * k / n
            f = self.at(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
            out.append(f.at(turn=a))
        return out

    def grid(self, nu, nv=1, pitch=None, w=None, h=None, margin=0.0):
        """`nu x nv` frames along x and y, spaced by `pitch` or spread across `w x h` less `margin`."""
        pu, pv = (pitch, pitch) if isinstance(pitch, (int, float)) else (pitch or (None, None))
        if pu is None:
            pu = (w - 2 * margin) / nu if w is not None else 0.0
        if pv is None:
            pv = (h - 2 * margin) / nv if h is not None else 0.0
        return [self.at((i - (nu - 1) / 2) * pu, (j - (nv - 1) / 2) * pv) for j in range(nv) for i in range(nu)]

    def transform(self):
        return self.R, self.origin


class Face(Frame):
    """A flat face of a solid with its plane, triangles, dimensions `w x h`, and area."""

    def __init__(self, n, d, tris, part=None, world=False, name="", role=None, xy=None):
        n = np.asarray(n, float)
        x, y = axes(n) if xy is None else (np.asarray(xy[0], float), np.asarray(xy[1], float))
        pts = tris.reshape(-1, 3)
        u, v = pts @ x, pts @ y
        mid = ((u.min() + u.max()) / 2) * x + ((v.min() + v.max()) / 2) * y + d * n
        role = part.role if part is not None else role
        super().__init__(mid, np.column_stack([x, y, n]), world, role)
        self.n, self.d, self.tris, self.part, self.name = n, float(d), tris, part, name
        self.w, self.h = float(u.max() - u.min()), float(v.max() - v.min())
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        self.area = float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)

    def grid(self, nu, nv=1, pitch=None, margin=0.0):
        return super().grid(nu, nv, pitch, self.w, self.h, margin)

    def corners(self, inset=0.0):
        """Four corners of the face bounds inset by `inset` (bl, br, tr, tl)."""
        a, b = self.w / 2 - inset, self.h / 2 - inset
        return [self.at(-a, -b), self.at(a, -b), self.at(a, b), self.at(-a, b)]

    def outline(self):
        """Convex hull outline of the face in its own `(x, y)` coordinates."""
        from .words import Outline, _area

        pts = [(float((p - self.origin) @ self.x), float((p - self.origin) @ self.y)) for p in self.tris.reshape(-1, 3)]
        hull = _hull2(pts)
        if abs(_area(hull) - self.area) > 1e-6 * max(1.0, self.area) + 1e-9:
            raise ValueError(
                f"the {self.name or 'face'} is not convex ({self.area:.4f} m2 of its hull's {_area(hull):.4f}): give an outline"
            )
        return Outline(hull)

    def __repr__(self):
        return f"<{self.name} face {self.w:.3f} x {self.h:.3f} at {np.round(self.origin, 3).tolist()}>"


def _hull2(pts):
    pts = sorted(set((round(u, 9), round(v, 9)) for u, v in pts))

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (
                p[0] - out[-2][0]
            ) <= 1e-15:
                out.pop()
            out.append(p)
        return out

    lo, hi = half(pts), half(pts[::-1])
    return lo[:-1] + hi[:-1]


def _islands(t, ks):
    """Split coplanar triangle indices `ks` into vertex-connected components (1e-5 m grid)."""
    if len(ks) <= 1:
        return [ks]
    by_v: dict = {}
    for pos, k in enumerate(ks):
        for v in t[k]:
            key = (int(round(float(v[0]) * 1e5)), int(round(float(v[1]) * 1e5)), int(round(float(v[2]) * 1e5)))
            by_v.setdefault(key, []).append(pos)
    seen = [False] * len(ks)
    comps = []
    for start in range(len(ks)):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            cur = stack.pop()
            comp.append(ks[cur])
            for v in t[ks[cur]]:
                key = (int(round(float(v[0]) * 1e5)), int(round(float(v[1]) * 1e5)), int(round(float(v[2]) * 1e5)))
                for nb in by_v[key]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
        comps.append(comp)
    return comps


def faces_of(solid, part=None, islands=False):
    """Every flat face of `solid` meshed alone, grouped by plane (or disconnected islands)."""
    pos, _, _, tri = _plane.mesh([(solid.parts, 1)], (-_plane.BIG / 2,) * 3, (_plane.BIG / 2,) * 3)
    t = pos[tri].astype(float)
    if not len(t):
        return []
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    planes = sorted({(tuple(pl[0]), pl[1]) for q in solid.parts for pl in q})
    N = np.array([pn for pn, _ in planes], float).reshape(-1, 3)
    D = np.array([d for _, d in planes], float)
    off = np.abs(np.einsum("kvi,pi->kvp", t, N) - D).max(1)
    score = (n @ N.T > 0.5) & (off < 1e-5)
    groups: dict[int, list[int]] = {}
    for k in range(len(tri)):
        hit = np.flatnonzero(score[k])
        if len(hit):
            groups.setdefault(int(hit[0]), []).append(k)
    finish = {tuple(np.round(np.append(pn, pd), 5)) for pn, pd in getattr(solid, "finish", ())}
    out = []
    for k, ks in sorted(groups.items()):
        is_chamfer = tuple(np.round(np.append(N[k], D[k]), 5)) in finish
        for chunk in (_islands(t, ks) if islands else [ks]):
            f = Face(N[k], D[k], t[chunk], part, solid.world)
            f.name = "chamfer" if is_chamfer else names(f.n, solid.R)
            if f.area >= SLIVER:
                out.append(f)
    return out


def names(n, R):
    """Semantic words for a face normal `n` given solid orientation `R`."""
    out = [w for w, (a, s) in WORLD.items() if s * n[a] > COS45]
    m = np.asarray(R, float).T @ n
    out.append("out" if m[2] > COS45 else "seat" if m[2] < -COS45 else "upright")
    return " ".join(out)


def picked(faces, word, flat=False):
    """Filter `faces` matching `word` (e.g. `'front'`, `'top&out'`); `flat=True` requires strict axis alignment."""
    want = word.split("&")
    out = [f for f in faces if all(w in f.name.split() for w in want)]
    if flat:
        axes_ = [WORLD[w] for w in want if w in WORLD]
        out = [f for f in out if (all(s * f.n[a] > FLAT for a, s in axes_) if axes_ else max(abs(f.n)) > FLAT)]
    return out


def flatten(place):
    """Normalize a Frame, Face, or nested list/tuple of them into a flat list of Frames."""
    if isinstance(place, (list, tuple)):
        return [f for p in place for f in flatten(p)]
    if not isinstance(place, Frame):
        raise TypeError(f"not a place: {place!r} (a face, a frame, or a list of them)")
    return [place]


def frame(at=(0, 0, 0), z="+z", x=None):
    """Create a local Frame at `at` with z along `z` (e.g. `'+x'`, `'-y'`, or vector)."""
    z_v = _vec(z)
    if x is None:
        xa, ya = axes(z_v)
    else:
        xa = _vec(x)
        ya = np.cross(z_v, xa)
    return Frame(at, np.column_stack([xa, ya, z_v]))


def floor(x=0.0, z=0.0):
    """Floor frame at `(x, 0, z)` with +z pointing straight up (+Y)."""
    return frame((x, 0.0, z), "+y")


def around(n, start=0.0):
    """`n` frames turned about the active frame's z axis, starting at `start` degrees."""
    return Frame((0, 0, 0), np.eye(3)).around(n, 0.0, start)


def _vec(v):
    if isinstance(v, str):
        e = np.zeros(3)
        e["xyz".index(v[1])] = 1.0 if v[0] == "+" else -1.0
        return e
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def mirror_of(frame_, axis):
    """`(R, t)` reflection across `frame_`'s plane perpendicular to `axis` (`'x'`, `'y'`, or `'z'`)."""
    N = frame_.R[:, "xyz".index(axis)]
    return np.eye(3) - 2 * np.outer(N, N), 2 * (frame_.origin @ N) * N
