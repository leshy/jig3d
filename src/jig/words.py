"""The hard-surface CAD vocabulary: 2D outlines and 3D solids built from half-space planes.

One argument order: what, how big (across, up, out), how many sides; then finish (`.chamfer`,
`.corner`), then placement (`.move`, `.turn`, `.mirror`, `.on`, `.into`); material only at `p.put`.
"""

from __future__ import annotations

import inspect
from itertools import permutations

import numpy as np

from . import _places, _plane
from .solid import Shape, box_shape, hull_parts, plane

NM = 9  # decimals planes are rounded to (1 nm)
PROBE = 1e-4  # metres off an edge where `chamfer` tests convexity
SOFT = 20  # degrees: edges turning less than this stay unchamfered


# --- 2D outlines ---


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _area(pts):
    return sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1])) / 2


def _convex_loop(pts, idx):
    k = len(idx)
    return all(_cross(pts[idx[i - 1]], pts[idx[i]], pts[idx[(i + 1) % k]]) >= -1e-12 for i in range(k))


def _join(a, b):
    for k in range(len(a)):
        x, y = a[k], a[(k + 1) % len(a)]
        for m in range(len(b)):
            if b[m] == y and b[(m + 1) % len(b)] == x:
                return a[: k + 1] + [b[(m + 2 + t) % len(b)] for t in range(len(b) - 2)] + a[k + 1 :]
    return None


def _pieces(pts):
    n = len(pts)
    if _convex_loop(pts, list(range(n))):
        return [list(range(n))]
    idx, out = list(range(n)), []
    while len(idx) > 3:
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if _cross(pts[a], pts[b], pts[c]) <= 1e-12:
                continue
            if any(
                min(_cross(pts[a], pts[b], pts[i]), _cross(pts[b], pts[c], pts[i]), _cross(pts[c], pts[a], pts[i])) > 1e-12
                for i in idx
                if i not in (a, b, c)
            ):
                continue
            out.append([a, b, c])
            del idx[k]
            break
        else:
            raise ValueError("poly: the outline crosses itself")
    out.append(idx)
    joined = True
    while joined:
        joined = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                u = _join(out[i], out[j])
                if u and _convex_loop(pts, u):
                    out[i], joined = u, True
                    del out[j]
                    break
            if joined:
                break
    return out


def _clean(pts):
    out = []
    for q in pts:
        if not out or abs(q[0] - out[-1][0]) + abs(q[1] - out[-1][1]) > 1e-12:
            out.append(q)
    return out[:-1] if len(out) > 1 and abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1]) <= 1e-12 else out


def _bisected(pts):
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        if _cross(a, b, c) >= -1e-12:
            continue
        d1, d2 = np.subtract(b, a), np.subtract(c, b)
        r = d1 / np.linalg.norm(d1) - d2 / np.linalg.norm(d2)
        best = None
        for j in range(n):
            if j in (i, (i - 1) % n):
                continue
            p, q = np.array(pts[j]), np.array(pts[(j + 1) % n])
            m = np.array([[r[0], p[0] - q[0]], [r[1], p[1] - q[1]]])
            if abs(np.linalg.det(m)) < 1e-15:
                continue
            t, s = np.linalg.solve(m, p - np.array(b))
            if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best[0]):
                best = (t, j, s)
        if best is None:
            raise ValueError("poly: the outline crosses itself")
        t, j, s = best
        hit = tuple(np.array(pts[j]) + s * (np.array(pts[(j + 1) % n]) - np.array(pts[j])))
        one = [pts[(i + k) % n] for k in range((j - i) % n + 1)]
        two = [pts[(j + 1 + k) % n] for k in range((i - j - 1) % n + 1)]
        if s > 1 - 1e-9:
            one, two = one + [pts[(j + 1) % n]], two
        elif s < 1e-9:
            two = [pts[j]] + two
        else:
            one, two = one + [hit], [hit] + two
        return _bisected(_clean(one)) + _bisected(_clean(two))
    return [list(pts)]


def _left(p, q):
    eu, ev = q[0] - p[0], q[1] - p[1]
    k = (eu * eu + ev * ev) ** 0.5
    return (-ev / k, eu / k)


def _clipped(pts, d):
    poly_ = [(p[0], p[1], 0.0) for p in pts]
    n = len(pts)
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        nu, nv = _left(a, b)
        poly_ = _plane._clip(poly_, (-nu, -nv, 0.0), -(nu * a[0] + nv * a[1]) - d)
    return _clean([(p[0], p[1]) for p in poly_])


class Outline:
    """A closed 2D outline (u across, v up), counter-clockwise."""

    def __init__(self, pts, pieces=None, band=None):
        pts = [(float(u), float(v)) for u, v in pts]
        if pieces is None and _area(pts) < 0:
            pts = pts[::-1]
        self.pts, self.loop = pts, pieces is None
        self._pieces = pieces
        self._band = band

    def pieces(self):
        if self._pieces is None:
            self._pieces = _pieces(self.pts)
        return self._pieces

    def convex(self):
        if not self.loop:
            return [[self.pts[i] for i in q] for q in self.pieces()]
        return _bisected(self.pts)

    def at(self, u, v):
        """The same outline moved by `(u, v)`."""
        band = (self._band[0].at(u, v), self._band[1]) if self._band is not None else None
        out = Outline([(a + u, b + v) for a, b in self.pts], self._pieces, band)
        out.loop = self.loop
        return out

    def inset(self, d):
        """Every edge moved `d` inward (outward if negative), corners mitred."""
        if d == 0:
            return self
        if not self.loop:
            raise ValueError("inset: a band has no single inside; inset the outline, then band it")
        n, out = len(self.pts), []
        inward = [_left(self.pts[k], self.pts[(k + 1) % n]) for k in range(n)]
        for j in range(n):
            n0, n1 = inward[j - 1], inward[j]
            det = n0[0] * n1[1] - n0[1] * n1[0]
            m = n0 if abs(det) < 1e-12 else ((n1[1] - n0[1]) / det, (n0[0] - n1[0]) / det)
            out.append((self.pts[j][0] + d * m[0], self.pts[j][1] + d * m[1]))
        for k in range(n):
            (a, b), (c, e) = self.pts[k], self.pts[(k + 1) % n]
            (p, q), (r, s) = out[k], out[(k + 1) % n]
            if (c - a) * (r - p) + (e - b) * (s - q) <= 0:
                if d > 0 and _convex_loop(self.pts, list(range(n))):
                    return Outline(_clipped(self.pts, d))
                raise ValueError(f"inset({d}): deeper than edge {k} is long; inset less or cut smaller corners")
        return Outline(out)

    def band(self, w):
        """The strip of width `w` between the outline and its inset."""
        w = float(w)
        if w <= 0:
            raise ValueError(f"band({w:g}): width must be positive")
        inner = self.inset(w)
        n = len(self.pts)
        if len(inner.pts) != n:
            raise ValueError(f"band({w:g}): inset dropped corners ({n} -> {len(inner.pts)}); use a smaller width")
        return Outline(self.pts + inner.pts, [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)], (self, w))


def _outline(o) -> Outline:
    return o if isinstance(o, Outline) else Outline(o)


def rect(w, h, cut=0) -> Outline:
    """A `w x h` rectangle; `cut` clips corners at 45°: scalar, `(top, bottom)`, or `(tl, tr, br, bl)`."""
    c = (
        (cut,) * 4
        if isinstance(cut, (int, float))
        else (cut[0], cut[0], cut[1], cut[1])
        if len(cut) == 2
        else tuple(cut)
    )
    x, y = w / 2, h / 2
    tl, tr, br, bl = c
    corners = (
        ((-x, -y + bl), (-x + bl, -y)),
        ((x - br, -y), (x, -y + br)),
        ((x, y - tr), (x - tr, y)),
        ((-x + tl, y), (-x, y - tl)),
    )
    pts = []
    for (a, b), k in zip(corners, (bl, br, tr, tl)):
        pts += [a, b] if k else [a]
    return Outline(pts)


def ngon(sides, r, flat="+y") -> Outline:
    """Regular `sides`-gon with apothem `r` (centre to flat), one flat facing `flat` (`+y`, `-y`, `+x`, `-x`)."""
    face = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}[flat]
    R = r / np.cos(np.pi / sides)
    return Outline(
        [
            (
                R * np.cos(np.radians(face + 180 / sides + 360 * k / sides)),
                R * np.sin(np.radians(face + 180 / sides + 360 * k / sides)),
            )
            for k in range(sides)
        ]
    )


def poly(points) -> Outline:
    """Any simple polygon `[(u, v), ...]`, concave allowed, either winding."""
    return Outline(points)


def stripes(region, pitch=0.02, turn=45.0) -> list[Outline]:
    """Diagonal stripe outlines across a convex `region` (for `p.mark`)."""
    o = _outline(region)
    c, s_ = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    across = np.array([-s_, c])
    proj = [np.array(q) @ across for q in o.pts]
    out = []
    for k in range(int(np.floor(min(proj) / pitch)), int(np.ceil(max(proj) / pitch)) + 1):
        p = [(q[0], q[1], 0.0) for q in o.pts]
        lo, hi = k * pitch, k * pitch + pitch / 2
        p = _plane._clip(p, (-across[0], -across[1], 0.0), -lo)
        p = _plane._clip(p, (across[0], across[1], 0.0), hi)
        if len(p) >= 3 and abs(_area([(q[0], q[1]) for q in p])) > 1e-9:
            out.append(Outline([(q[0], q[1]) for q in p]))
    return out


# --- 3D solids ---


def _round(pl):
    n, d, strict, smooth = pl
    return (tuple(round(float(c), NM) + 0.0 for c in n), round(float(d), NM) + 0.0, strict, smooth)


def _part(planes):
    out = []
    for pl in map(_round, planes):
        if not any(pl[0] == q[0] and pl[1] == q[1] for q in out):
            out.append(pl)
    return out


class Solid(Shape):
    """Composable half-space solid supporting `.chamfer`, `.corner`, `.move`, `.turn`, `.mirror`, `.on`, `.into`, `.face`, `.faces`, `&`, `|`."""

    def __init__(self, parts, R=None, world=False, finish=()):
        super().__init__(parts)
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.world = world
        self.finish = tuple(finish)

    def _keep(self, parts, finish=()):
        return Solid(parts, self.R, self.world, self.finish + tuple(finish))

    def __and__(self, o):
        if not isinstance(o, Shape):
            return NotImplemented
        return self._keep([q for a in self.parts for b in o.parts for q in [_part(list(a) + list(b))] if _plane._faces(q)])

    def __or__(self, o):
        if not isinstance(o, Shape):
            return NotImplemented
        return Solid(
            self.parts + [_part(q) for q in o.parts],
            self.R,
            self.world or getattr(o, "world", False),
            self.finish + getattr(o, "finish", ()),
        )

    def _xf(self, R, t, world=None):
        finish = [pl[:2] for q in _places.xf_parts([[(n, d, True, None) for n, d in self.finish]], R, t) for pl in q]
        return Solid(
            _places.xf_parts(self.parts, R, t),
            np.asarray(R) @ self.R,
            self.world if world is None else world,
            finish,
        )

    def move(self, x=0.0, y=0.0, z=0.0):
        """Translate by `(x, y, z)`."""
        return self._xf(np.eye(3), (x, y, z))

    def turn(self, axis, deg, about=(0, 0, 0)):
        """Rotate `deg` degrees (right-handed) about `axis` (`'x'`, `'y'`, `'z'`) through `about`."""
        c, s, k = np.cos(np.radians(deg)), np.sin(np.radians(deg)), "xyz".index(axis)
        R = np.eye(3)
        i, j = (k + 1) % 3, (k + 2) % 3
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        about_v = np.asarray(about, float)
        return self._xf(R, about_v - R @ about_v)

    def mirror(self, axis):
        """Mirror across the plane through the origin perpendicular to `axis`."""
        R = np.eye(3)
        R["xyz".index(axis)] *= -1
        return self._xf(R, (0, 0, 0))

    def on(self, place, sink=0.0, lift=0.0):
        """Stand seat (`z = 0`) on `place` (Face, Frame, or list); `sink` into or `lift` off it."""
        return self._placed(place, lift - sink, 1.0)

    def into(self, place, lift=0.0):
        """Orient as a cutter directed into `place` (`lift` starts it slightly outside)."""
        return self._placed(place, lift, -1.0)

    def _placed(self, place, off, zs):
        copies = [self._xf(f.R @ np.diag([1.0, 1.0, zs]), f.origin + off * f.z, f.world) for f in _places.flatten(place)]
        out = copies[0]
        for c in copies[1:]:
            out = out | c
        return out

    def faces(self, word=None, by=None, flat=False, islands=False):
        """Return matching faces (`'front'`, `'out'`, `'top&out'`, ...), sorted by `by` axis or descending area."""
        got = _places.faces_of(self, getattr(self, "part", None), islands=islands)
        if word is not None:
            got = _places.picked(got, word, flat=flat)
        elif flat:
            got = [f for f in got if max(abs(f.n)) > _places.FLAT]
        if by:
            return sorted(got, key=lambda f: f.origin["xyz".index(by)])
        return sorted(got, key=lambda f: -f.area)

    def face(self, word, flat=False):
        """Pick one face by semantic `word`; returns largest if >= 2x next, else raises with available faces."""
        got = self.faces(word, flat=flat)
        if len(got) > 1 and got[0].area >= 2 * got[1].area:
            return got[0]
        if len(got) != 1:
            every = ", ".join(f"{f.name} {f.area:.4f} m2" for f in self.faces())
            raise ValueError(
                f"face({word!r}): {len(got)} faces; this solid's faces are {every}"
                + ("; faces(word, by=) gives them in order" if got else "")
            )
        return got[0]

    def _inside(self, x):
        return any(all(n[0] * x[0] + n[1] * x[1] + n[2] * x[2] <= d + 1e-9 for n, d, *_ in q) for q in self.parts)

    def _edges(self):
        out = []
        for i, q in enumerate(self.parts):
            faces = _plane._faces(q)
            for x, (ka, poly_) in enumerate(faces):
                for kb, _ in faces[x + 1 :]:
                    nb, db = np.array(q[kb][0]), q[kb][1]
                    on = [np.array(p) for p in poly_ if abs(nb @ np.array(p) - db) < 1e-7]
                    if len(on) < 2:
                        continue
                    mid = (on[0] + on[-1]) / 2
                    na = np.array(q[ka][0])
                    if not any(self._inside(mid + PROBE * v) for v in (na - nb, nb - na)):
                        out.append((i, q[ka], q[kb], mid))
        return out

    def chamfer(self, b, where="shown"):
        """Cut picked convex edges flat: `b` back along both faces, or `(a, b)` for `'face1&face2'`."""
        a, b = (b, b) if isinstance(b, (int, float)) else b
        words = where.split("&")
        if a != b and len(words) != 2:
            raise ValueError(f"chamfer(({a}, {b}), {where!r}): asymmetric chamfer needs 'face1&face2'")
        cuts, seen = [], set()
        for i, pa, pb, mid in self._edges():
            if np.dot(pa[0], pb[0]) > np.cos(np.radians(SOFT)):
                continue
            na, nb = (_places.names(np.array(p[0]), self.R).split() for p in (pa, pb))
            seen.add("&".join(sorted((na[0], nb[0]))))
            if not _picked(na, nb, words):
                continue
            if len(words) == 2 and words[0] != words[1] and words[0] not in na:
                pa, pb = pb, pa
            cuts.append((i, _bevel(pa, pb, mid, a, b)))
        if not cuts:
            raise ValueError(f"chamfer({where!r}): no such edge; available: {', '.join(sorted(seen))}")
        return self._cut(cuts)

    def corner(self, d, where):
        """Cut the 3-face corner `'face1&face2&face3'` flat, `d` back along each edge."""
        words = where.split("&")
        if len(words) != 3:
            raise ValueError(f"corner({where!r}): a corner needs three faces, e.g. 'front&top&right'")
        cuts, seen = [], set()
        for q in self.parts:
            for v in sorted({tuple(np.round(p, 9)) for _, f in _plane._faces(q) for p in f}):
                v_arr = np.array(v)
                at = [pl for pl in q if abs(np.array(pl[0]) @ v_arr - pl[1]) < 1e-7]
                for k, p1 in enumerate(at):
                    for m, p2 in enumerate(at[k + 1 :], k + 1):
                        for p3 in at[m + 1 :]:
                            ns = [np.array(p[0]) for p in (p1, p2, p3)]
                            if abs(np.linalg.det(ns)) < 1e-9 or any(
                                self._inside(v_arr + PROBE * (2 * n - sum(ns))) for n in ns
                            ):
                                continue
                            got = [_places.names(n, self.R).split() for n in ns]
                            seen.add("&".join(sorted(g[0] for g in got)))
                            if _covers(got, words):
                                cuts.append((v_arr, _snip(ns, v_arr, d)))
        if not cuts:
            raise ValueError(f"corner({where!r}): no such corner; available: {', '.join(sorted(seen))}")
        out, done = [], set()
        for v_arr, pl in cuts:
            key = tuple(np.round(v_arr, 9))
            if key in done:
                continue
            done.add(key)
            out += [(i, pl) for i, q in enumerate(self.parts) if all(np.array(n) @ v_arr <= dd + 1e-7 for n, dd, *_ in q)]
        return self._cut(out)

    def _cut(self, cuts):
        parts = [list(q) for q in self.parts]
        for i, pl in cuts:
            parts[i].append(pl)
        return self._keep([q for q in map(_part, parts) if _plane._faces(q)], [_round(pl)[:2] for _, pl in cuts])


def _covers(got, words):
    return any(all(w in g for w, g in zip(words, order)) for order in permutations(got))


def _picked(na, nb, words):
    if words == ["all"]:
        return True
    if words == ["shown"]:
        return "seat" not in na and "seat" not in nb
    if words == ["upright"]:
        return "upright" in na and "upright" in nb
    if len(words) == 1:
        return words[0] in na or words[0] in nb
    return _covers([na, nb], words)


def _bevel(pa, pb, mid, a, b):
    na, nb = np.array(pa[0]), np.array(pb[0])
    e = np.cross(na, nb)
    e /= np.linalg.norm(e)
    ta, tb = np.cross(e, na), np.cross(e, nb)
    ta, tb = (-ta if ta @ nb > 0 else ta), (-tb if tb @ na > 0 else tb)
    qa, qb = mid + a * ta, mid + b * tb
    c = np.cross(e, qb - qa)
    c = -c if c @ (na + nb) < 0 else c
    return plane(c, c @ qa)


def _snip(ns, v, d):
    pts = []
    for k in range(3):
        e = np.cross(ns[(k + 1) % 3], ns[(k + 2) % 3])
        e /= np.linalg.norm(e)
        pts.append(v + d * (-e if e @ ns[k] > 0 else e))
    c = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    c = -c if c @ sum(ns) < 0 else c
    return plane(c, c @ pts[0])


def _loft(sections, axes):
    sections = [(float(z), _outline(o)) for z, o in sections]
    if len(sections) < 2:
        raise ValueError("loft: give at least two (depth, outline) sections")
    counts = [len(o.pts) for _, o in sections]
    if len(set(counts)) > 1:
        raise ValueError(
            f"loft: sections of {counts} corners; each needs as many as the first (rect with cut has 8, without has 4)"
        )
    if any(z1 <= z0 for (z0, _), (z1, _) in zip(sections, sections[1:])):
        raise ValueError(f"loft: depths {[z for z, _ in sections]} must rise")

    def put3(u, v, w):
        p = [0.0, 0.0, 0.0]
        p[axes[0]], p[axes[1]], p[axes[2]] = u, v, w
        return p

    parts = []
    for (z0, o0), (z1, o1) in zip(sections, sections[1:]):
        if o0.pts == o1.pts:
            for piece in o0.convex():
                parts.append(_part(hull_parts([put3(*q, z0) for q in piece] + [put3(*q, z1) for q in piece])[0]))
            continue
        for piece in sections[0][1].pieces():
            pts = [put3(*o0.pts[i], z0) for i in piece] + [put3(*o1.pts[i], z1) for i in piece]
            parts.append(_part(hull_parts(pts)[0]))
    return Solid(parts)


def loft(*sections) -> Solid:
    """Loft through `(depth, outline)` sections with rising depths and matching corner counts."""
    return _loft(sections, (0, 1, 2))


_ALONG = {
    "z": ((0, 1, 2), 1), "+z": ((0, 1, 2), 1), "-z": ((0, 1, 2), -1),
    "x": ((2, 1, 0), 1), "+x": ((2, 1, 0), 1), "-x": ((2, 1, 0), -1),
    "y": ((0, 2, 1), 1), "+y": ((0, 2, 1), 1), "-y": ((0, 2, 1), -1),
}


def _is_num(v):
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)


def _draft(o: Outline, draft):
    if o.loop:
        if not _is_num(draft):
            raise ValueError(f"extrude: draft={draft!r} must be a number for a closed outline")
        return o.inset(draft)
    if _is_num(draft):
        d_out, d_in = float(draft), 0.0
    elif isinstance(draft, (tuple, list)) and len(draft) == 2 and all(_is_num(v) for v in draft):
        d_out, d_in = float(draft[0]), float(draft[1])
    else:
        raise ValueError(f"extrude: band draft={draft!r} must be a number or (outer, inner)")
    if d_out == 0.0 and d_in == 0.0:
        return o
    if o._band is None:
        raise ValueError("extrude: draft on a non-loop outline requires one made with .band(w)")
    base, w = o._band
    if d_out + d_in >= w - 1e-12:
        raise ValueError(f"extrude: band draft ({d_out:g}, {d_in:g}) collapses band width {w:g}")
    n = len(base.pts)
    outer = o.pts[:n] if d_out == 0.0 else base.inset(d_out).pts
    inner = o.pts[n:] if d_in == 0.0 else base.inset(w - d_in).pts
    if len(outer) != n or len(inner) != n:
        raise ValueError(f"extrude: band draft ({d_out:g}, {d_in:g}) dropped corners; use smaller draft or larger cut")
    return Outline(outer + inner, o._pieces)


def extrude(outline, depth, draft=0, along="z", center=False) -> Solid:
    """Extrude `outline` by `depth` along `along` (`±x`/`±y`/`±z`); `draft` tapers inward toward out."""
    if not isinstance(along, str) or along not in _ALONG:
        raise ValueError(f"extrude: unknown along={along!r} (expected 'x', 'y', 'z', '-x', '-y', or '-z')")
    o = _outline(outline)
    top = _draft(o, draft)
    axes, sign = _ALONG[along]
    if sign > 0:
        w0, w1 = (-depth / 2, depth / 2) if center else (0, depth)
        s = _loft(((w0, o), (w1, top)), axes)
    else:
        w0, w1 = (-depth / 2, depth / 2) if center else (-depth, 0)
        s = _loft(((w0, top), (w1, o)), axes)
    if axes != (0, 1, 2) or sign < 0:
        R = np.zeros((3, 3))
        R[axes[0], 0], R[axes[1], 1], R[axes[2], 2] = 1.0, 1.0, float(sign)
        s.R = R
    return s


def body(front, side=None, top=None, depth=None, draft=0) -> Solid:
    """Intersect front, side `(out, up)`, and top `(across, out)` orthographic profiles into a sculpted solid."""
    front = _outline(front)
    side = None if side is None else _outline(side)
    top = None if top is None else _outline(top)
    outs = ([u for u, _ in side.pts] if side else []) + ([v for _, v in top.pts] if top else [])
    if depth is None:
        if not outs:
            raise ValueError("body: give a side, a top, or a depth")
        z0, z1 = min(0.0, min(outs)), max(outs)
    else:
        z0, z1 = (min(0.0, min(outs)) if outs else 0.0), float(depth)
    if z1 <= z0:
        raise ValueError(f"body: out span [{z0}, {z1}] must rise")
    s = _loft(((z0, front), (z1, front.inset(draft))), (0, 1, 2))
    us, vs = [u for u, _ in front.pts], [v for _, v in front.pts]
    if side is not None:
        s = s & _loft(((min(us) - 1, side), (max(us) + 1, side)), (2, 1, 0))
    if top is not None:
        s = s & _loft(((min(vs) - 1, top), (max(vs) + 1, top)), (0, 2, 1))
    return s


def block(w, h, d) -> Solid:
    """A `w x h x d` block seated on `z = 0`, centred on `(x, y)`."""
    return extrude(rect(w, h), d)


def box(x0, x1, y0, y1, z0, z1) -> Solid:
    """Axis-aligned box by explicit world bounds `x0..x1, y0..y1, z0..z1` as a Solid."""
    return Solid(box_shape(x0, x1, y0, y1, z0, z1).parts)


def hull(pts) -> Solid:
    """Convex hull of 3D points as a Solid."""
    return Solid([_part(hull_parts(pts)[0])])


def lathe(profile, sides, flat="+y") -> Solid:
    """Revolve `[(along, r), ...]` about z with `sides` flats (`r` to a flat; `0` closes into a cone)."""
    profile = [(float(z), float(r)) for z, r in profile]
    if any(z1 <= z0 for (z0, _), (z1, _) in zip(profile, profile[1:])):
        raise ValueError(f"lathe: along coordinates {[z for z, _ in profile]} must rise")
    parts = []
    for (z0, r0), (z1, r1) in zip(profile, profile[1:]):
        pts = [(*q, z0) for q in ngon(sides, r0, flat).pts] + [(*q, z1) for q in ngon(sides, r1, flat).pts]
        parts.append(_part(hull_parts(pts)[0]))
    return Solid(parts)


def _turning(a, b):
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def sweep(section, path, closed=False, up=None) -> Solid:
    """Sweep 2D `section` along 3D `path` with mitred corners and twist-free parallel transport."""
    o = _outline(section)
    pts = [np.asarray(q, float) for q in path]
    legs = [
        (a, b)
        for a, b in (list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed else []))
        if np.linalg.norm(b - a) > 1e-12
    ]
    if not legs:
        raise ValueError("sweep: path has no non-zero leg")
    dirs = [(b - a) / np.linalg.norm(b - a) for a, b in legs]
    if up is None:
        y = _places.axes(dirs[0])[1]
    else:
        y = np.asarray(up, float) - (np.asarray(up, float) @ dirs[0]) * dirs[0]
        if np.linalg.norm(y) < 1e-9:
            raise ValueError(f"sweep: first leg runs along up {tuple(up)}: pass another up=")
        y /= np.linalg.norm(y)
    x = np.cross(dirs[0], y)
    parts = []
    for k, ((a, b), t) in enumerate(zip(legs, dirs)):
        if k:
            R = _turning(dirs[k - 1], t)
            x, y = R @ x, R @ y
        span = np.linalg.norm(b - a)
        ends = []
        for end, other in (
            (a, dirs[k - 1] if (k or closed) else None),
            (b, dirs[(k + 1) % len(dirs)] if (k + 1 < len(dirs) or closed) else None),
        ):
            if other is None:
                m = t
            else:
                s = t + other
                sn = np.linalg.norm(s)
                m = t if sn < 1e-12 else s / sn
            ends.append((m, end))
        for piece in o.pieces():
            q = [a - span * t + o.pts[i][0] * x + o.pts[i][1] * y for i in piece] + [
                b + span * t + o.pts[i][0] * x + o.pts[i][1] * y for i in piece
            ]
            planes = list(hull_parts(q)[0])
            (m0, e0), (m1, e1) = ends
            planes += [plane(-m0, -(m0 @ e0)), plane(m1, m1 @ e1)]
            parts.append(_part(planes))
    return Solid(parts)


def ring(r, section, sides) -> Solid:
    """Closed ring of radius `r` around z with cross-section `section` in `sides` straight segments."""
    R = r / np.cos(np.pi / sides)
    circle = [(R * np.cos(np.pi * 2 * k / sides), R * np.sin(np.pi * 2 * k / sides), 0.0) for k in range(sides)]
    return sweep(section, circle, closed=True, up=(0, 0, 1))


def bolt(across, proud=None, sides=6) -> Solid:
    """Fastener head `across` flats, standing `proud` (default `across / 3`)."""
    return lathe([(0, across / 2), (proud if proud is not None else across / 3, across / 2)], sides)


# --- introspection / live vocabulary page ---


def tris(solid: Solid) -> int:
    """Triangle count of `solid` meshed alone."""
    return len(_plane.mesh([(solid.parts, 1)], (-_plane.BIG / 2,) * 3, (_plane.BIG / 2,) * 3)[3])


def volume(solid: Solid) -> float:
    """Signed volume (m^3) of `solid` meshed alone."""
    pos, _, _, tri = _plane.mesh([(solid.parts, 1)], (-_plane.BIG / 2,) * 3, (_plane.BIG / 2,) * 3)
    if not len(tri):
        return 0.0
    a, b, c = (pos[tri[:, k]].astype(float) for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


_BOX = "extrude(rect(0.3, 0.2), 0.1)"
WORDS: dict[str, tuple[str, str]] = {
    "rect": ("0: an outline", "extrude(rect(0.3, 0.2, cut=(0.06, 0.02)), 0.05)"),
    "ngon": ("0: an outline", "extrude(ngon(8, 0.05), 0.04)"),
    "poly": ("0: an outline", "extrude(poly([(0, 0), (0.3, 0), (0.3, 0.1), (0.1, 0.1), (0.1, 0.25), (0, 0.25)]), 0.05)"),
    "Outline.inset": ("0", "extrude(rect(0.3, 0.2, cut=0.04).inset(0.02), 0.03)"),
    "Outline.at": ("0", "extrude(rect(0.1, 0.1).at(0.2, 0), 0.03)"),
    "Outline.band": ("8n extruded", "extrude(rect(0.3, 0.2, cut=0.04).band(0.01), 0.01)"),
    "extrude": ("4n - 4", "extrude(rect(0.3, 0.2, cut=0.04), 0.1, draft=0.02)"),
    "body": ("orthographic intersection", "body(rect(0.46, 0.68, cut=0.07).at(0, 0.34), [(0, 0), (0.23, 0.012), (0.14, 0.34), (0.14, 0.68), (0, 0.68)], draft=0.08)"),
    "loft": ("2n per step + ends", "loft((0, rect(0.46, 0.68, cut=0.07)), (0.1, rect(0.4, 0.62, cut=0.06)), (0.14, rect(0.34, 0.56, cut=0.05)))"),
    "Solid.chamfer": ("+4 per edge", _BOX + ".chamfer(0.01)"),
    "Solid.corner": ("+4", _BOX + ".corner(0.03, 'front&top&right')"),
    "Solid.move": ("0", _BOX + ".move(0, 0.1, 0)"),
    "Solid.turn": ("0", _BOX + ".turn('y', 30)"),
    "Solid.mirror": ("0", _BOX + ".mirror('x')"),
    "Solid.on": ("copies per place; laid drops seat", "bolt(0.02).on(" + _BOX + ".face('front').grid(2, 2, pitch=0.1))"),
    "Solid.into": ("cutter into face", "block(0.1, 0.01, 0.01).into(" + _BOX + ".face('front'))"),
    "Solid.face": ("0", _BOX + ".face('top&upright').at(0, 0)"),
    "Solid.faces": ("0", _BOX + ".faces('upright', by='x')"),
    "block": ("12; 10 seated", "block(0.2, 0.1, 0.05)"),
    "box": ("12; 10 seated", "box(-0.1, 0.1, 0, 0.1, 0, 0.05)"),
    "lathe": ("2*sides per step + ends", "lathe([(0, 0.05), (0.1, 0.05), (0.13, 0.03)], 8)"),
    "sweep": ("2*corners per leg + ends", "sweep(ngon(6, 0.01), [(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0)])"),
    "ring": ("2*corners*sides", "ring(0.1, rect(0.01, 0.02), 12)"),
    "bolt": ("20 alone, 16 laid", "bolt(0.02)"),
    "stripes": ("list of outlines for p.mark", "stripes(rect(0.1, 0.06), 0.03)"),
    "p.paint": ("0", "p.paint(f, 'slate')"),
    "p.inset": ("16 rect, 32 oct; +8 mouth", "p.inset(f, rect(0.1, 0.05), 0.02, mouth=0.01)"),
    "p.slots": ("16 + 12 per bar", "p.slots(f, rect(0.1, 0.09), 5, 0.02, mouth=0.003)"),
    "p.mark": ("4n - 4 each", "p.mark(f, stripes(rect(0.1, 0.06), 0.03), 'hazard')"),
    "p.cut": ("16 slot; role= 0", "p.cut(block(0.2, 0.01, 0.01).into(f), role='trim')"),
    "p.slot": ("0 (glTF slot node)", "p.slot('screen', f)"),
    "p.on": ("0", "p.on(frame(z='+x'))"),
    "p.mirror": ("0", "p.mirror('z')"),
    "QuadCage": ("organic SubD cage", "QuadCage.ellipsoid(0.14, 0.16, 0.18, nu=8, nv=6).subdivide(1)"),
}


def _cost(ex: str):
    from .scene import Model
    from .sculpt import QuadCage

    ns = {**globals(), "frame": _places.frame, "floor": _places.floor, "around": _places.around, "QuadCage": QuadCage}
    if not ex.startswith("p."):
        got = eval(ex, ns)
        if isinstance(got, Solid):
            return tris(got)
        if isinstance(got, QuadCage):
            return sum(len(f) - 2 for f in got.quads)
        return f"{len(got)} items" if isinstance(got, list) else "a place"
    if ex.startswith(("p.on", "p.mirror")):
        return 0
    counts = []
    for stated in ("", ex):
        p = Model(laid=False)
        f = p.put(block(0.3, 0.2, 0.1), "body").face("front")
        eval(stated or "None", ns, {"p": p, "f": f})
        counts.append(len(p.build_mesh()[3]))
    return counts[1] - counts[0]


def page(only: str | None = None) -> str:
    """Format the live vocabulary cheat-sheet with measured triangle counts."""
    from .scene import Model
    from .sculpt import QuadCage

    rows = [
        "jig vocabulary — hard-surface CAD + SubD QuadCage",
        "Order: what, how big (across, up, out), sides -> finish (.chamfer/.corner) -> placement (.on/.move/.turn) -> material at p.put.",
    ]
    for word, (cost, ex) in WORDS.items():
        if only and only.lower() not in word.lower():
            continue
        if word == "QuadCage":
            sig = "(verts, quads, ...)"
            doc = "Creased Catmull-Clark control cage: .ellipsoid, .box_grid, .loft_grid, .uv_patch, .sculpt, .tag, .keep, .drop, .crease, .subdivide, .thicken, .mirror_x."
            rows += ["", f"QuadCage{sig}", f"    {doc}", f"    triangles: {cost};  {ex}: {_cost(ex)}"]
            continue
        owner = {"Solid": Solid, "Outline": Outline, "p": Model}.get(word.split(".")[0]) if "." in word else None
        fn = getattr(owner, word.split(".")[1]) if owner else globals()[word]
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
        doc = " ".join((fn.__doc__ or "").split())
        rows += [
            "",
            f"{word.split('.')[-1] if '.' in word else word}{sig}" + (f"   on {word.split('.')[0]}" if "." in word else ""),
            f"    {doc}",
            f"    triangles: {cost};  {ex}: {_cost(ex)}",
        ]
    return "\n".join(rows)
