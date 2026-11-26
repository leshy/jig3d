"""Half-space convex solid primitives (pure NumPy, voxel-free)."""

from __future__ import annotations

import itertools
import math

import numpy as np

from . import _plane

BIG = _plane.BIG
AXES = {"x": (0, 1, 2), "y": (1, 0, 2), "z": (2, 0, 1)}


def plane(n, d, strict=True, smooth=None):
    """Create a normalized half-space plane `(unit_normal, offset, strict, smooth)`."""
    n = np.asarray(n, float)
    k = float(np.linalg.norm(n))
    return (n / k, float(d) / k, strict, smooth)


def _unit(a, s=1.0):
    e = np.zeros(3)
    e[a] = s
    return e


def _span(axis, a0, a1):
    a = AXES[axis][0]
    return [plane(_unit(a, -1), -a0, False), plane(_unit(a), a1)]


class Shape:
    """A solid represented as a union of convex half-space polyhedra (`parts`)."""

    def __init__(self, parts):
        self.parts = parts

    __array_ufunc__ = None

    def __and__(self, o):
        if not isinstance(o, Shape):
            return NotImplemented
        return Shape([a + b for a in self.parts for b in o.parts])

    def __or__(self, o):
        if not isinstance(o, Shape):
            return NotImplemented
        return Shape(self.parts + o.parts)


def box_shape(x0, x1, y0, y1, z0, z1) -> Shape:
    """Axis-aligned box `x0..x1, y0..y1, z0..z1` as a Shape."""
    lo, hi = (x0, y0, z0), (x1, y1, z1)
    planes = []
    for a in range(3):
        planes += [plane(_unit(a, -1), -lo[a], False), plane(_unit(a), hi[a])]
    return Shape([planes])


def sym(axis, h, c=0.0) -> Shape:
    """Slab `|axis - c| < h` centred on `c`."""
    a = "xyz".index(axis)
    return Shape([[plane(_unit(a), c + h), plane(_unit(a, -1), h - c)]])


def slab(axis, a0, a1) -> Shape:
    """Slab `a0 <= axis < a1`."""
    a = "xyz".index(axis)
    return Shape([[plane(_unit(a, -1), -a0, False), plane(_unit(a), a1)]])


def half(axis, sign, d) -> Shape:
    """Half-space `sign * axis < d`."""
    return Shape([[plane(_unit("xyz".index(axis), sign), d)]])


def _poly_planes(axis, pts, smooth=None):
    _, ua, va = AXES[axis]
    pts = [np.asarray(p, float) for p in pts]
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
    out = []
    for i, (p, q) in enumerate(zip(pts, pts[1:] + pts[:1])):
        e = q - p
        if not e.any():
            continue
        nu, nv = (e[1], -e[0]) if area > 0 else (-e[1], e[0])
        n = np.zeros(3)
        n[ua], n[va] = nu, nv
        out.append(plane(n, nu * p[0] + nv * p[1], True, smooth[i] if smooth else None))
    return out


def prism(axis, pts, a0, a1) -> Shape:
    """Convex polygon `pts` across `axis` from `a0` to `a1`."""
    return Shape([_span(axis, a0, a1) + _poly_planes(axis, pts)])


def between(axis, a0, a1, us, lo, hi, u=None) -> Shape:
    """Extruded band along `axis` from `a0..a1` between polylines `(us, lo)` and `(us, hi)`."""
    lo = [float(lo(np.float64(x))) for x in us] if callable(lo) else lo
    hi = [float(hi(np.float64(x))) for x in us] if callable(hi) else hi
    _, ua, va = AXES[axis]
    if u is not None and "xyz".index(u) != ua:
        ua, va = va, ua
    parts = []
    for i in range(len(us) - 1):
        u0, u1 = us[i], us[i + 1]
        planes = []
        for (v0, v1), sign in (((lo[i], lo[i + 1]), -1), ((hi[i], hi[i + 1]), 1)):
            n = np.zeros(3)
            n[ua], n[va] = -(v1 - v0) * sign, (u1 - u0) * sign
            planes.append(plane(n, n[ua] * u0 + n[va] * v0, sign > 0))
        planes += [plane(_unit(ua, -1), -u0, False), plane(_unit(ua), u1)]
        parts.append(_span(axis, a0, a1) + planes)
    return Shape(parts)


def cyl(axis, cu, cv, r, a0, a1, n=None, smooth=True) -> Shape:
    """Cylinder along `axis` centred at `(cu, cv)` with radius `r` and `n` sides."""
    n = n or (12 if r < 0.05 else 24)
    _, ua, va = AXES[axis]
    k = r * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    ang = (np.arange(n) + 0.5) * 2 * np.pi / n
    pts = [(cu + k * np.cos(t), cv + k * np.sin(t)) for t in ang]
    c3, ax = np.zeros(3), _unit(AXES[axis][0])
    c3[ua], c3[va] = cu, cv
    sm = [(c3, ax)] * n if smooth else None
    return Shape([_span(axis, a0, a1) + _poly_planes(axis, pts, sm)])


def annulus(axis, cu, cv, r0, r1, a0, a1, n=32) -> Shape:
    """Hollow ring of radii `r0..r1` around `(cu, cv)` along `axis` from `a0..a1`."""
    k = 1 / np.cos(np.pi / n)
    parts = []
    for i in range(n):
        t0, t1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        pts = [(cu + r * np.cos(t), cv + r * np.sin(t)) for r, t in ((r0, t0), (r1 * k, t0), (r1 * k, t1), (r0, t1))]
        parts.append(_span(axis, a0, a1) + _poly_planes(axis, pts))
    return Shape(parts)


def _strips(outline):
    pts = [tuple(map(float, p)) for p in outline]
    edges = [(a, b) for a, b in zip(pts, pts[1:] + pts[:1]) if a[1] != b[1]]
    out = []
    vs = sorted({p[1] for p in pts})
    for v0, v1 in zip(vs, vs[1:]):
        vm = (v0 + v1) / 2
        at = lambda a, b, v: a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1])  # noqa: E731
        cross = sorted((at(a, b, vm), (at(a, b, v0), at(a, b, v1))) for a, b in edges if (a[1] > vm) != (b[1] > vm))
        for (_, left), (_, right) in zip(cross[::2], cross[1::2]):
            out.append((v0, v1, left, right))
    return out


def revolved(axis, centre, outline, n=24) -> Shape:
    """Profile `[(radius, t), ...]` revolved around `axis` (0/1/2 or 'x'/'y'/'z') through `centre`."""
    if isinstance(axis, str):
        axis = "xyz".index(axis)
    c = np.asarray(centre, float)
    ax = _unit(axis)
    e0, e1 = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
    parts = []
    for t0, t1, (l0, l1), (r0, r1) in _strips(outline):
        l0, l1 = max(l0, 0.0), max(l1, 0.0)
        for k in range(n):
            a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
            d0, d1 = np.cos(a0) * e0 + np.sin(a0) * e1, np.cos(a1) * e0 + np.sin(a1) * e1
            planes = [plane(-ax, -(ax @ c) - t0, False), plane(ax, ax @ c + t1)]
            planes += [
                plane(-np.cross(ax, d0), -(np.cross(ax, d0) @ c), False),
                plane(np.cross(ax, d1), np.cross(ax, d1) @ c),
            ]
            for (q0, q1), outer in (((r0, r1), True), ((l0, l1), False)):
                if not outer and q0 <= 1e-9 and q1 <= 1e-9:
                    continue
                p00, p10 = c + q0 * d0 + t0 * ax, c + q0 * d1 + t0 * ax
                p01, p11 = c + q1 * d0 + t1 * ax, c + q1 * d1 + t1 * ax
                nn = np.cross(p10 - p00, p01 - p00)
                if np.linalg.norm(nn) < 1e-12:
                    nn = np.cross(p01 - p00, p11 - p00)
                mid = (d0 + d1) / 2
                if (nn @ mid > 0) != outer:
                    nn = -nn
                if np.linalg.norm(nn) < 1e-12:
                    continue
                smooth = (c, ax) if abs(q0 - q1) < 1e-9 else None
                planes.append(plane(nn, nn @ p00, True, smooth))
            parts.append(planes)
    return Shape(parts)


_DIRS = [np.array(d, float) / np.linalg.norm(d) for d in np.ndindex(3, 3, 3) for d in [np.array(d) - 1] if d.any()]


def rounded_box(lo, hi, r, flat=()) -> Shape:
    """Box `lo..hi` with edges and corners rounded by `r` (45° facets), except sides in `flat` (`'y0'`, `'x1'`, ...)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    glo, ghi = lo.copy(), hi.copy()
    for f in flat:
        k = "xyz".index(f[0])
        (glo if f[1] == "0" else ghi)[k] += -r if f[1] == "0" else r
    mid, half_ = (glo + ghi) / 2, (ghi - glo) / 2
    planes = box_shape(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]).parts[0]
    for d in _DIRS:
        planes.append(plane(d, d @ mid + np.abs(d) @ (half_ - r) + r))
    return Shape([planes])


def rounded(axis, a0, a1, u0, u1, v0, v1, r, corners, n=6) -> Shape:
    """Bar along `axis` with listed corners (`'u0v1'`, `'u1v1'`, ...) rounded at radius `r` in `n` facets."""
    _, ua, va = AXES[axis]
    planes = _span(axis, a0, a1)
    planes += [
        plane(_unit(ua, -1), -u0, False),
        plane(_unit(ua), u1),
        plane(_unit(va, -1), -v0, False),
        plane(_unit(va), v1),
    ]
    for c in corners:
        du, dv = (1 if c[1] == "1" else -1), (1 if c[3] == "1" else -1)
        cu, cv = (u1 if du > 0 else u0) - du * r, (v1 if dv > 0 else v0) - dv * r
        c3 = np.zeros(3)
        c3[ua], c3[va] = cu, cv
        ta, tb = np.arctan2(0, du), np.arctan2(dv, 0)
        tb += 2 * np.pi * (tb - ta < -np.pi) - 2 * np.pi * (tb - ta > np.pi)
        t = np.linspace(ta, tb, n + 1)
        for s0, s1 in zip(t, t[1:]):
            m = (s0 + s1) / 2
            nn = np.zeros(3)
            nn[ua], nn[va] = np.cos(m), np.sin(m)
            planes.append(plane(nn, nn @ c3 + r * np.cos((s1 - s0) / 2), True, (c3, _unit(AXES[axis][0]))))
    return Shape([planes])


def _tube(a, b, r, n=12, smooth=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = (b - a) / np.linalg.norm(b - a)
    u = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    k = r * np.sqrt(np.pi / n / np.tan(np.pi / n))
    planes = [plane(-ax, -(ax @ a), False), plane(ax, ax @ b)]
    for t in (np.arange(n) + 0.5) * 2 * np.pi / n:
        nn = np.cos(t) * u + np.sin(t) * v
        planes.append(plane(nn, nn @ a + k, True, (a, ax) if smooth else None))
    return planes


def rod(a, b, r, n=8, smooth=False) -> Shape:
    """An n-sided bar/strut from point `a` to `b` of radius `r`."""
    return Shape([_tube(a, b, r, n=n, smooth=smooth)])


def frustum(bottom, top, y0, y1) -> Shape:
    """Tapered block from rectangle `bottom=(x0, x1, z0, z1)` at `y0` to `top` at `y1`."""
    planes = [plane((0, -1, 0), -y0, False), plane((0, 1, 0), y1)]
    h = y1 - y0
    for axis, k0, k1 in ((0, 0, 1), (2, 2, 3)):
        for sign, kb, kt in ((-1, bottom[k0], top[k0]), (1, bottom[k1], top[k1])):
            n = np.zeros(3)
            n[axis], n[1] = sign, -sign * (kt - kb) / h
            planes.append(plane(n, sign * kb - sign * (kt - kb) / h * y0))
    return Shape([planes])


def flat_poly(pts, t=0.002) -> Shape:
    """Thin convex planar plate thickened by `t` along its normal."""
    pts = [np.array(v, float) for v in pts]
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    n /= np.linalg.norm(n)
    cen = sum(pts) / len(pts)
    planes = [plane(n, n @ cen + t / 2), plane(-n, -(n @ cen) + t / 2)]
    for p, q in zip(pts, pts[1:] + pts[:1]):
        e = np.cross(q - p, n)
        if e @ (cen - p) > 0:
            e = -e
        planes.append(plane(e, e @ p))
    return Shape([planes])


_HULLS: dict = {}


def hull_parts(pts):
    """Convex hull of 3D points as one convex part (list of half-space planes)."""
    key = tuple(tuple(float(c) for c in p) for p in pts)
    if key not in _HULLS:
        _HULLS[key] = _hull([np.asarray(p, float) for p in key])
    return [[(n.copy(), d, st, sm) for n, d, st, sm in _HULLS[key]]]


def _hull(pts):
    cen = sum(pts) / len(pts)
    tri = np.array(list(itertools.combinations(range(len(pts)), 3)), int).reshape(-1, 3)
    P = np.array(pts)
    a = P[tri[:, 0]]
    n = np.cross(P[tri[:, 1]] - a, P[tri[:, 2]] - a)
    size = np.sqrt((n * n).sum(1))
    over = P @ n.T - (a * n).sum(1)
    over = np.where(((cen - a) * n).sum(1) > 0, -over, over)
    t = np.flatnonzero((size >= 1e-12) & (over <= 4e-9 * size).all(0))
    n = n[t] / size[t, None]
    n = np.where((((cen - a[t]) * n).sum(1) > 0)[:, None], -n, n)
    d = (n * a[t]).sum(1)
    return [plane(n[x], d[x]) for x in np.flatnonzero((P @ n.T <= d + 1e-9).all(0))]


def moved(parts, at, half_turn=False):
    """Move convex parts by `at`, optionally turning 180° around Y first."""
    out = []
    at_v = np.asarray(at, float)
    for part in parts:
        q = []
        for n, d, st, sm in part:
            n_arr = np.array([-n[0], n[1], -n[2]], float) if half_turn else np.asarray(n, float)
            q.append((n_arr, float(d + n_arr @ at_v), st, sm))
        out.append(q)
    return out


def bevelled(x0, x1, y0, y1, z0, z1, b, edges) -> Shape:
    """Box with listed edges (`'+z-x'`, `'+y+x'`, ...) cut at 45° by `b`."""
    planes = list(box_shape(x0, x1, y0, y1, z0, z1).parts[0])
    lo, hi = (x0, y0, z0), (x1, y1, z1)
    for e in edges:
        n, corner = np.zeros(3), np.zeros(3)
        for sign, axis in ((e[0], e[1]), (e[2], e[3])):
            a = "xyz".index(axis)
            n[a] = 1.0 if sign == "+" else -1.0
            corner[a] = hi[a] if sign == "+" else lo[a]
        n /= math.sqrt(2)
        planes.append(plane(n, n @ corner - b / math.sqrt(2)))
    return Shape([planes])
