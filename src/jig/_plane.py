"""Exact half-space CSG polygon clipper, convex face merger, and surface mesher.

Each put is a list of convex parts, each part a list of half-space planes `(n, d, strict, smooth)`
with a material spec (or `None` for a cut). What fills a point is the last put containing it.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .materials import Palette

EPS = 1e-7
TINY = 1e-10
BIG = 100.0
MESHER_VERSION = 1

# Counters of the last `mesh()` call for diagnostics/testing.
WORK = {"pairs": 0, "splits": 0, "unions": 0}


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _clip(poly, n, d):
    """The part of a convex polygon (list of 3D points) where n.p <= d."""
    out = []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        dp, dq = _dot(p, n) - d, _dot(q, n) - d
        if dp <= EPS:
            out.append(p)
        if (dp < -EPS and dq > EPS) or (dp > EPS and dq < -EPS):
            t = dp / (dp - dq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t))
    return out


def _area(poly):
    if len(poly) < 3:
        return 0.0
    x = y = z = 0.0
    p0 = poly[0]
    for i in range(1, len(poly) - 1):
        c = _cross(_sub(poly[i], p0), _sub(poly[i + 1], p0))
        x, y, z = x + c[0], y + c[1], z + c[2]
    return (x * x + y * y + z * z) ** 0.5 / 2


def _quad(n, d):
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _cross(n, a)
    k = _dot(u, u) ** 0.5
    u = (u[0] / k, u[1] / k, u[2] / k)
    v = _cross(n, u)
    c = (n[0] * d, n[1] * d, n[2] * d)
    return [
        tuple(c[i] + BIG * (su * u[i] + sv * v[i]) for i in range(3))
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def _same_plane(a, b, sign=1.0):
    return all(abs(a[0][i] - sign * b[0][i]) < 1e-7 for i in range(3)) and abs(a[1] - sign * b[1]) < 1e-7


def _dedup(planes):
    out = []
    for p in planes:
        if not any(all(abs(p[0][i] - q[0][i]) < 1e-9 for i in range(3)) and abs(p[1] - q[1]) < 1e-9 for q in out):
            out.append(p)
    return out


def _faces(planes):
    """A convex part's faces: `(plane_index, polygon)`; `[]` for an empty or flat part."""
    out = []
    for k, (n, d, *_) in enumerate(planes):
        poly = _quad(n, d)
        for j, (m, e, *_) in enumerate(planes):
            if j != k:
                poly = _clip(poly, m, e)
                if len(poly) < 3:
                    break
        if _area(poly) > TINY:
            out.append((k, poly))
    return out if len(out) >= 4 else []


def _split(poly, planes):
    """A convex polygon split by a convex region: `(inside_or_None, outside_list)`."""
    WORK["splits"] += 1
    left, rest = [], poly
    for n, d, *_ in planes:
        outside = _clip(rest, (-n[0], -n[1], -n[2]), -d)
        if _area(outside) > TINY:
            left.append(outside)
        rest = _clip(rest, n, d)
        if _area(rest) <= TINY:
            return None, left
    return rest, left


def _relation(plane, planes):
    """Relation of a face plane to another convex part's planes."""
    for i, q in enumerate(planes):
        if _same_plane(plane, q):
            return "same", planes[:i] + planes[i + 1 :]
        if _same_plane(plane, q, -1.0):
            return "back", planes[:i] + planes[i + 1 :]
    return "apart", planes


def _box(pts, pad=1e-6):
    return (
        tuple(min(p[i] for p in pts) - pad for i in range(3)),
        tuple(max(p[i] for p in pts) + pad for i in range(3)),
    )


def _overlap(a, b):
    WORK["pairs"] += 1
    return all(a[0][i] <= b[1][i] and b[0][i] <= a[1][i] for i in range(3))


def _near(p, q):
    return abs(p[0] - q[0]) < 1e-7 and abs(p[1] - q[1]) < 1e-7 and abs(p[2] - q[2]) < 1e-7


def _union(a, b, n):
    """Join convex polygons `a` and `b` (same winding about `n`) along a shared edge if still convex."""
    WORK["unions"] += 1
    for i in range(len(a)):
        p, q = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            if _near(b[j], q) and _near(b[(j + 1) % len(b)], p):
                poly = a[: i + 1] + [b[(j + 2 + k) % len(b)] for k in range(len(b) - 2)] + a[i + 1 :]
                out = []
                for k in range(len(poly)):
                    e0, e1 = _sub(poly[k], poly[k - 1]), _sub(poly[(k + 1) % len(poly)], poly[k])
                    t = _dot(_cross(e0, e1), n)
                    if t < -1e-12:
                        return None
                    if t > 1e-12 or _dot(e0, e1) < 0:
                        out.append(poly[k])
                return out if len(out) >= 3 else None
    return None


def _merge(frags, n):
    """Merge neighbouring coplanar fragments wherever their union remains convex."""
    if len(frags) <= 1:
        return [list(f) for f in frags]
    frags = [list(f) for f in frags]
    boxes = [_box(f, 1e-7) for f in frags]
    lo = np.array([b[0] for b in boxes]).reshape(-1, 3)
    hi = np.array([b[1] for b in boxes]).reshape(-1, 3)

    def touching(r, rows):
        idx = np.arange(len(frags))[rows]
        return idx[(lo[rows] <= hi[r]).all(1) & (lo[r] <= hi[rows]).all(1)]

    def join(a, b):
        nonlocal lo, hi
        frags[a] = _union(frags[a], frags[b], n)
        lo[a], hi[a] = _box(frags[a], 1e-7)
        del frags[b]
        lo, hi = np.delete(lo, b, 0), np.delete(hi, b, 0)

    r = 0
    while r < len(frags):
        c = next((int(j) for j in touching(r, slice(r + 1, None)) if _union(frags[r], frags[j], n) is not None), None)
        if c is None:
            r += 1
            continue
        join(r, c)
        while True:
            a = next((int(a) for a in touching(r, slice(0, r)) if _union(frags[a], frags[r], n) is not None), None)
            if a is None:
                break
            join(a, r)
            r = a
    return frags


def mesh(puts, lo, hi, palette: Palette | None = None, cut_of=None):
    """Mesh `puts` (`[(convex_parts, material_or_None)]`) inside box `lo..hi`.

    Returns `(positions (n, 3) f32, normals (n, 3) f32, mat_ids (n,) u16, triangles (m, 3) u32)`.
    """
    WORK.update(pairs=0, splits=0, unions=0)
    pal = palette if palette is not None else Palette()
    cut_of = cut_of or {}
    t = lambda v: tuple(float(x) for x in v)  # noqa: E731
    bound = []
    for a in range(3):
        e = [0.0, 0.0, 0.0]
        e[a] = 1.0
        bound += [((-e[0], -e[1], -e[2]), -float(lo[a]), False, None), (tuple(e), float(hi[a]), True, None)]

    parts = []  # (rank, mat_id_or_None, planes, faces, box, put_idx)
    for put_idx, (parts_of, paint) in enumerate(puts):
        if paint is None and put_idx in cut_of and not cut_of[put_idx]:
            continue
        mat_id = None if paint is None else (int(paint) if isinstance(paint, (int, np.integer)) else pal.id_of(paint))
        for planes in parts_of:
            planes = [
                (t(n), float(d), strict, smooth if smooth is None else (t(smooth[0]), t(smooth[1])))
                for n, d, strict, smooth in planes
            ]
            planes = _dedup(planes + bound)
            faces = _faces(planes)
            if faces:
                parts.append((len(parts), mat_id, planes, faces, _box([p for _, f in faces for p in f]), put_idx))

    pos, nor, pnt, tri = [], [], [], []
    los = np.array([o[4][0] for o in parts]).reshape(-1, 3)
    his = np.array([o[4][1] for o in parts]).reshape(-1, 3)

    for rank, paint, planes, faces, pbox, put_idx in parts:
        WORK["pairs"] += max(0, len(parts) - 1)
        touch = (los <= np.array(pbox[1])).all(1) & (np.array(pbox[0]) <= his).all(1)
        touch[rank] = False
        near_part = [parts[i] for i in np.flatnonzero(touch)]
        for k, poly in faces:
            n, d, _, smooth = planes[k]
            fbox = _box(poly)
            near = [o for o in near_part if _overlap(o[4], fbox)]
            if not cut_of:
                frags = [poly]
                for o in near:
                    if o[0] < rank:
                        continue
                    rel, clip = _relation(planes[k], o[2])
                    if rel != "back":
                        frags = [f for g in frags for f in _split(g, clip)[1]]
                groups: dict = {}
                for o in sorted(near, key=lambda o: -o[0]):
                    if not frags:
                        break
                    rel, clip = _relation(planes[k], o[2])
                    if rel == "same":
                        continue
                    mine = not (rel == "back" and o[0] > rank)
                    rest = []
                    for f in frags:
                        inside, outside = _split(f, clip)
                        if inside is not None and mine:
                            groups.setdefault(o[1], []).append(inside)
                        rest += outside
                    frags = rest
                groups.setdefault(None, []).extend(frags)
            else:
                my_targets = cut_of.get(put_idx) if paint is None else None
                live, kept = [(poly, frozenset())], []
                for o in sorted(near, key=lambda o: -o[0]):
                    if not live:
                        break
                    if o[0] < rank and my_targets is None:
                        continue
                    rel, clip = _relation(planes[k], o[2])
                    if rel == "back":
                        continue
                    nxt = []
                    if o[0] > rank:
                        if o[1] is not None:
                            for f, blocked in live:
                                if o[5] in blocked:
                                    nxt.append((f, blocked))
                                else:
                                    nxt.extend((g, blocked) for g in _split(f, clip)[1])
                        else:
                            o_targets = cut_of.get(o[5])
                            covers = (
                                o_targets is None
                                or (paint is not None and put_idx in o_targets)
                                or (my_targets is not None and my_targets <= o_targets)
                            )
                            for f, blocked in live:
                                inn, out = _split(f, clip)
                                nxt.extend((g, blocked) for g in out)
                                if inn is not None and not covers:
                                    nxt.append((inn, blocked | o_targets))
                    else:
                        if o[1] is not None:
                            for f, blocked in live:
                                if o[5] in my_targets or o[5] in blocked:
                                    nxt.append((f, blocked))
                                else:
                                    nxt.extend((g, blocked) for g in _split(f, clip)[1])
                        else:
                            o_targets = cut_of.get(o[5])
                            for f, blocked in live:
                                inn, out = _split(f, clip)
                                nxt.extend((g, blocked) for g in out)
                                if inn is not None:
                                    if o_targets is None:
                                        kept.append(inn)
                                    else:
                                        nxt.append((inn, blocked | o_targets))
                    live = nxt
                frags = kept + [f for f, _ in live]
                groups = {}
                live = [(f, frozenset()) for f in frags]
                for o in sorted(near, key=lambda o: -o[0]):
                    if not live:
                        break
                    rel, clip = _relation(planes[k], o[2])
                    if rel == "same":
                        continue
                    mine = not (rel == "back" and o[0] > rank)
                    nxt = []
                    if o[1] is not None:
                        draw = mine and (my_targets is None or (o[0] < rank and o[5] in my_targets))
                        for f, blocked in live:
                            if o[5] in blocked:
                                nxt.append((f, blocked))
                            else:
                                inn, out = _split(f, clip)
                                if inn is not None and draw:
                                    groups.setdefault(o[1], []).append(inn)
                                nxt.extend((g, blocked) for g in out)
                    else:
                        o_targets = cut_of.get(o[5])
                        for f, blocked in live:
                            inn, out = _split(f, clip)
                            nxt.extend((g, blocked) for g in out)
                            if inn is not None:
                                if o_targets is None:
                                    if mine:
                                        groups.setdefault(None, []).append(inn)
                                elif not (paint is not None and put_idx in o_targets and not mine):
                                    nxt.append((inn, blocked | o_targets))
                    live = nxt
                groups.setdefault(None, []).extend(f for f, _ in live)

            for other, fs in groups.items():
                if (paint is None) == (other is None):
                    continue
                face_paint, sign = (paint, 1.0) if paint is not None else (other, -1.0)
                for f in _merge(fs, n):
                    f = f if sign > 0 else f[::-1]
                    base = len(pos)
                    for p in f:
                        pos.append(p)
                        if smooth is None:
                            nor.append((sign * n[0], sign * n[1], sign * n[2]))
                        else:
                            c, ax = smooth
                            pc = _sub(p, c)
                            h = _dot(pc, ax)
                            r = (pc[0] - h * ax[0], pc[1] - h * ax[1], pc[2] - h * ax[2])
                            k2 = sign / max(_dot(r, r) ** 0.5, 1e-12)
                            nor.append((r[0] * k2, r[1] * k2, r[2] * k2))
                        pnt.append(face_paint)
                    tri += [(base, base + j, base + j + 1) for j in range(1, len(f) - 1)]

    return (
        np.array(pos, np.float32).reshape(-1, 3),
        np.array(nor, np.float32).reshape(-1, 3),
        np.array(pnt, np.uint16),
        np.array(tri, np.uint32).reshape(-1, 3),
    )


def digest(puts, lo, hi) -> str:
    """Short deterministic content hash of puts, bounds, and mesher version."""
    h = hashlib.sha1(f"jig-mesher {MESHER_VERSION}".encode())
    h.update(np.round(np.concatenate([lo, hi]), 9).tobytes())
    for parts, paint in puts:
        h.update(repr(paint).encode())
        for planes in parts:
            for n, d, strict, smooth in planes:
                h.update(np.round(np.append(n, d), 9).tobytes() + bytes([strict]))
                if smooth is not None:
                    h.update(np.round(np.concatenate(smooth), 9).tobytes())
    return h.hexdigest()[:16]
