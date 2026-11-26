"""Mesh containers: welded CSG (`Mesh`) and laid shell meshing (`Laid`) with per-line cost and warnings.

In `Laid` mode (the default for `jig`), each drawn put is meshed as its own shell against only the
cuts stated after it (plus any later put of another material lying flush over one of its faces).
Buried triangles wholly inside another put are dropped. This saves 20–35% triangles vs full CSG
welding and gives exact per-line triangle cost plus flush/overhang/coplanar diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
import os

import numpy as np

from . import _plane
from .materials import Palette

EPS = 1e-6  # metres: on a plane
FLAT = 1e-5  # metres: two faces this close lie in one plane
AREA = 1e-7  # m^2: minimum overlap/overhang area to report
PROUD = 0.1  # metres padding around measured bounding box


class Mesh:
    """Welded half-space CSG mesh over `lo..hi`."""

    def __init__(self, lo, hi, palette: Palette | None = None):
        self.lo = np.asarray(lo, float) - PROUD
        self.hi = np.asarray(hi, float) + PROUD
        self.puts: list[tuple] = []
        self.lines: list[tuple[str, int]] = []
        self.palette: Palette = palette if palette is not None else Palette()
        self.cut_of: dict[int, frozenset[int]] = {}

    def put(self, parts, paint):
        self.puts.append((parts, paint))

    def digest(self) -> str:
        extra = [([], repr(self.palette.rgb_table().tolist()))]
        if self.cut_of:
            extra.append(([], repr(sorted((c, sorted(ts)) for c, ts in self.cut_of.items()))))
        return _plane.digest(self.puts + extra, self.lo, self.hi)

    def mesh(self):
        return _plane.mesh(self.puts, self.lo, self.hi, palette=self.palette, cut_of=self.cut_of or None)


def _part(planes, lo, hi):
    pl = [(tuple(float(x) for x in n), float(d), s, None) for n, d, s, *_ in planes]
    for a in range(3):
        e = [0.0, 0.0, 0.0]
        e[a] = 1.0
        pl += [((-e[0], -e[1], -e[2]), -float(lo[a]), False, None), (tuple(e), float(hi[a]), True, None)]
    pl = _plane._dedup(pl)
    faces = _plane._faces(pl)
    if not faces:
        return None
    pts = np.array([p for _, f in faces for p in f])
    N = np.array([n for n, *_ in planes], float).reshape(-1, 3)
    D = np.array([d for _, d, *_ in planes], float)
    area = np.zeros(len(N))
    for k, f in faces:
        area[(np.abs(N - pl[k][0]).sum(1) + np.abs(D - pl[k][1])) < 1e-9] += _plane._area(f)
    return N, D, (pts.min(0) - EPS, pts.max(0) + EPS), area, [(np.array(pl[k][0]), pl[k][1], f) for k, f in faces]


def _meets(a, b) -> bool:
    (a0, a1), (b0, b1) = a, b
    return bool(
        a0[0] <= b1[0] and a0[1] <= b1[1] and a0[2] <= b1[2] and b0[0] <= a1[0] and b0[1] <= a1[1] and b0[2] <= a1[2]
    )


def _over_poly(poly, under, n):
    edges = []
    n0, n1, n2 = [float(c) for c in n]
    under_pts = np.asarray(under, float).tolist()
    for k, a in enumerate(under_pts):
        b = under_pts[(k + 1) % len(under_pts)]
        d0, d1, d2 = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        e = (d1 * n2 - d2 * n1, d2 * n0 - d0 * n2, d0 * n1 - d1 * n0)
        edges.append((e, e[0] * a[0] + e[1] * a[1] + e[2] * a[2]))
    inner, _ = _plane._split([tuple(v) for v in poly], edges)
    return inner


def _over(poly, under, n) -> float:
    inner = _over_poly(poly, under, n)
    return _plane._area(inner) if inner else 0.0


def _split_faces(polys, n, d, part_list, sign=1.0):
    ins, outs = [], list(polys)
    n0, n1, n2 = [float(c) for c in n]
    for _, _, _, _, faces in part_list:
        for fn, fd, fpoly in faces:
            if abs(sign * fd - d) >= FLAT or np.abs(sign * fn - n).max() >= FLAT:
                continue
            under = fpoly if sign > 0 else fpoly[::-1]
            edges = []
            for k, a in enumerate(under):
                b = under[(k + 1) % len(under)]
                d0, d1, d2 = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                e = (d1 * n2 - d2 * n1, d2 * n0 - d0 * n2, d0 * n1 - d1 * n0)
                edges.append((e, e[0] * a[0] + e[1] * a[1] + e[2] * a[2]))
            next_outs = []
            for poly in outs:
                inside, outside = _plane._split(poly, edges)
                if inside is not None:
                    ins.append(inside)
                next_outs.extend(outside)
            outs = next_outs
            if not outs:
                return ins, []
    return ins, outs


class Laid(Mesh):
    """Laid-shell CSG mesher with hidden-face culling, per-line cost, and overlap/overhang warnings."""

    def digest(self) -> str:
        extra = [([], repr(self.palette.rgb_table().tolist())), ([], "laid")]
        if self.cut_of:
            extra.append(([], repr(sorted((c, sorted(ts)) for c, ts in self.cut_of.items()))))
        return _plane.digest(self.puts + extra, self.lo, self.hi)

    def plan(self):
        parts = []
        for pl, _ in self.puts:
            got = [q for q in (_part(x, self.lo, self.hi) for x in pl) if q is not None]
            parts.append((got, (np.min([q[2][0] for q in got], 0), np.max([q[2][1] for q in got], 0)) if got else None))
        cut_of = self.cut_of
        cuts = {}
        for i, (_, paint) in enumerate(self.puts):
            if paint is not None and parts[i][1] is not None:
                cuts[i] = [
                    c
                    for c in range(i + 1, len(self.puts))
                    if self.puts[c][1] is None
                    and parts[c][1] is not None
                    and (c not in cut_of or i in cut_of[c])
                    and _meets(parts[c][1], parts[i][1])
                ]
        return parts, cuts

    def _lay(self, i, removers, parts, later, memo=None):
        if memo is not None:
            key = (i, tuple(removers))
            if key not in memo:
                memo[key] = self._lay(i, removers, parts, later)
            return memo[key]
        mesh = _plane.mesh(
            [self.puts[i]] + [(self.puts[j][0], None) for j in removers],
            self.lo,
            self.hi,
            palette=self.palette,
        )
        tri, box = mesh[3], parts[i][1]
        keep = np.ones(len(tri), bool)
        t = mesh[0][tri].astype(float)
        pts = np.concatenate([t, t.mean(1, keepdims=True)], 1)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        tlo, thi = pts.min(1), pts.max(1)
        for j, (got, jbox) in enumerate(parts):
            if j == i or self.puts[j][1] is None or jbox is None or not _meets(jbox, box):
                continue
            buried = np.zeros(len(tri), bool)
            for N, D, *_ in got:
                s = pts @ N.T - D
                inside = (s <= EPS).all((1, 2))
                outer = ((np.abs(s) <= EPS).all(1) & (n @ N.T > 0)).any(1) if j < i else np.zeros(len(tri), bool)
                buried |= inside & ~outer
            for x in np.flatnonzero(buried & keep):
                poly = [tuple(v) for v in t[x]]
                if not any(
                    _meets(parts[c][1], (tlo[x], thi[x]))
                    and any(
                        _plane._area(_plane._split(poly, list(zip(map(tuple, N), D)))[0] or []) > AREA
                        for N, D, *_ in parts[c][0]
                    )
                    for c in later[j]
                ):
                    keep[x] = False
        return mesh, keep, t[keep], n[keep]

    def _flush(self, i, t, n, first, parts, cuts_by=None):
        k = np.linalg.norm(n, axis=1)
        ok = k > 1e-12
        t, n = t[ok], n[ok] / k[ok, None]
        d = np.einsum("ij,ij->i", n, t[:, 0])
        all_out: dict[int, float] = defaultdict(float)
        warn_out: dict[int, float] = defaultdict(float)
        cuts_i = (cuts_by or {}).get(i, ())
        for j, (u, m, e) in first.items():
            q = self.puts[j][1]
            if j <= i or q == self.puts[i][1] or not _meets(parts[j][1], parts[i][1]):
                continue
            near = (np.abs(d[:, None] - e[None]) < FLAT) & (np.abs(n[:, None] - m[None]).max(2) < FLAT)
            pairs = list(zip(*np.nonzero(near)))
            if not pairs:
                continue
            cj_parts = [p for c in (cuts_by or {}).get(j, ()) for p in parts[c][0]]
            ci_parts = [p for c in cuts_i if c > j for p in parts[c][0]]
            for x, y in pairs:
                inner = _over_poly(t[x], u[y], m[y])
                if not inner:
                    continue
                area = _plane._area(inner)
                if area <= 0:
                    continue
                all_out[j] += area
                polys = [inner]
                if cj_parts:
                    in_c, out_c = _split_faces(polys, m[y], e[y], cj_parts, sign=-1.0)
                    if in_c:
                        polys = out_c + _split_faces(in_c, m[y], e[y], parts[j][0], sign=1.0)[0]
                if polys and ci_parts:
                    in_c, out_c = _split_faces(polys, m[y], e[y], ci_parts, sign=-1.0)
                    if in_c:
                        polys = out_c + _split_faces(in_c, m[y], e[y], parts[i][0], sign=1.0)[0]
                if polys:
                    warn_out[j] += sum(_plane._area(p) for p in polys)
        return (
            {j: a for j, a in all_out.items() if a > AREA},
            {j: a for j, a in warn_out.items() if a > AREA},
        )

    def shells(self, without=(), planned=None, only=None, memo=None):
        parts, cuts = planned or self.plan()
        cut_of = self.cut_of
        later = {
            j: [
                c
                for c in range(j + 1, len(self.puts))
                if self.puts[c][1] is None
                and parts[c][1] is not None
                and (c not in cut_of or j in cut_of[c])
            ]
            for j in range(len(self.puts))
        }
        want = set(cuts) if only is None else set(only)
        need = {j for j in cuts if any(j >= i and _meets(parts[j][1], parts[i][1]) for i in want)}
        first, laid = {}, {}
        for j in sorted(need):
            mine = [c for c in cuts[j] if (j, c) not in without]
            laid[j] = (mine, *self._lay(j, mine, parts, later, memo))
            u, m = laid[j][3], laid[j][4]
            k = np.linalg.norm(m, axis=1)
            ok = k > 1e-12
            m = m[ok] / k[ok, None]
            first[j] = (u[ok], m, np.einsum("ij,ij->i", m, u[ok][:, 0]))
        cuts_by = {j: laid[j][0] for j in laid}
        out = []
        for i in sorted(want):
            mine, mesh, keep, t, n = laid[i]
            all_flush, warn_flush = self._flush(i, t, n, first, parts, cuts_by)
            all_flush = {j: a for j, a in all_flush.items() if (i, j) not in without}
            warn_flush = {j: a for j, a in warn_flush.items() if (i, j) not in without}
            if all_flush:
                mine = sorted(mine + list(all_flush))
                mesh, keep, _, _ = self._lay(i, mine, parts, later, memo)
            out.append((i, (*mesh[:3], mesh[3][keep]), mine, warn_flush))
        return out

    def mesh(self):
        pos, nor, pnt, tri, base = [], [], [], [], 0
        for _, (p, n, q, t), *_ in self.shells():
            if not len(t):
                continue
            used, back = np.unique(t, return_inverse=True)
            pos.append(p[used])
            nor.append(n[used])
            pnt.append(q[used])
            tri.append(back.reshape(-1, 3) + base)
            base += len(used)
        if not pos:
            return _plane.mesh([], self.lo, self.hi, palette=self.palette)
        return (
            np.concatenate(pos).astype(np.float32),
            np.concatenate(nor).astype(np.float32),
            np.concatenate(pnt).astype(np.uint16),
            np.concatenate(tri).astype(np.uint32),
        )

    def cost(self) -> dict[tuple[str, int], int]:
        """Return `{(file, line): triangles}` summing to total laid triangles."""
        planned, memo = self.plan(), {}
        parts, _ = planned
        later = {j: [] for j in range(len(self.puts))}
        out: dict[tuple[str, int], int] = defaultdict(int)
        for i, (pos, _, _, tri), rs, _ in self.shells(planned=planned, memo=memo):
            mine = [(N, D) for N, D, *_ in parts[i][0]]
            alone = self._lay(i, [], parts, later)
            have = {}
            for name, t in (("alone", alone[2]), ("laid", pos[tri].astype(float))):
                if len(t):
                    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
                    d = np.einsum("ij,ij->i", n, t[:, 0])
                else:
                    n = np.zeros((0, 3))
                    d = np.zeros((0,))
                have[name] = (t, n, d)
            own: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
            for side, (t, n, d) in enumerate((have["alone"], have["laid"])):
                for x in range(len(t)):
                    k = next(
                        (
                            (a, m)
                            for a, (N, D) in enumerate(mine)
                            for m in np.flatnonzero(
                                (np.abs(N - n[x]).max(1) < FLAT) & (np.abs(D - d[x]) < FLAT)
                            )
                        ),
                        None,
                    )
                    if k is not None:
                        own[k][side] += 1
                    elif side:
                        r = next(
                            (
                                r
                                for r in rs
                                for N, D, *_ in parts[r][0]
                                if (
                                    (np.abs(np.abs(N @ n[x]) - 1) < FLAT)
                                    & (np.abs(np.abs(D) - abs(d[x])) < FLAT)
                                    & (np.abs(N @ t[x].T - D[:, None]) < FLAT).all(1)
                                ).any()
                            ),
                            None,
                        )
                        out[self.lines[r if r is not None else i]] += 1
            for (a, m), (was, now) in own.items():
                N, D = mine[a]
                on = have["laid"][0][
                    (np.abs(have["laid"][1] - N[m]).max(1) < FLAT) & (np.abs(have["laid"][2] - D[m]) < FLAT)
                ]
                box = (on.reshape(-1, 3).min(0) - EPS, on.reshape(-1, 3).max(0) + EPS) if len(on) else None
                by = [r for r in rs if box is not None and _meets(parts[r][1], box)]
                extra = max(0, now - was) if by else 0
                out[self.lines[i]] += now - extra
                for k, r in enumerate(by):
                    out[self.lines[r]] += extra // len(by) + (k < extra % len(by))
        return dict(out)

    def warnings(self) -> list[tuple[tuple[str, int], str]]:
        """Detect overhanging seats, flush material overlaps, and coplanar flicker."""
        planned = self.plan()
        parts = planned[0]
        shells = self.shells(planned=planned)
        out = []
        tris = []
        for i, (pos, _, _, tri), *_ in shells:
            for t in pos[tri].astype(float):
                n = np.cross(t[1] - t[0], t[2] - t[0])
                k = np.linalg.norm(n)
                if k > 1e-12:
                    tris.append((i, self.puts[i][1], t, n / k, float(n @ t[0] / k)))

        for i, *_ in shells:
            alone = _plane.mesh([self.puts[i]], self.lo, self.hi, palette=self.palette)
            ts = alone[0][alone[3]].astype(float)
            if not len(ts):
                continue
            ns = np.cross(ts[:, 1] - ts[:, 0], ts[:, 2] - ts[:, 0])
            areas = np.linalg.norm(ns, axis=1) / 2
            ns /= np.maximum(2 * areas, 1e-12)[:, None]
            earlier = [
                (j, P)
                for j, (got, jbox) in enumerate(parts[:i])
                if self.puts[j][1] is not None and jbox is not None and _meets(jbox, parts[i][1])
                for P in got
            ]
            groups_i = []
            for idx_t in range(len(ts)):
                if areas[idx_t] <= 1e-12:
                    continue
                nt, dt = ns[idx_t], float(ns[idx_t] @ ts[idx_t, 0])
                for gn, gd, gidxs in groups_i:
                    if abs(dt - gd) <= FLAT and float(nt @ gn) > 1 - 1e-6:
                        gidxs.append(idx_t)
                        break
                else:
                    groups_i.append((nt, dt, [idx_t]))
            bare, far, host = 0.0, 0.0, None
            for gn, _, gidxs in groups_i:
                ft = ts[gidxs]
                mine = float(areas[gidxs].sum())
                fbox = (ft.reshape(-1, 3).min(0) - EPS, ft.reshape(-1, 3).max(0) + EPS)
                seat, best_ov = None, AREA
                for j, (N, D, pbox, face, pfaces) in earlier:
                    if not _meets(pbox, fbox):
                        continue
                    back = np.flatnonzero((np.abs(ft[0] @ N.T - D) <= FLAT).all(0) & (N @ gn < -1 + 1e-6))
                    if not len(back):
                        continue
                    m = int(back[0])
                    if face[m] + AREA < mine:
                        continue
                    ov = 0.0
                    for pfn, pfd, fpoly in pfaces:
                        if abs(pfd - D[m]) < FLAT and np.abs(pfn - N[m]).max() < FLAT:
                            under = fpoly[::-1]
                            for t in ft:
                                ov += _over(t, under, gn)
                    if ov > best_ov:
                        best_ov, seat = ov, (j, N, D, m)
                if seat is None:
                    continue
                j, N, D, m = seat
                for t in ft:
                    rest = [[tuple(v) for v in t]]
                    for _, (eN, eD, *_) in earlier:
                        rest = [o for r in rest for o in _plane._split(r, list(zip(map(tuple, eN), eD + FLAT)))[1]]
                    left = sum(_plane._area(r) for r in rest)
                    if left > AREA:
                        bare += left
                        far = max(far, max(float(N[k] @ v - D[k]) for r in rest for v in r for k in range(len(N)) if k != m))
                        host = j
            if host is not None and far > 1e-3 and bare > 1e-5:
                out.append(
                    (
                        self.lines[i],
                        f"stands up to {far * 1000:.0f} mm past the face it sits on (line {self.lines[host][1]}, "
                        f"{bare * 1e4:.1f} cm2 of its seat bare)",
                    )
                )

        for i, _, _, flush in shells:
            for j, a in sorted(flush.items()):
                out.append(
                    (
                        self.lines[j],
                        f"lies flush over line {self.lines[i][1]} ({self.puts[j][1]} over {self.puts[i][1]}, "
                        f"{a * 1e4:.1f} cm2): that face is re-cut round it; lift or sink it",
                    )
                )

        groups = defaultdict(list)
        for i, q, t, n, d in tris:
            groups[(tuple(np.round(n, 4)), round(d, 5))].append((i, q, t, n))
        seen = defaultdict(float)
        for g in groups.values():
            if len({q for _, q, _, _ in g}) < 2:
                continue
            for x, (i, q, t, n) in enumerate(g):
                for j, r, u, _ in g[x + 1 :]:
                    if i == j or q == r:
                        continue
                    edges = [
                        (
                            tuple(np.cross(u[(e + 1) % 3] - u[e], n)),
                            float(np.cross(u[(e + 1) % 3] - u[e], n) @ u[e]),
                        )
                        for e in range(3)
                    ]
                    inner, _ = _plane._split([tuple(v) for v in t], edges)
                    if inner:
                        seen[(min(i, j), max(i, j))] += _plane._area(inner)
        for (i, j), a in sorted(seen.items()):
            if a > AREA:
                out.append(
                    (
                        self.lines[j],
                        f"lies coplanar over line {self.lines[i][1]} ({self.puts[j][1]} over {self.puts[i][1]}, "
                        f"{a * 1e4:.1f} cm2): they flicker",
                    )
                )
        dedup: dict[tuple[tuple[str, int], str], int] = defaultdict(int)
        for w in out:
            dedup[w] += 1
        return [(ln, text + (f" (x{k})" if k > 1 else "")) for (ln, text), k in dedup.items()]

    def report(self, top=6) -> list[str]:
        pos, _, pnt, tri = self.mesh()
        roles: dict[str, int] = defaultdict(int)
        for q in pnt[tri[:, 0]]:
            roles[self.palette.name_of(int(q))] += 1
        cost = sorted(self.cost().items(), key=lambda kv: -kv[1])
        src: dict[str, list[str]] = {}
        for (f, _), _ in cost:
            if f not in src:
                src[f] = open(f).read().split("\n") if os.path.isfile(f) else []
        out = [
            f"triangles {len(tri)} (laid)",
            "  by material  " + "  ".join(f"{k} {v}" for k, v in sorted(roles.items(), key=lambda kv: -kv[1])),
        ]
        for (f, ln), v in cost[:top]:
            line_txt = src[f][ln - 1].strip()[:80] if 0 < ln <= len(src[f]) else ""
            out.append(f"  {v:5d}  :{ln} {line_txt}")
        if len(cost) > top:
            out.append(f"  {sum(v for _, v in cost[top:]):5d}  {len(cost) - top} more lines")
        out += [f"warning :{ln} {text}" for (_, ln), text in self.warnings()]
        return out
