"""Feature statements beside put and cut: `with p.on`, `with p.mirror`, `p.paint`, `p.inset`, `p.slots`, `p.mark`, `p.slot`."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np

from . import _places

LIFT = 0.001  # metres a recess cutter starts outside its face (never coplanar)
MARK = 0.0003  # metres a surface mark stands proud of its face
ON = 1e-5  # metres tolerance for matching triangles on a plane


class Part:
    """Returned by `p.put(...)`; exposes semantic face picking (`.face()`, `.faces()`)."""

    def __init__(self, solid, role):
        self.solid, self.role = solid, role
        solid.part = self

    def face(self, word, flat=False):
        return self.solid.face(word, flat=flat)

    def faces(self, word=None, by=None, flat=False, islands=False):
        return self.solid.faces(word, by, flat=flat, islands=islands)


def _cut_targets(of):
    if of is None:
        return None
    if isinstance(of, Part):
        items = (of,)
    elif isinstance(of, (str, bytes)):
        raise TypeError(f"cut(of=...) expects a Part or collection of Parts, got {type(of).__name__}")
    else:
        try:
            items = tuple(of)
        except TypeError as e:
            raise TypeError(f"cut(of=...) expects a Part or collection of Parts, got {type(of).__name__}") from e
    for item in items:
        if not isinstance(item, Part) or item.role is None:
            raise TypeError(f"cut(of=...) targets must be drawn Parts from p.put, got {item!r}")
    return items


class Features:
    """Mixin for Model/Description providing frames, mirroring, face painting, insets, slots, marks, and named slots."""

    def _features_init(self):
        self._frames: list[tuple[np.ndarray, np.ndarray]] = []
        self._mirrors: list[tuple[np.ndarray, np.ndarray]] = []
        self.painted: list[tuple] = []
        self.named: list[tuple] = []  # (name, centre, normal, size)
        self.drawn_parts: list[tuple] = []  # ((file, line), Part)

    def _copies(self, solid):
        from .words import Solid

        if isinstance(solid, list) and solid and all(isinstance(x, Solid) for x in solid):
            first = solid[0]
            for x in solid[1:]:
                first = first | x
            solid = first
        if not self._frames and not self._mirrors and not isinstance(solid, Solid):
            return [solid]
        s = solid if isinstance(solid, Solid) else Solid(solid.parts if hasattr(solid, "parts") else solid)
        if self._frames and not s.world:
            s = s._xf(*self._frames[-1], world=True)
        copies = [s]
        for R, t in self._mirrors:
            copies += [c._xf(R, t, world=True) for c in copies]
        return copies

    def _draw(self, kind, solid, paint, **options):
        from .words import Solid

        copies = self._copies(solid)
        first = copies[0] if isinstance(copies[0], Solid) else Solid(copies[0].parts if hasattr(copies[0], "parts") else copies[0])
        part = Part(first, paint)
        for c in copies:
            self._state(kind, c, paint, part=part, **options)
        self.drawn_parts.append((self.statements[-1][4], part))
        return part

    @contextmanager
    def on(self, place):
        """Place everything stated inside relative to `place` (a Face or Frame)."""
        (f,) = _places.flatten(place)
        R, t = f.R, f.origin
        if not f.world and self._frames:
            R0, t0 = self._frames[-1]
            R, t = R0 @ R, R0 @ t + t0
        self._frames.append((R, t))
        try:
            yield
        finally:
            self._frames.pop()

    @contextmanager
    def mirror(self, axis):
        """Draw everything stated inside twice: as stated and mirrored across the active frame's `axis` plane."""
        R, t = self._frames[-1] if self._frames else (np.eye(3), np.zeros(3))
        self._mirrors.append(_places.mirror_of(_places.Frame(t, R), axis))
        try:
            yield
        finally:
            self._mirrors.pop()

    def paint(self, faces, role):
        """Repaint specific faces in another material role (0 extra triangles)."""
        for f in _places.flatten(faces):
            host = f.part.role if getattr(f, "part", None) is not None else None
            copies = [(f.tris, f.n, f.d)]
            for R, t in self._mirrors:
                copies += [(tr @ R.T + t, R @ n, d + (R @ n) @ t) for tr, n, d in copies]
            for tr, n, d in copies:
                self.painted.append(("face", (tr, n, d, host), role))
            f.role = role

    def _mouthed(self, cutter, mouth, s0=None):
        from .words import Solid, _part

        w = cutter.R[:, 2]
        parts = list(cutter.parts)
        for q in cutter.parts:
            seat = next((d for n, d, *_ in q if np.dot(n, w) < -1 + 1e-6), None)
            top = -seat if s0 is None else s0
            planes = [(tuple(-w), -(top - mouth), False, None), (tuple(w), top + mouth, True, None)]
            for n, d, s, _ in q:
                if abs(np.dot(n, w)) < 1e-6:
                    k = np.asarray(n) + w
                    planes.append((tuple(k / np.linalg.norm(k)), (d + mouth + top) / np.linalg.norm(k), s, None))
            parts.append(_part(planes))
        return Solid(parts, cutter.R, cutter.world)

    def _cut(self, solid, role=None, mouth=0.0, s0=None, of=None):
        targets = _cut_targets(of)
        if mouth:
            solid = self._mouthed(solid, mouth, s0)
        opts = {"of": targets} if targets is not None else {}
        part = self._draw("put", solid, None, **opts)
        if role:
            host_roles = None if targets is None else frozenset(t.role for t in targets if t.role is not None)
            for c in self._copies(solid):
                if host_roles is None:
                    self.painted.append(("cut", c.parts, role))
                else:
                    self.painted.append(("cut_of", (c.parts, host_roles), role))
        return part

    def _place(self, place, fn):
        if not isinstance(place, _places.Frame):
            raise TypeError(f"{fn}: expected a Face or Frame, got {type(place).__name__}")
        if not place.world and self._frames:
            R0, t0 = self._frames[-1]
            if isinstance(place, _places.Face):
                n = R0 @ place.n
                return _places.Face(
                    n,
                    float(n @ (R0 @ place.origin + t0)),
                    place.tris @ R0.T + t0,
                    place.part,
                    True,
                    place.name,
                    place.role,
                    xy=(R0 @ place.x, R0 @ place.y),
                )
            return _places.Frame(R0 @ place.origin + t0, R0 @ place.R, True, place.role)
        return place

    def inset(self, face, spec, depth, role=None, mouth=0.0):
        """Recess into a face/frame (or raised panel if `depth < 0`); returns the floor Face."""
        from .words import _outline, extrude

        place = self._place(face, "inset")
        if isinstance(spec, (int, float)):
            if not isinstance(place, _places.Face):
                raise TypeError("inset: numeric margin requires a Face; pass an Outline (e.g. rect(w, h)) for a Frame")
            outline = place.outline().inset(spec)
        else:
            outline = _outline(spec)
        if depth < 0:
            paint = role or place.role
            if paint is None:
                raise TypeError("inset: raised panel (depth < 0) on a Frame without a host role requires role=")
            return self.put(extrude(outline, -depth).on(place), paint).face("out")
        n = place.n if isinstance(place, _places.Face) else place.z
        d = place.d if isinstance(place, _places.Face) else float(n @ place.origin)
        self._cut(extrude(outline, depth + LIFT).into(place, lift=LIFT), mouth=mouth, s0=-d)
        floor = self._floor(place, outline, depth)
        if role:
            self.paint(floor, role)
        return floor

    def _floor(self, face, outline, depth):
        n = face.n if isinstance(face, _places.Face) else face.z
        d = face.d if isinstance(face, _places.Face) else float(n @ face.origin)
        tris = []
        for piece in outline.pieces():
            p = [outline.pts[i] for i in piece]
            tris += [(p[0], p[k], p[k + 1]) for k in range(1, len(p) - 1)]
        world = np.array([[face.origin + u * face.x + v * face.y - depth * n for u, v in t] for t in tris])
        return _places.Face(n, d - depth, world, None, face.world, "floor", face.role, xy=(face.x, face.y))

    def slots(self, face, region, n, depth, role="trim", mouth=0.0, along="u", bar_role=None):
        """`n` vent/grille slots in `region`: one recess (floor `role`) + `n - 1` bars (`bar_role`)."""
        from .words import _outline, block

        place = self._place(face, "slots")
        bar_paint = bar_role if bar_role is not None else place.role
        if n > 1 and bar_paint is None:
            raise TypeError("slots: bare Frame has no host role for bars; pass bar_role=... or use a Face")
        r = _outline(region)
        us, vs = [u for u, _ in r.pts], [v for _, v in r.pts]
        w, h = max(us) - min(us), max(vs) - min(vs)
        floor = self.inset(place, r, depth, role=role, mouth=mouth)
        s = (h if along == "u" else w) / (2 * n - 1)
        steps = [(2 * k + 1.5) * s - (h if along == "u" else w) / 2 for k in range(n - 1)]
        base = floor if depth > 0 else place.at((min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2).lift(-depth)
        bars = [base.at(0, t) if along == "u" else base.at(t, 0) for t in steps]
        if bars:
            bar_d = abs(depth)
            bar = block(w, s, bar_d) if along == "u" else block(s, h, bar_d)
            self.put((bar.chamfer(mouth, "out") if mouth else bar).on(bars), bar_paint)
        return floor

    def mark(self, face, outlines, role):
        """Flat surface patches 0.3 mm proud of `face` (stripes, indicator strips, plates)."""
        from .words import _outline, extrude

        solid = None
        for o in outlines if isinstance(outlines, list) else [outlines]:
            s = extrude(_outline(o), MARK).on(face)
            solid = s if solid is None else solid | s
        return self._draw("put", solid, role, mark=True)

    def slot(self, name, place, size=None):
        """Record a named attachment slot (exported as a `slot:<name>` node in the `.glb`)."""
        names = list(name) if isinstance(name, (list, tuple)) else [name]
        (f,) = _places.flatten(place)
        size = size or (getattr(f, "w", 0.0), getattr(f, "h", 0.0))
        at, n = f.origin, f.z
        if not f.world and self._frames:
            R, t = self._frames[-1]
            at, n = R @ at + t, R @ n
        spots = [(at, n)]
        for R, t in self._mirrors:
            spots += [(R @ a + t, R @ m) for a, m in spots]
        for nm, (a, m) in zip(names, spots):
            self.named.append((nm, a, m, tuple(float(x) for x in size)))
            self._point(nm, a, tuple(float(x) for x in size), None)


def repaint(mesh_data, painted, palette):
    """Apply face and cut material repaints to `(pos, nor, mat_ids, tri)`."""
    if not painted:
        return mesh_data
    pos, nor, pnt, tri = mesh_data
    if not len(tri):
        return mesh_data
    t = pos[tri].astype(float)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    c = t.mean(1)
    role = pnt[tri[:, 0]].astype(int)
    for kind, what, to in painted:
        to_id = palette.id_of(to)
        if kind == "face":
            ftris, fn, fd, host = what
            on = (n @ fn > 1 - 1e-4) & (np.abs(c @ fn - fd) < ON)
            if host is not None:
                on &= role == palette.id_of(host)
            on[on] = _within(c[on], ftris, fn)
        elif kind == "cut_of":
            cparts, host_roles = what
            on = np.zeros(len(tri), bool)
            if host_roles:
                for q in cparts:
                    N, D = np.array([p[0] for p in q], float), np.array([p[1] for p in q], float)
                    inside = (c @ N.T <= D + ON).all(1)
                    for k in range(len(N)):
                        on |= inside & (n @ -N[k] > 1 - 1e-4) & (np.abs(c @ N[k] - D[k]) < ON)
                host_ids = [palette.id_of(r) for r in host_roles]
                on &= np.isin(role, host_ids)
        else:
            on = np.zeros(len(tri), bool)
            for q in what:
                N, D = np.array([p[0] for p in q], float), np.array([p[1] for p in q], float)
                inside = (c @ N.T <= D + ON).all(1)
                for k in range(len(N)):
                    on |= inside & (n @ -N[k] > 1 - 1e-4) & (np.abs(c @ N[k] - D[k]) < ON)
        role[on] = to_id
    changed = np.flatnonzero(role != pnt[tri[:, 0]])
    if not len(changed):
        return mesh_data
    extra = tri[changed].reshape(-1)
    base = len(pos)
    tri = tri.copy()
    tri[changed] = base + np.arange(3 * len(changed)).reshape(-1, 3)
    return (
        np.concatenate([pos, pos[extra]]),
        np.concatenate([nor, nor[extra]]),
        np.concatenate([pnt, np.repeat(role[changed], 3).astype(pnt.dtype)]),
        tri,
    )


def _within(pts, ftris, fn):
    x, y = _places.axes(fn)
    P = np.stack([pts @ x, pts @ y], 1)
    T = np.stack([ftris @ x, ftris @ y], 2)
    hit = np.zeros(len(P), bool)
    for a, b, cc in T:
        d = (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0])
        if abs(d) < 1e-15:
            continue
        l1 = ((b[0] - P[:, 0]) * (cc[1] - P[:, 1]) - (b[1] - P[:, 1]) * (cc[0] - P[:, 0])) / d
        l2 = ((cc[0] - P[:, 0]) * (a[1] - P[:, 1]) - (cc[1] - P[:, 1]) * (a[0] - P[:, 0])) / d
        hit |= (l1 >= -1e-6) & (l2 >= -1e-6) & (1 - l1 - l2 >= -1e-6)
    return hit


def painted(m, paints):
    """Wrap a Mesh/Laid so face/cut repaints apply after meshing and enter its digest."""
    if not paints:
        return m
    orig_mesh, orig_digest = m.mesh, m.digest
    h = hashlib.sha1()
    for kind, what, role in paints:
        h.update(f"{kind} {role}".encode())
        if kind == "face":
            h.update(np.round(what[0], 9).tobytes() + repr(what[3]).encode())
        elif kind == "cut_of":
            cparts, host_roles = what
            h.update(repr(sorted(str(r) for r in host_roles)).encode())
            for q in cparts:
                for n, d, *_ in q:
                    h.update(np.round(np.append(n, d), 9).tobytes())
        else:
            for q in what:
                for n, d, *_ in q:
                    h.update(np.round(np.append(n, d), 9).tobytes())
    key = h.hexdigest()
    m.mesh = lambda: repaint(orig_mesh(), paints, m.palette)
    m.digest = lambda: hashlib.sha1((orig_digest() + key).encode()).hexdigest()[:16]
    return m
