"""Geometry quality metrics for 3D meshes (`soft`, `deep`, `plain`, bounds, triangle count).

- `soft`: share of edge turning (length * angle > 10°) achieved in chamfer/round steps (< 60°).
  A plain box is 0.0; a chamfered hero model sits around 0.55–0.75.
- `deep`: depth / height (`z / y`).
- `plain`: largest unobstructed flat side/top face area relative to front bounding box (`w * h`).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .glb import _DTYPE, _WIDTH, load_gltf_doc, node_matrix

HERO_SOFT = (0.55, 0.75)


def triangles_from_file(path: str | os.PathLike) -> np.ndarray:
    """Load world-space triangles `(N, 3, 3)` float64 from a `.glb` or `.gltf` file."""
    doc, bufs = load_gltf_doc(path)

    def read(i: int) -> np.ndarray:
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        dt, w = np.dtype(_DTYPE[a["componentType"]]), _WIDTH[a["type"]]
        step = v.get("byteStride") or dt.itemsize * w
        at = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        raw = np.frombuffer(bufs[v["buffer"]], np.uint8, step * (a["count"] - 1) + dt.itemsize * w, at)
        return np.lib.stride_tricks.as_strided(raw, (a["count"], w * dt.itemsize), (step, 1)).copy().view(dt).reshape(-1, w)

    out = []

    def visit(i: int, m: np.ndarray):
        n = doc["nodes"][i]
        m2 = m @ node_matrix(n)
        for p in doc["meshes"][n["mesh"]]["primitives"] if "mesh" in n else []:
            if p.get("mode", 4) != 4:
                continue
            pos = read(p["attributes"]["POSITION"]).astype(np.float64)
            idx = read(p["indices"]).reshape(-1) if "indices" in p else np.arange(len(pos))
            pos = pos @ m2[:3, :3].T + m2[:3, 3]
            out.append(pos[idx[: len(idx) // 3 * 3]].reshape(-1, 3, 3))
        for ch in n.get("children", []):
            visit(ch, m2)

    for root in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return np.concatenate(out) if out else np.zeros((0, 3, 3), dtype=np.float64)


def open_share(t: np.ndarray, normal: np.ndarray, face: list[int], area: np.ndarray, k: int = 200) -> float:
    """Fraction of flat face `face` unobstructed along its outward `normal` (ray-cast sample)."""
    rng = np.random.default_rng(0)
    pick = rng.choice(face, k, p=area[face] / area[face].sum())
    u, v = rng.random(k), rng.random(k)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tri = t[pick]
    p = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0]) + 1e-4 * normal
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    h = np.cross(normal, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-12
    e1, e2, h, det, t0 = e1[ok], e2[ok], h[ok], det[ok], t[ok, 0]
    s = p[:, None, :] - t0[None]
    a = (s * h[None]).sum(2) / det
    q = np.cross(s, e1[None])
    b = (q @ normal) / det
    dist = (q * e2[None]).sum(2) / det
    hit = (a >= 0) & (b >= 0) & (a + b <= 1) & (dist > 1e-4)
    return float(1.0 - hit.any(1).mean())


def measure(tris: np.ndarray) -> dict:
    """Compute `{soft, deep, plain, size, triangles}` for `(N, 3, 3)` triangles."""
    t = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    if not len(t):
        return {"soft": 0.0, "deep": 0.0, "plain": 0.0, "size": (0.0, 0.0, 0.0), "triangles": 0}
    lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
    w, h, d = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
    cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 1e-9
    t, cross, area = t[keep], cross[keep], area[keep]
    if not len(t):
        return {"soft": 0.0, "deep": d / max(h, 1e-9), "plain": 0.0, "size": (w, h, d), "triangles": 0}
    n = cross / (2 * area[:, None])
    _, vid = np.unique(np.round(t.reshape(-1, 3) / 5e-4).astype(np.int64), axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    e = np.sort(np.stack([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]], 1), axis=2).reshape(-1, 2)
    owner = np.repeat(np.arange(len(t)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = (e[1:] == e[:-1]).all(1)
    first = np.nonzero(same & ~np.r_[same[1:], False] & ~np.r_[False, same[:-1]])[0]
    a, b = owner[first], owner[first + 1]
    turn = np.degrees(np.arccos(np.clip((n[a] * n[b]).sum(1), -1, 1)))
    at = np.zeros((int(vid.max()) + 1, 3))
    at[vid.reshape(-1)] = t.reshape(-1, 3)
    length = np.linalg.norm(at[e[first, 0]] - at[e[first, 1]], axis=1)
    work = length * turn * (turn > 10)
    soft = float(work[turn < 60].sum() / work.sum()) if work.sum() else 0.0

    shown = (n[:, 2] > -0.7) & (n[:, 1] > -0.7) & (n[:, 2] < 0.7)
    plane_q = np.round(np.c_[n * 50, (n * t[:, 0]).sum(1) / 2e-3]).astype(np.int64)
    flat = (turn < 1) & (plane_q[a] == plane_q[b]).all(1)
    parent = list(range(len(t)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(a[flat], b[flat]):
        parent[root(int(i))] = root(int(j))
    faces: dict[int, list[int]] = {}
    for i in np.nonzero(shown)[0]:
        faces.setdefault(root(int(i)), []).append(int(i))
    big = sorted(faces.values(), key=lambda f: -area[f].sum())[:8]
    plain = max((area[f].sum() * open_share(t, n[f[0]], f, area) for f in big), default=0.0) / max(w * h, 1e-9)
    return {
        "soft": soft,
        "deep": d / max(h, 1e-9),
        "plain": float(plain),
        "size": (w, h, d),
        "triangles": int(len(tris)),
    }


def format_line(nums: dict) -> str:
    """Format one summary metrics line with hero soft target range."""
    return (
        f"soft {nums['soft']:.2f} (hero {HERO_SOFT[0]:.2f}-{HERO_SOFT[1]:.2f})  "
        f"deep {nums['deep']:.2f}  plain {nums['plain']:.2f}"
    )


def measure_file(path: str | os.PathLike) -> dict:
    return measure(triangles_from_file(Path(path)))
