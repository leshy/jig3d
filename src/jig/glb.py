"""Deterministic binary glTF 2.0 (.glb) writer (multi-material PBR + slot nodes) and reader/merger."""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import struct

import numpy as np

from .materials import Palette

# 1x1 white PNG used as emissiveTexture so f3d and strict glTF viewers render emissiveFactor lit.
WHITE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8ffff3f0005fe02fe0def46b80000000049454e44ae426082"
)

_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _quat_from_z_to(normal) -> list[float]:
    """Unit quaternion `[x, y, z, w]` rotating `+Z` to unit vector `normal`."""
    n = np.asarray(normal, dtype=np.float64)
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    n = n / ln
    dot = float(n[2])
    if dot > 1.0 - 1e-7:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-7:
        return [0.0, 1.0, 0.0, 0.0]
    axis = np.cross(np.array([0.0, 0.0, 1.0]), n)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w], dtype=np.float64)
    q /= float(np.linalg.norm(q))
    return [round(float(v), 6) + 0.0 for v in q]


def pack_glb(
    pos: np.ndarray,
    nor: np.ndarray,
    mat_ids: np.ndarray,
    tri: np.ndarray,
    palette: Palette,
    *,
    name: str = "model",
    slots: list[tuple] | None = None,
    digest: str | None = None,
    vertex_colors: bool = False,
) -> bytes:
    """Serialize `(pos, nor, mat_ids, tri)` into deterministic binary glTF 2.0 (`.glb`) bytes.

    By default emits one primitive per material with full PBR `metallicRoughness`, emissive
    (with 1x1 white `emissiveTexture` so f3d renders glow lit), and alpha blend for glass,
    plus `slot:<name>` empty nodes for every declared `p.slot(...)`.
    With `vertex_colors=True`, emits a single primitive carrying sRGB `COLOR_0` attributes.
    """
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
    nor = np.asarray(nor, dtype=np.float32).reshape(-1, 3)
    mat_ids = np.asarray(mat_ids, dtype=np.uint16).reshape(-1)
    tri = np.asarray(tri, dtype=np.uint32).reshape(-1, 3)

    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    primitives: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []

    def push_bytes(raw: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(raw)
        v: dict = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target is not None:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_acc(arr: np.ndarray, acc_type: str, comp: int, target: int | None = None, norm: bool = False, bounds: bool = False) -> int:
        bv = push_bytes(arr.tobytes(), target=target)
        acc: dict = {"bufferView": bv, "componentType": comp, "count": int(len(arr)), "type": acc_type}
        if norm:
            acc["normalized"] = True
        if bounds:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    def clean_f32(arr: np.ndarray) -> np.ndarray:
        out = np.round(arr.astype(np.float64), 6)
        out[np.abs(out) < 5e-7] = 0.0
        return out.astype("<f4")

    if vertex_colors:
        rgb_tbl = palette.rgb_table()
        col = np.concatenate([rgb_tbl[mat_ids], np.full((len(mat_ids), 1), 255, dtype=np.uint8)], axis=1)
        p_c, n_c = clean_f32(pos), clean_f32(nor)
        rows = np.concatenate([p_c.view(np.uint8), n_c.view(np.uint8), col], axis=1)
        _, first, inv = np.unique(rows, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        inv_map = np.empty_like(order)
        inv_map[order] = np.arange(len(order))
        first = first[order]
        p_u, n_u, c_u = p_c[first], n_c[first], col[first]
        idx_u = inv_map[inv.reshape(-1)[tri]].astype("<u4").reshape(-1)
        primitives.append(
            {
                "attributes": {
                    "POSITION": add_acc(p_u, "VEC3", 5126, target=34962, bounds=True),
                    "NORMAL": add_acc(n_u, "VEC3", 5126, target=34962),
                    "COLOR_0": add_acc(c_u, "VEC4", 5121, target=34962, norm=True),
                },
                "indices": add_acc(idx_u, "SCALAR", 5125, target=34963),
            }
        )
    else:
        tri_mats = mat_ids[tri[:, 0]]
        used_ids = sorted(int(x) for x in np.unique(tri_mats))
        white_tex_idx: int | None = None
        if any(palette.material_of(mid).emissive for mid in used_ids):
            wbv = push_bytes(WHITE_PNG)
            images.append({"bufferView": wbv, "mimeType": "image/png"})
            textures.append({"source": 0})
            white_tex_idx = 0

        for mid in used_ids:
            mat = palette.material_of(mid)
            sub_tri = tri[tri_mats == mid]
            if not len(sub_tri):
                continue
            used_v, back = np.unique(sub_tri, return_inverse=True)
            p_sub = clean_f32(pos[used_v])
            n_sub = clean_f32(nor[used_v])
            rows = np.concatenate([p_sub.view(np.uint8), n_sub.view(np.uint8)], axis=1)
            _, first, inv = np.unique(rows, axis=0, return_index=True, return_inverse=True)
            order = np.argsort(first)
            inv_map = np.empty_like(order)
            inv_map[order] = np.arange(len(order))
            first = first[order]
            p_u, n_u = p_sub[first], n_sub[first]
            idx_u = inv_map[inv[back]].astype("<u4").reshape(-1)
            attrs: dict[str, int] = {
                "POSITION": add_acc(p_u, "VEC3", 5126, target=34962, bounds=True),
                "NORMAL": add_acc(n_u, "VEC3", 5126, target=34962),
            }
            if mat.emissive and white_tex_idx is not None:
                uv = np.full((len(p_u), 2), 0.5, dtype="<f4")
                attrs["TEXCOORD_0"] = add_acc(uv, "VEC2", 5126, target=34962)
            mat_idx = len(materials)
            materials.append(mat.gltf_dict(emissive_texture_index=white_tex_idx if mat.emissive else None))
            primitives.append(
                {
                    "attributes": attrs,
                    "indices": add_acc(idx_u, "SCALAR", 5125, target=34963),
                    "material": mat_idx,
                }
            )

    nodes: list[dict] = [{"name": name, "mesh": 0}]
    scene_nodes = [0]
    if slots:
        children = []
        for s_name, at, normal, sz in slots:
            node_idx = len(nodes)
            nodes.append(
                {
                    "name": f"slot:{s_name}",
                    "translation": [round(float(c), 6) + 0.0 for c in at],
                    "rotation": _quat_from_z_to(normal),
                    "extras": {
                        "slot": s_name,
                        "normal": [round(float(c), 6) + 0.0 for c in normal],
                        "size": [round(float(c), 6) + 0.0 for c in sz],
                    },
                }
            )
            children.append(node_idx)
        nodes[0]["children"] = children

    while len(blob) % 4:
        blob.append(0)

    asset: dict = {"version": "2.0", "generator": "jig"}
    if digest:
        asset["extras"] = {"source": digest}

    doc: dict = {
        "asset": asset,
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": [{"name": name, "primitives": primitives}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if materials:
        doc["materials"] = materials
    if images:
        doc["images"] = images
    if textures:
        doc["textures"] = textures

    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(js), b"JSON")
        + js
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + bytes(blob)
    )


def write_glb(path: str | os.PathLike, data: bytes) -> int:
    """Atomically write `.glb` bytes to `path` and return byte length."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, p)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return len(data)


def read_glb_parts(data: bytes) -> tuple[dict, bytes]:
    """Unpack a binary `.glb` into `(json_dict, bin_bytes)`."""
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary (.glb)")
    js, bin_ = None, b""
    at = 12
    while at < total:
        size, kind = struct.unpack_from("<II", data, at)
        chunk = data[at + 8 : at + 8 + size]
        if kind == 0x4E4F534A:
            js = json.loads(chunk)
        elif kind == 0x004E4942:
            bin_ = chunk
        at += 8 + size
    if js is None:
        raise ValueError("glTF binary missing JSON chunk")
    return js, bin_


def load_gltf_doc(path: str | os.PathLike) -> tuple[dict, list[bytes]]:
    """Load a `.glb` or `.gltf` file into `(doc, buffers_list)`."""
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix.lower() == ".gltf":
        doc = json.loads(raw)
        bufs = []
        for b in doc.get("buffers", []):
            uri = b.get("uri", "")
            if uri.startswith("data:"):
                bufs.append(base64.b64decode(uri.split(",", 1)[1]))
            else:
                bufs.append((p.parent / uri).read_bytes())
        return doc, bufs
    doc, bin_ = read_glb_parts(raw)
    return doc, [bin_]


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = node.get("scale", [1.0, 1.0, 1.0])
    t = node.get("translation", [0.0, 0.0, 0.0])
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r * np.asarray(s, dtype=np.float64)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def bounds_of_gltf(doc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space axis-aligned bounding box `(lo, hi)` from glTF accessor bounds."""
    lo = np.full(3, math.inf)
    hi = np.full(3, -math.inf)

    def visit(i: int, m: np.ndarray):
        n = doc["nodes"][i]
        m2 = m @ node_matrix(n)
        at = np.eye(4) if "skin" in n else m2
        for prim in doc["meshes"][n["mesh"]]["primitives"] if "mesh" in n else []:
            a = doc["accessors"][prim["attributes"]["POSITION"]]
            amn, amx = a["min"], a["max"]
            for c in range(8):
                pt = np.array([amx[j] if (c >> j) & 1 else amn[j] for j in range(3)], dtype=np.float64)
                q = at[:3, :3] @ pt + at[:3, 3]
                lo[:] = np.minimum(lo, q)
                hi[:] = np.maximum(hi, q)
        for ch in n.get("children", []):
            visit(ch, m2)

    for root in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return lo, hi


def extract_slots(doc: dict) -> list[tuple[str, list[float], list[float], list[float]]]:
    """Read `slot:<name>` nodes from a glTF document."""
    out = []
    for n in doc.get("nodes", []):
        name = n.get("name", "")
        if name.startswith("slot:"):
            ex = n.get("extras", {})
            out.append(
                (
                    ex.get("slot", name[5:]),
                    n.get("translation", [0.0, 0.0, 0.0]),
                    ex.get("normal", [0.0, 0.0, 1.0]),
                    ex.get("size", [0.0, 0.0]),
                )
            )
    return out


def merge_glbs(items: list[tuple[dict, list[bytes], tuple[float, float, float], float]]) -> bytes:
    """Merge multiple `(doc, bufs, translation, yaw_deg)` scenes into one binary `.glb`."""
    keys = ("bufferViews", "accessors", "images", "samplers", "textures", "materials", "meshes", "nodes", "skins")
    out: dict = {"asset": {"version": "2.0", "generator": "jig"}, **{k: [] for k in keys}, "scenes": [{"nodes": []}], "scene": 0}
    blob = bytearray()

    for g, bufs, move, yaw_deg in items:
        base = {k: len(out[k]) for k in keys}
        sh = lambda k, v: v + base[k]  # noqa: E731
        buf_offsets = []
        for b in bufs:
            while len(blob) % 4:
                blob.append(0)
            buf_offsets.append(len(blob))
            blob.extend(b)
        for v in g.get("bufferViews", []):
            nv = dict(v)
            nv["byteOffset"] = nv.get("byteOffset", 0) + buf_offsets[nv["buffer"]]
            nv["buffer"] = 0
            out["bufferViews"].append(nv)
        for a in g.get("accessors", []):
            na = dict(a)
            if "bufferView" in na:
                na["bufferView"] = sh("bufferViews", na["bufferView"])
            na.pop("sparse", None)
            out["accessors"].append(na)
        for im in g.get("images", []):
            nim = dict(im)
            if "bufferView" in nim:
                nim["bufferView"] = sh("bufferViews", nim["bufferView"])
            out["images"].append(nim)
        for t in g.get("textures", []):
            nt = {}
            for k, v in t.items():
                if k == "sampler":
                    nt[k] = sh("samplers", v)
                elif k == "source":
                    nt[k] = sh("images", v)
                elif k != "extensions":
                    nt[k] = v
            out["textures"].append(nt)
        for m in g.get("materials", []):
            nm = json.loads(json.dumps(m))
            for holder in (nm, nm.get("pbrMetallicRoughness", {})):
                for k, v in list(holder.items()):
                    if k.endswith("Texture") and isinstance(v, dict):
                        v["index"] = sh("textures", v["index"])
                        v.pop("extensions", None)
            nm.pop("extensions", None)
            out["materials"].append(nm)
        for mesh in g.get("meshes", []):
            prims = []
            for p in mesh["primitives"]:
                q: dict = {"attributes": {k: sh("accessors", v) for k, v in p["attributes"].items()}}
                if "indices" in p:
                    q["indices"] = sh("accessors", p["indices"])
                if "material" in p:
                    q["material"] = sh("materials", p["material"])
                if "mode" in p:
                    q["mode"] = p["mode"]
                prims.append(q)
            out["meshes"].append({"primitives": prims})
        for n in g.get("nodes", []):
            nn = {k: v for k, v in n.items() if k not in ("camera", "extensions", "weights")}
            if "mesh" in nn:
                nn["mesh"] = sh("meshes", nn["mesh"])
            if "skin" in nn:
                nn["skin"] = sh("skins", nn["skin"])
            if "children" in nn:
                nn["children"] = [sh("nodes", c) for c in nn["children"]]
            out["nodes"].append(nn)
        for s in g.get("skins", []):
            ns = {**s, "joints": [sh("nodes", j) for j in s["joints"]]}
            if "inverseBindMatrices" in ns:
                ns["inverseBindMatrices"] = sh("accessors", ns["inverseBindMatrices"])
            ns.pop("skeleton", None)
            out["skins"].append(ns)
        out["samplers"] += g.get("samplers", [])
        roots = [sh("nodes", i) for i in g["scenes"][g.get("scene", 0)]["nodes"]]
        rad = math.radians(yaw_deg)
        wrapper: dict = {"translation": [float(c) for c in move], "children": roots}
        if abs(yaw_deg) > 1e-6:
            wrapper["rotation"] = [0.0, math.sin(rad / 2), 0.0, math.cos(rad / 2)]
        out["nodes"].append(wrapper)
        out["scenes"][0]["nodes"].append(len(out["nodes"]) - 1)

    while len(blob) % 4:
        blob.append(0)
    out["buffers"] = [{"byteLength": len(blob)}]
    doc = {k: v for k, v in out.items() if v != []}
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(js), b"JSON")
        + js
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + bytes(blob)
    )
