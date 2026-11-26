"""Tests for PBR glTF 2.0 export, slot nodes, themes, custom materials, and form metrics."""

import numpy as np

from jig import Material, Model, Palette, bevelled, block, measure, measure_file, pack_glb, write_glb
from jig import glb as glb_mod


def test_pbr_glb_materials_emissive_glass_and_slots(tmp_path):
    custom = Material("anodized_teal", (24, 148, 162), metallic=0.8, roughness=0.25)
    p = Model(laid=True)
    body_part = p.put(block(0.3, 0.2, 0.1).chamfer(0.01), custom)
    f = body_part.face("front")
    well = p.inset(f, 0.03, 0.015, role="glass", mouth=0.004)
    p.mark(well, [((0.0, 0.0), (0.05, 0.0), (0.05, 0.02), (0.0, 0.02))], "glow")
    p.put(block(0.04, 0.04, 0.01).on(f.at(-0.08, 0.04)), "#d4552a")
    p.slot("display", well)

    pos, nor, pnt, tri = p.build_mesh()
    data1 = pack_glb(pos, nor, pnt, tri, p.palette, name="test_unit", slots=p.named, digest=p.digest())
    data2 = pack_glb(pos, nor, pnt, tri, p.palette, name="test_unit", slots=p.named, digest=p.digest())
    assert data1 == data2, "GLB export must be deterministic"

    out_glb = tmp_path / "test_unit.glb"
    write_glb(out_glb, data1)
    doc, _ = glb_mod.read_glb_parts(out_glb.read_bytes())

    mat_names = {m["name"]: m for m in doc["materials"]}
    assert {"anodized_teal", "glass", "glow", "color_d4552a"} <= set(mat_names)
    assert mat_names["glass"]["alphaMode"] == "BLEND"
    assert mat_names["glass"]["doubleSided"] is True
    assert "emissiveTexture" in mat_names["glow"]
    assert len(doc["images"]) == 1 and len(doc["textures"]) == 1

    slots = glb_mod.extract_slots(doc)
    assert len(slots) == 1 and slots[0][0] == "display"
    assert np.allclose(slots[0][2], [0.0, 0.0, 1.0], atol=1e-4)

    # Vertex-color mode
    vc_data = pack_glb(pos, nor, pnt, tri, p.palette, name="vc", vertex_colors=True)
    vc_doc, _ = glb_mod.read_glb_parts(vc_data)
    assert len(vc_doc["meshes"][0]["primitives"]) == 1
    assert "COLOR_0" in vc_doc["meshes"][0]["primitives"][0]["attributes"]


def test_palette_overrides_and_metrics(tmp_path):
    p_default = Palette()
    p_custom = Palette(overrides={"body": "#d0d4d8", "custom_tint": "#336699"})
    assert p_default.material_of(p_default["body"]).rgb != p_custom.material_of(p_custom["body"]).rgb
    assert p_custom.material_of(p_custom["custom_tint"]).rgb == (0x33, 0x66, 0x99)

    # Plain box vs fully bevelled box metrics
    def measure_box(b):
        m = Model(laid=False)
        edges = [s + a + t + c for a, c in ("xy", "xz", "yz") for s in "+-" for t in "+-"]
        m.put(bevelled(0, 1, 0, 1, 0, 0.25, b, edges if b else []), "body")
        pos, nor, pnt, tri = m.build_mesh()
        out = tmp_path / f"box_{b}.glb"
        write_glb(out, pack_glb(pos, nor, pnt, tri, m.palette, name="box"))
        return measure_file(out), measure(pos[tri])

    plain_f, plain_m = measure_box(0.0)
    assert abs(plain_f["deep"] - 0.25) < 1e-6 and plain_f["soft"] == 0.0
    assert abs(plain_f["plain"] - 0.25) < 1e-6
    assert abs(plain_f["soft"] - plain_m["soft"]) < 1e-6

    cut_f, _ = measure_box(0.05)
    assert cut_f["soft"] > 0.95
    assert abs(cut_f["plain"] - 0.9 * 0.15) < 1e-5
