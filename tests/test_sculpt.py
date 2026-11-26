"""Unit tests for QuadCage creased Catmull-Clark sculpting, thickening, projection, and hybrid scenes."""

import numpy as np

from jig import Model, QuadCage, block, extrude, frame, pack_glb, project_cage, rect


def test_quadcage_watertight_and_subdivide():
    ell = QuadCage.ellipsoid(0.14, 0.16, 0.18, nu=8, nv=6).subdivide(1)
    ok, vol = ell.is_watertight()
    assert ok and vol > 0.0

    bx = QuadCage.box_grid((0, 0, 0), (0.2, 0.3, 0.15), (2, 2, 2)).crease("all").subdivide(1)
    ok2, vol2 = bx.is_watertight()
    assert ok2 and abs(vol2 - 0.2 * 0.3 * 0.15) < 1e-6

    # pin_corners pins sharp >45° boundary corners
    sq = QuadCage.uv_patch([[[x, y, 0.0] for x in (0.0, 0.5, 1.0)] for y in (0.0, 0.5, 1.0)])
    pinned = sq.subdivide(1, pin_corners=True)
    rounded = sq.subdivide(1, pin_corners=False)
    for ci in (0, 2, 6, 8):
        assert np.allclose(pinned.verts[ci], sq.verts[ci])
        assert not np.allclose(rounded.verts[ci], sq.verts[ci])


def test_quadcage_thicken_bevel_tuck_and_mirror():
    dome = QuadCage.ellipsoid(0.15, 0.15, 0.15, nu=10, nv=6).drop(lambda c: c[1] < 0.0)
    shell = dome.thicken(0.012, inner_material="trim", rim_material="body2", bevel=0.003)
    ok, vol = shell.is_watertight()
    assert ok and vol > 0.0

    tucked = dome.thicken(0.0, tuck_in=0.010, bevel=0.002)
    assert len(tucked.quads) > len(dome.quads)

    # Mirror +X half across X=0
    patch = QuadCage.uv_patch([[[x, y, 0.1 - 0.5 * x * x] for x in (0.0, 0.05, 0.10)] for y in (0.0, 0.05, 0.10)])
    mirrored = patch.mirror_x()
    assert len(mirrored.quads) == 2 * len(patch.quads)
    assert np.isclose(mirrored.verts[:, 0].min(), -mirrored.verts[:, 0].max(), atol=1e-6)


def test_project_cage_onto_solid_and_quadcage():
    target_cage = QuadCage.ellipsoid(0.2, 0.2, 0.2, nu=12, nv=8).subdivide(1)
    patch = QuadCage.uv_patch([[[x, y, 0.28] for x in (-0.06, 0.0, 0.06)] for y in (-0.06, 0.0, 0.06)])
    snapped = project_cage(target_cage, patch, offset=0.005)
    radii = np.linalg.norm(snapped.verts, axis=1)
    assert np.all((radii > 0.18) & (radii < 0.22)), radii

    target_solid = block(0.4, 0.4, 0.2)
    snapped2 = project_cage(target_solid, patch, offset=0.002)
    assert np.allclose(snapped2.verts[:, 2], 0.202, atol=1e-4)


def test_hybrid_model_transforms_and_determinism():
    def build_once():
        p = Model(laid=True)
        base = p.put(extrude(rect(0.3, 0.3, cut=0.04), 0.06).chamfer(0.008), "body")
        cap = (
            QuadCage.ellipsoid(0.11, 0.08, 0.11, center=(0, 0, 0.06), nu=10, nv=6)
            .drop(lambda c: c[2] < 0.06)
            .subdivide(1)
            .thicken(0.008, inner_material="trim", bevel=0.002)
        )
        with p.on(base.face("out")):
            p.sculpt(cap, lambda c, _t: "accent" if c[2] > 0.11 else "panel")
        with p.mirror("x"):
            with p.on(frame(at=(0.14, 0.0, 0.03), z="+x")):
                lug = QuadCage.box_grid((0, 0, 0.01), (0.03, 0.03, 0.02), (1, 1, 1)).crease("all")
                p.put(lug, "steel")
        pos, nor, pnt, tri = p.build_mesh()
        return p, pack_glb(pos, nor, pnt, tri, p.palette, name="hybrid", digest=p.digest())

    p1, b1 = build_once()
    _, b2 = build_once()
    assert b1 == b2, "hybrid solid + QuadCage .glb must be bit-identical across runs"
    assert sum(p1.cost().values()) == len(p1.build_mesh()[3])
