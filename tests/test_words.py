"""Unit tests for hard-surface CAD vocabulary, semantic faces, features, and laid-shell meshing."""

import warnings

import numpy as np
import pytest

from jig import (
    Model,
    Solid,
    around,
    block,
    body,
    bolt,
    box,
    extrude,
    floor,
    frame,
    lathe,
    loft,
    ngon,
    poly,
    rect,
    ring,
    stripes,
    sweep,
    tris,
    volume,
)
from jig import _lay, _places, _plane, words


def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_outlines_and_body_words():
    assert len(rect(0.2, 0.1).pts) == 4
    assert len(rect(0.2, 0.1, cut=0.02).pts) == 8
    assert len(rect(0.2, 0.1, cut=(0.02, 0)).pts) == 6
    assert _close(words._area(rect(0.2, 0.1, cut=0.02).pts), 0.02 - 4 * 0.0002)

    h = ngon(6, 0.05).pts
    assert len(h) == 6 and _close(max(v for _, v in h), 0.05)
    assert _close(max(u for u, _ in ngon(8, 0.05, "+x").pts), 0.05)

    L = poly([(0, 0), (0.3, 0), (0.3, 0.1), (0.1, 0.1), (0.1, 0.25), (0, 0.25)])
    assert len(L.pieces()) == 2
    assert _close(sum(words._area([L.pts[i] for i in p]) for p in L.pieces()), 0.3 * 0.1 + 0.1 * 0.15)
    assert _close(words._area(rect(0.2, 0.1).inset(0.01).pts), 0.18 * 0.08)
    assert _close(volume(extrude(rect(0.2, 0.1).band(0.01), 0.01)), (0.02 - 0.18 * 0.08) * 0.01)

    # Convex inset drops short corners cleanly
    small = rect(0.3, 0.2, cut=0.01).inset(0.03)
    assert len(small.pts) == 4 and _close(words._area(small.pts), 0.24 * 0.14, 1e-5)

    # Stripes
    st = stripes(rect(0.2, 0.1), 0.02)
    assert _close(sum(words._area(o.pts) for o in st), 0.01, 2e-4)

    # Extrude + draft + along + center
    o = rect(0.3, 0.2, cut=0.04)
    assert tris(extrude(o, 0.1)) == 28 and _close(volume(extrude(o, 0.1)), words._area(o.pts) * 0.1)
    d = extrude(o, 0.1, draft=0.02)
    assert tris(d) == 28
    assert tris(extrude(L, 0.05)) == 4 * 6 - 4

    ex_x = extrude(rect(0.2, 0.1), 0.4, along="x", center=True)
    assert _close(volume(ex_x), 0.2 * 0.1 * 0.4)
    assert np.allclose(ex_x.face("out").origin, (0.2, 0.0, 0.0))

    # Zero-area sliver join guard
    sliver_a = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    sliver_b = [(2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert _plane._union(sliver_a, sliver_b, (0.0, 0.0, 1.0)) is None
    assert len(_plane._merge([sliver_a, sliver_b], (0.0, 0.0, 1.0))) == 2


def test_body_loft_chamfer_corner_and_box():
    w = body(rect(0.2, 0.3).at(0, 0.15), [(0, 0), (0.1, 0), (0, 0.3)])
    assert tris(w) == 8 and _close(volume(w), 0.2 * 0.1 * 0.3 / 2)

    bt = body(rect(0.2, 0.3).at(0, 0.15), top=rect(0.2, 0.1, cut=(0.04, 0)).at(0, 0.05))
    assert _close(volume(bt), 0.3 * (0.02 - 2 * 0.0008))

    # Negative out span in body()
    span_straddle = body(rect(0.2, 0.3), [(-0.1, -0.15), (0.2, -0.15), (0.2, 0.15), (-0.1, 0.15)])
    assert _close(volume(span_straddle), 0.2 * 0.3 * 0.3)

    # Loft frustum
    lf = loft((0, rect(0.2, 0.2)), (0.1, rect(0.1, 0.1)), (0.2, rect(0.1, 0.1)))
    assert tris(lf) == 20 and _close(volume(lf), 0.1 / 3 * (0.04 + 0.01 + 0.02) + 0.001)

    # Chamfer & asymmetric chamfer & corner
    bx = extrude(rect(0.3, 0.2), 0.1)
    c = bx.chamfer(0.01)
    assert tris(c) == 12 + 8 * 4
    assert tris(bx.chamfer(0.01, "all")) == 12 + 12 * 4
    assert tris(bx.chamfer(0.01, "front&top")) == 12 + 4
    ab = bx.chamfer((0.02, 0.06), "front&bottom")
    assert _close(volume(ab), 0.006 - 0.3 * 0.02 * 0.06 / 2)
    assert _close(volume(bx.chamfer((0.06, 0.02), "bottom&front")), volume(ab))

    with pytest.raises(ValueError, match="front&top"):
        bx.chamfer(0.01, "front&back")

    cc = bx.corner(0.03, "front&top&right")
    assert tris(cc) == 12 + 4 and _close(volume(cc), 0.006 - 0.03**3 / 6)

    # box() returns Solid
    b0 = box(-0.15, 0.15, -0.1, 0.1, 0.0, 0.1)
    assert isinstance(b0, Solid) and tris(b0) == 12 and _close(volume(b0), 0.006)
    assert tris(b0.chamfer(0.01)) == 12 + 8 * 4


def test_round_words_and_sweep():
    cyl_ = lathe([(0, 0.05), (0.1, 0.05)], 8)
    assert tris(cyl_) == 2 * 8 + 2 * (8 - 2)
    assert _close(max(f.origin[1] for f in cyl_.faces()), 0.05)

    cone = lathe([(0, 0.05), (0.1, 0.05), (0.15, 0.0)], 8)
    assert tris(cone) == 8 * 2 + 8 + 6
    assert tris(bolt(0.02)) == 20

    sq = rect(0.02, 0.02)
    el = sweep(sq, [(0, 0, 0), (0.2, 0, 0), (0.2, 0.3, 0)])
    assert _close(volume(el), 0.0004 * 0.5, 1e-9)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dup = sweep(sq, [(0, 0, 0), (0.2, 0, 0), (0.2, 0, 0), (0.2, 0.3, 0)])
        assert tris(dup) == tris(el) and _close(volume(dup), volume(el), 1e-9)

    with pytest.raises(ValueError, match="no non-zero leg"):
        sweep(sq, [(0, 0, 0), (0, 0, 0)])

    r = ring(0.1, rect(0.01, 0.02), 12)
    assert tris(r) == 12 * 8
    assert _close(volume(r), 0.0002 * 2 * 12 * 0.1 * np.tan(np.pi / 12), 1e-9)


def test_places_islands_and_flat():
    b = block(0.4, 0.3, 0.1).chamfer(0.01)
    f = b.face("front")
    assert np.allclose(f.origin, (0, 0, 0.1)) and _close(f.w, 0.38) and _close(f.h, 0.28)
    assert np.allclose(b.face("top").y, (0, 0, -1))
    assert len(f.grid(3, 2, pitch=(0.1, 0.1))) == 6
    assert np.allclose(f.corners(0.02)[2].origin, (0.17, 0.12, 0.1))
    assert np.allclose(f.around(4, 0.1, start=90)[0].origin, (0, 0.1, 0.1))

    # Disconnected coplanar islands + strict flat filtering
    ibeam = extrude(
        poly([
            (0.0, -0.088), (0.076, -0.088), (0.076, -0.048), (0.058, -0.040),
            (0.058, 0.040), (0.076, 0.048), (0.076, 0.088), (0.0, 0.088),
        ]),
        1.0,
        along="x",
        center=True,
    )
    split = ibeam.faces("front", by="y", flat=True, islands=True)
    assert len(split) == 3
    flanges = [fc for fc in split if fc.origin[2] > 0.07]
    assert len(flanges) == 2 and np.allclose([fc.origin[1] for fc in flanges], [-0.068, 0.068], atol=1e-4)


def test_features_paint_inset_slots_pose_and_mirror():
    p = Model(laid=True)
    host = p.put(block(0.4, 0.3, 0.1), "body")
    front = host.face("front")
    p.paint(front, "slate")
    well = p.inset(front, rect(0.2, 0.12), 0.02, role="screen", mouth=0.005)
    p.slots(front.at(0, -0.09), rect(0.16, 0.05), 3, depth=0.015, role="ink", mouth=0.002)
    p.mark(well, rect(0.12, 0.01), "glow")
    p.slot("screen", well)

    with p.pose("lid", hinge=((0.2, 0.15, 0.0), (-0.2, 0.15, 0.0)), turn=(0, 90), at=1.0):
        p.put(block(0.4, 0.1, 0.02).on(host.face("top")), "panel")

    pos, nor, pnt, tri = p.build_mesh()
    assert len(tri) > 0
    used_roles = {p.palette.name_of(int(i)) for i in np.unique(pnt)}
    assert {"body", "slate", "screen", "ink", "glow", "panel"} <= used_roles
    assert len(p.named) == 1 and p.named[0][0] == "screen"
    assert p.warnings() == []


def test_laid_shells_culling_and_warnings():
    m = _lay.Laid((-1, -1, -1), (1, 1, 1))
    m.puts = [
        (box(-0.5, 0.5, -0.5, 0.5, 0, 0.1).parts, "body"),
        (box(-0.05, 0.05, -0.05, 0.05, 0.1, 0.12).parts, "panel"),
    ]
    m.lines = [(__file__, 1), (__file__, 2)]
    assert len(m.mesh()[3]) == 22  # 12 + 10 (seat dropped)
    assert len(_lay.Mesh.mesh(m)[3]) > 22
    assert m.warnings() == []
    assert m.cost() == {(__file__, 1): 12, (__file__, 2): 10}

    # Scoped cut_of cuts only targeted sibling
    m2 = _lay.Laid((-1, -1, -1), (1, 1, 1))
    m2.puts = [
        (box(-0.5, 0.5, -0.5, 0.5, 0, 0.1).parts, "body"),
        (box(-0.1, 0.1, -0.1, 0.1, 0.08, 0.125).parts, "trim"),
        (box(-0.05, 0.05, -0.05, 0.05, 0.085, 0.2).parts, None),
    ]
    m2.lines = [(__file__, 1), (__file__, 2), (__file__, 3)]
    m2.cut_of = {2: frozenset({1})}
    shells = m2.shells()
    assert shells[0][2] == [] and shells[1][2] == [2] and m2.warnings() == []
    assert len(shells[0][1][3]) == 12

    # Overhang warning
    m3 = _lay.Laid((-1, -1, -1), (1, 1, 1))
    m3.puts = [
        (box(-0.5, 0.5, -0.5, 0.5, 0, 0.1).parts, "body"),
        (box(0.45, 0.6, -0.05, 0.05, 0.1, 0.12).parts, "trim"),
    ]
    m3.lines = [(__file__, 1), (__file__, 4)]
    w = m3.warnings()
    assert len(w) == 1 and "100 mm past" in w[0][1]

    # Flush overlap warning
    m4 = _lay.Laid((-1, -1, -1), (1, 1, 1))
    m4.puts = [
        (box(-0.5, 0.5, -0.5, 0.5, 0, 0.1).parts, "body"),
        (box(-0.1, 0.1, -0.1, 0.1, 0.06, 0.1).parts, "glow"),
    ]
    m4.lines = [(__file__, 1), (__file__, 5)]
    w4 = m4.warnings()
    assert len(w4) == 1 and "flush" in w4[0][1]


def test_words_page():
    txt = words.page()
    assert txt.count("triangles:") == len(words.WORDS)
    assert "QuadCage" in txt
