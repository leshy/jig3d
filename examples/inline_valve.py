"""Parametric inline pipe valve — multi-variant CAD example.

Everything scales cleanly from pipe radius `R`. Demonstrates `with p.on(frame(...))`,
`with p.mirror("z")`, `lathe()`, `bolt().on(face.around(...))`, `ring()`, `sweep()`,
and paired slots (`("outlet", "inlet")`). Pipe runs along X; base rests on Y = 0.
"""

from jig import *  # noqa: F403

VARIANTS = {
    "inline_valve": dict(R=0.06),
    "inline_valve_r05": dict(R=0.05),
    "inline_valve_r08": dict(R=0.08),
}
LAID = True
HALF = 0.32


def model(p, R=0.06):
    H, F, B = R + 0.13, R + 0.045, R + 0.035

    with p.on(frame(at=(0, H, 0), z="+x")):
        p.put(lathe([(-HALF, R), (HALF, R)], 12), "copper")
        valve = p.put(lathe([(-0.11, B), (0.11, B)], 8).chamfer(0.012), "body2")
        p.put(lathe([(-0.045, B + 0.014), (0.045, B + 0.014)], 8).chamfer(0.01), "body2")
        with p.mirror("z"):
            flange = p.put(lathe([(HALF - 0.03, F), (HALF, F)], 12).chamfer(0.008), "body2")
            p.put(lathe([(HALF - 0.075, R + 0.008), (HALF - 0.03, R + 0.022)], 12), "body2")
            p.put(bolt(0.018, 0.006).on(flange.face("seat").around(6, R + 0.024, start=30)), "steel")
            p.put(lathe([(0.11, B - 0.008), (0.16, R + 0.012)], 8), "body2")
            p.put(lathe([(0.075, B + 0.01), (0.095, B + 0.01)], 8), "trim")
            p.slot(("outlet", "inlet"), flange.face("out"))

    # Floor mounting saddle + anchor bolts
    pad = p.put(block(0.14, 2 * B, 0.025).chamfer(0.008, "out").on(floor()), "body")
    p.put(block(0.10, 2 * B - 0.04, H - 0.025).chamfer(0.01, "upright").on(pad.face("out")), "body")
    p.put(bolt(0.016, 0.006).on(pad.face("out").corners(inset=0.02)), "steel")

    # Top actuator housing, position indicator window, and rear hydraulic line
    A = 0.035 + 0.4 * R
    act = p.put(extrude(rect(2 * A, 2 * A, cut=0.025), 0.08).on(valve.face("top"), sink=0.01), "slate")
    cap = p.put(block(2 * A + 0.01, 2 * A + 0.01, 0.015).chamfer(0.012, "out").on(act.face("out")), "body2")
    p.inset(act.face("front"), rect(0.07, 0.025).at(0, 0.005), 0.01, role="led_green", mouth=0.004)
    zl, y = -B - 0.03, act.face("back").origin[1]
    p.put(sweep(ngon(6, 0.011), [(0, y, -A), (0, y, zl), (0, 0.035, zl)]), "trim")
    p.put(block(0.05, 0.05, 0.035).chamfer(0.006).on(floor(0, zl)), "body2")

    # Valve stem + spoked handwheel
    stem = p.put(lathe([(0, 0.014), (0.06, 0.014)], 6).on(cap.face("out")), "steel")
    p.put(lathe([(0, 0.032), (0.015, 0.026)], 8).on(cap.face("out")), "trim")
    W = R + 0.045
    with p.on(stem.face("out")):
        p.put(ring(W, rect(0.016, 0.018), 12).move(0, 0, -0.02), "hazard")
        p.put(lathe([(-0.034, 0.022), (-0.006, 0.022)], 6), "trim")
        p.put(block(W - 0.03, 0.014, 0.014).move(W / 2, 0, -0.027).on(around(4, start=45)), "trim")
