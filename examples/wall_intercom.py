"""Wall-mounted intercom unit — hard-surface CAD example.

Demonstrates orthographic `body()`, `.chamfer()`, semantic face picking (`shell.faces("front", by="y")`),
zero-triangle `p.paint()`, `p.inset()`, `p.slots()`, `p.mark()`, `bolt().on(face.grid(...))`, `sweep()`,
and named attachment slots (`p.slot`). Front +Z, up +Y, back against wall Z = 0.
"""

from jig import *  # noqa: F403

VARIANTS = {"wall_intercom": {}}
LAID = True
W, H, D, CREASE = 0.46, 0.68, 0.14, 0.34


def model(p):
    p.budget(1500)

    # Wall mounting backplate with corner bolts
    plate = p.put(extrude(rect(W + 0.05, H + 0.12, cut=0.09), 0.02).chamfer(0.006).move(0, 0.44, 0), "body")
    p.put(bolt(0.026).on(plate.face("front").grid(2, 2, pitch=(0.28, 0.74))), "panel")

    # Raked housing intersected from front + side profiles, drafted & chamfered
    side = [(0, -H / 2), (D + 0.094, -H / 2), (D, CREASE - H / 2), (D, H / 2), (0, H / 2)]
    shell = p.put(body(rect(W, H, cut=0.07), side, draft=0.08).chamfer(0.015).on(plate.face("front")), "body2")

    # Two-tone front faces (zero extra triangles via p.paint)
    controls, face = shell.faces("front", by="y")
    p.paint([controls, face], "slate")
    p.cut(block(W - 0.12, 0.012, 0.008).into(face.at(0, -face.h / 2)), role="trim")
    p.put(
        bolt(0.018, 0.004).on([face.at(x, -face.h / 2 + 0.03) for x in (-0.15, 0.15)] + controls.grid(2, 1, pitch=0.31)),
        "trim",
    )

    # Recessed display bezel with glowing status bars + overhead brow hood + camera lens
    bezel = p.put(extrude(rect(0.31, 0.22, cut=0.04), 0.025).chamfer(0.008, "out").on(face.at(0, -0.01), sink=0.01), "trim")
    glass = p.inset(bezel.face("out"), 0.028, 0.025, role="screen", mouth=0.02)
    p.mark(glass, [rect(0.25, 0.008).at(0.025, 0.02), rect(0.14, 0.008).at(-0.03, -0.02)], "glow")

    hood = body(rect(0.27, 0.038), [(0, 0.019), (0.075, 0.019), (0.075, -0.001), (0, -0.019)])
    hood = p.put(hood.on(face.at(-0.02, 0.12)), "trim")
    cam = p.put(lathe([(0, 0.016), (0.02, 0.016)], 8).on(hood.face("front").at(-0.09, 0)), "trim")
    p.inset(cam.face("out"), 0.007, 0.004, role="glass")
    p.inset(face, rect(0.03, 0.05).at(0.145, 0.105), 0.006, role="glow", mouth=0.006)

    # Slanted control deck: speaker grille + tactile key bank + plate mark
    grille = p.inset(controls, rect(0.16, 0.20, cut=0.025).at(-0.05, 0.02), 0.012, role="trim", mouth=0.012)
    p.slots(grille, rect(0.11, 0.13), 5, depth=0.02, role="ink", mouth=0.002)
    strip = p.put(extrude(rect(0.08, 0.22, cut=0.015), 0.013).chamfer(0.004, "out").on(controls.at(0.1, 0.02)), "trim")
    for place, role in zip(strip.face("out").grid(1, 3, margin=0.02), ("panel", "panel", "accent")):
        p.put(block(0.05, 0.05, 0.02).chamfer(0.006, "out").on(place), role)
    p.mark(controls, rect(0.16, 0.03).at(-0.05, -0.125), "panel2")

    # Side vents, grab rail, bottom cable gland & swept conduit
    rake = shell.face("right")
    p.slots(rake, rect(0.06, 0.16).at(0, -0.15), 4, depth=0.03, role="ink")
    p.put(block(0.04, 0.03, 0.05).on(rake.at(0, 0.1).grid(1, 2, pitch=0.22)), "trim")
    p.put(block(0.03, 0.28, 0.025).chamfer(0.006, "upright").on(rake.at(0, 0.1), lift=0.05), "trim")
    p.put(lathe([(0, 0.025), (0.06, 0.025)], 8).on(shell.face("bottom").at(0.12, 0), sink=0.03), "trim")
    p.put(sweep(ngon(8, 0.018), [(0.12, 0.10, 0.06), (0.12, 0.04, 0.06), (0.12, 0.03, 0)]), "body")

    p.slot("screen", glass)
    p.slot("top", shell.face("top"))
