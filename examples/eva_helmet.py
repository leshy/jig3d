"""Aerospace EVA helmet — hybrid QuadCage SubD sculpt + hard-surface CAD example.

Combines organic creased Catmull-Clark `QuadCage` sculpting (aerodynamic shell, exact-fit curved
visor, ribbed neck ring) with hard-surface CSG (`extrude`, `lathe`, `bolt`, `p.inset`) for the
side-mounted comms pod and headlamp. Rests with collar base at Y = 0, front +Z.
"""

from jig import *  # noqa: F403

VARIANTS = {"eva_helmet": {}}
LAID = True
EYE = 0.25
MATERIALS = {
    "shell_white": Material("shell_white", (236, 240, 244), metallic=0.12, roughness=0.35),
    "shell_grey": Material("shell_grey", (195, 203, 212), metallic=0.18, roughness=0.42),
    "stripe_blue": Material("stripe_blue", (46, 154, 214), metallic=0.10, roughness=0.40),
}


def model(p):
    # Lift everything so collar bottom sits at y = 0
    y_off = 0.088

    # 1. Organic sculpted aerospace helmet shell via creased Catmull-Clark QuadCage
    base = (
        QuadCage.ellipsoid(0.160, 0.160, 0.172, center=(0.0, 0.090 + y_off, 0.002), nu=12, nv=8)
        .sculpt(
            lambda pt: (
                pt[0] * (0.95 if pt[1] < 0.02 + y_off else 1.0),
                pt[1] - (0.005 if abs(pt[0]) < 0.045 and pt[1] > 0.21 + y_off else 0.0),
                pt[2]
                - (0.022 if pt[2] < -0.07 else 0.0) * max(0.0, (pt[1] - (0.02 + y_off)) / 0.16)
                + (0.016 if pt[2] > 0.07 and pt[1] < 0.03 + y_off else 0.0),
            )
        )
        .tag(lambda c: c[2] > 0.075 and (0.005 + y_off) < c[1] < (0.165 + y_off) and abs(c[0]) < 0.132, "visor")
        .tag(lambda c: c[1] < (-0.042 + y_off), "neck")
        .crease("visor")
        .crease("neck")
        .subdivide(1)
    )

    shell = base.drop({"visor", "neck"}).thicken(
        0.014,
        inner_material="trim",
        rim_material="shell_grey",
        bevel=0.004,
    )
    p.sculpt(
        shell,
        lambda c, _t: "stripe_blue"
        if (0.085 + y_off < c[1] < 0.148 + y_off and c[2] < 0.08)
        else ("shell_grey" if c[1] < 0.015 + y_off else "shell_white"),
    )

    # 2. Exact-fit curved glass visor extracted from the same subdivided cage
    visor = base.keep("visor").move(0, 0, 0.005)
    p.sculpt(visor, "visor")

    # 3. Lofted neck sealing collar
    rows = [
        [
            [math.sin(2.0 * math.pi * i / 12) * r, y + y_off, -0.010 + math.cos(2.0 * math.pi * i / 12) * (r + 0.004)]
            for i in range(12)
        ]
        for y, r in ((-0.088, 0.094), (-0.068, 0.108), (-0.046, 0.120), (-0.030, 0.124))
    ]
    collar = QuadCage.loft_grid(rows, closed_u=True).thicken(0.008, inner_material="trim", bevel=0.002)
    p.sculpt(collar, lambda c, _t: "trim" if c[1] < -0.066 + y_off else "shell_grey")

    # 4. Hard-surface CAD side comms pod + forward LED headlamp mounted on right flank (+X)
    with p.on(frame(at=(0.155, 0.085 + y_off, -0.015), z="+x")):
        pod = p.put(extrude(rect(0.090, 0.072, cut=0.014), 0.028, draft=0.006).chamfer(0.004), "shell_white")
        p.put(bolt(0.008, 0.003).on(pod.face("out").corners(inset=0.014)), "steel")
        p.inset(pod.face("out"), rect(0.044, 0.014).at(0, -0.012), 0.005, role="glow", mouth=0.002)
        lamp = p.put(lathe([(0, 0.016), (0.042, 0.014)], 8).on(frame(at=(-0.032, 0.016, 0.014), z="-x")), "trim")
        p.inset(lamp.face("out"), 0.005, 0.004, role="lamp")

    # Antenna mast on left flank (-X)
    p.put(lathe([(0, 0.008), (0.02, 0.008), (0.14, 0.003)], 6).on(frame(at=(-0.165, 0.11 + y_off, -0.04), z="+y")), "trim")
    p.slot("head_center", frame(at=(0, 0.09 + y_off, 0), z="+z"))
