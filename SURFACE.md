# `jig` Surface Reference

A 10-minute reference to every modelling word in `jig`. Run `jig words [FILTER]` at any time to see live signatures and measured triangle counts, and `jig materials` for built-in PBR materials.

**Conventions:** metres, right-handed (`+X` right, `+Y` up, `+Z` front). Outlines are centred on `(0, 0)`; a solid's seat sits on `z = 0` and extrudes toward `+z`.

**One argument order:** what, how big (`across, up, out`), how many `sides`; then finish (`.chamfer`, `.corner`); then placement (`.move`, `.turn`, `.mirror`, `.on`, `.into`); material only at `p.put` / `p.sculpt`.

---

## 1. Hard-Surface Outlines & Solids

```python
from jig import *

# 2D outlines (u across, v up)
rect(w, h, cut=0)                 # cut: scalar, (top, bottom), or (tl, tr, br, bl)
ngon(sides, r, flat="+y")         # r from centre to flat
poly(points)                      # any simple polygon, concave allowed
stripes(region, pitch=0.02, turn=45.0)  # diagonal band outlines for p.mark()
  .inset(d)  .at(u, v)  .band(w)  # inset edges, translate, or make a frame strip

# 3D solid makers (all return Solid)
block(w, h, d)                    # centred w x h box from z=0 to z=d
box(x0, x1, y0, y1, z0, z1)       # axis-aligned box by explicit world bounds
extrude(outline, depth, draft=0, along="z", center=False)
body(front, side=None, top=None, depth=None, draft=0)  # intersect 2-3 orthographic views
loft((z0, out0), (z1, out1), ...) # matching corner counts, rising z
lathe([(along, r), ...], sides, flat="+y")             # revolve around z; r=0 closes cone
sweep(section, path, closed=False, up=None)            # mitred, twist-free sweep
ring(r, section, sides)           # closed torus/collar around z
bolt(across, proud=None, sides=6) # hex/square fastener head
hull(points)                      # convex hull of 3D points

# Finishing & transforms (return Solid)
.chamfer(b, where="shown")        # b scalar or (a, b) for "face1&face2"; "shown" skips seat
.corner(d, "front&top&right")     # flat 3-face corner cut
.move(x, y, z)  .turn(axis, deg, about=(0,0,0))  .mirror(axis)
.on(place, sink=0, lift=0)        # stand seat on Face, Frame, or list of places
.into(place, lift=0)              # orient cutter into a Face/Frame
a & b   /   a | b                 # convex intersection / union
```

## 2. Semantic Face Picking & Places

Never hardcode coordinates when you can ask a drawn `Part` for its faces:

```python
part.face("front", flat=False)             # one face (or largest if >=2x next); errors with full list otherwise
part.faces("right", by="y", flat=False, islands=False)  # sorted along axis; islands=True splits coplanar islands
# Words: front, back, top, bottom, right, left, out, seat, upright (join with "&", e.g. "top&out")

face.at(u=0, v=0, turn=0)                  # shifted frame on face (u right, v up; on level face v=-z)
face.lift(d)                               # shift along normal
face.grid(nu, nv=1, pitch=..., margin=0)   # list of nu*nv frames
face.corners(inset=0)                      # 4 corner frames (bl, br, tr, tl)
face.around(n, r=0, start=0)               # n frames around a circle, facing outward

frame(at=(0,0,0), z="+z", x=None)          # explicit local frame
floor(x=0, z=0)                            # y=0 frame with +z pointing straight up (+Y)
around(n, start=0)                         # n rotated frames around active frame's z
```

## 3. Model Statements (`p`)

```python
def model(p):
    p.budget(2000)                         # optional triangle budget check
    part = p.put(solid, "body2")           # draw laid solid (or QuadCage!)
    p.cut(cutter, role=None, mouth=0, of=None)  # boolean cut; mouth bevels rim; of=Part scopes cut
    floor = p.inset(face, margin_or_outline, depth, role=None, mouth=0)  # negative depth raises panel
    p.slots(face, region, n, depth, role="trim", mouth=0, along="u", bar_role=None)
    p.mark(face, outlines, "hazard")       # 0.3 mm proud surface patch
    p.paint(faces, "accent")               # repaint faces in-place (0 extra triangles)
    p.slot("screen", face, size=None)      # exports `slot:screen` empty node in .glb

    with p.on(place): ...                  # transform all statements inside into `place`
    with p.mirror("x"): ...                # mirror all statements inside across active frame axis
    with p.pose("door", hinge=(a, b), turn=(0, 90), at=0.5): ...  # hinge or slide=(dx,dy,dz)
    p.place(sub_fn, at=(0,0,0), turn=0, **knobs)  # reuse another model function
```

## 4. Organic & Curved SubD (`QuadCage`)

Use `QuadCage` for compound curves, aerospace shells, helmets, cushions, fairings, and organic forms — and mix freely with hard-surface `Solid`s in the same `model(p)`:

```python
cage = (
    QuadCage.ellipsoid(rx, ry, rz, center=(0,0,0), nu=12, nv=8)
    # Also: QuadCage.box_grid(center, size, segs=(2,2,2))
    #       QuadCage.loft_grid(rows_R_C_3, closed_u=True, cap_start=False, cap_end=False)
    #       QuadCage.uv_patch(grid_R_C_3)
    .sculpt(lambda pt: ...)                # per-vertex or vectorized deformer
    .mirror_x()                            # mirror +X across X=0, welding seam
    .tag(lambda c: c[2] > 0.08, "visor")   # tag faces by centroid / (c, tag) / (c, verts, tag)
    .crease("visor")                       # sharp edges: "all", "boundary", "tag_boundary", tag, or fn
    .subdivide(steps=1, pin_corners=False) # deterministic creased Catmull-Clark subdivision
    .drop("visor")                         # or .keep("visor")
    .thicken(0.012, inner_material="trim", rim_material="body2", bevel=0.004)
)
p.sculpt(cage, "panel")                    # or callable material: lambda c, tag: "accent" if ... else "panel"
p.sculpt(cage, "panel", smooth=True)       # smooth vertex normals (default False = crisp flat-shaded facets)

# Conform a curved patch onto any target QuadCage, Solid, or triangle array:
fitted = project_cage(target, patch, offset=0.002, relief_fn=lambda pt: ...)
```

## 5. Materials, Palettes & Script Knobs

Paint with any built-in PBR material (`body`, `body2`, `panel`, `panel2`, `trim`, `ink`, `slate`, `accent`, `hazard`, `steel`, `iron`, `aluminum`, `chrome`, `copper`, `brass`, `plastic`, `rubber`, `ceramic`, `wood`, `leather`, `cloth`, `foliage`, `glass`, `visor`, `screen`, `lamp`, `glow`, `led_red`, `led_amber`, `led_green`, `led_blue`), a hex string `"#d4552a"`, an RGB tuple `(210, 85, 42)`, or `Material("my_mat", (210, 85, 42), metallic=0.6, roughness=0.3, emissive=False, alpha=1.0)`.

Optional module-level constants in a `.py` script:
- `VARIANTS = {"name_a": dict(...), "name_b": dict(...)}` — multi-variant export (`jig build --all`)
- `LAID = True` (default `True`; set `False` for welded CSG)
- `MATERIALS = {"custom": Material(...)}` — register custom named PBR materials
- `PALETTE = "dark"` or `PALETTE = {"body": "#d0d4d8", "accent": "#2e9ad6"}` (or pass `--theme light|dark|industrial|warm|my_palette.json` on CLI, and `--themes` in `jig view` to compare all built-in themes) — override material colours cleanly without editing geometry
- `EYE = 1.7` / `CLOSE = [(x, y, z), (x, y, z)]` — customize `jig view` cameras
