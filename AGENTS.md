# Agent Guide for `jig`

Read [SURFACE.md](SURFACE.md) before writing model scripts. Run `jig words` for live signatures + measured triangle counts, and `jig materials` for built-in PBR materials.

## The Iterative Modelling Loop

Never one-shot a 3D model and call it done. Work in fast build-and-look rounds (2–5 seconds each):

```bash
uv run jig new my_model.py
uv run jig view my_model.py -o out/my_model.png          # inspect every round!
uv run jig describe my_model.py                         # faces, warnings, slots, metrics
uv run jig cost my_model.py                             # per-line triangle cost
uv run jig build my_model.py -o out/my_model.glb        # final deterministic PBR .glb
```

1. **Round 1 — Primary silhouette**: establish proportions, draft/taper, cut corners, or orthographic `body()` / `QuadCage` shell. Never start and stop at a bare box.
2. **Round 2 — Secondary form**: add housings, recessed panels (`p.inset`), bezels, flanges, ribs, mounting plates, and chamfer every visible outer edge (`.chamfer(b)` — aim for `soft` in `0.55–0.75` on hero models).
3. **Round 3 — Functional engineering**: show how the object stands, mounts, opens, cools, or connects (feet, bolted brackets, louvers via `p.slots`, cables via `sweep`, indicators via `glow`/`led_*`).
4. **Round 4+ — All-angle polish**: inspect the 4× three-quarter views, side elevation, eye-level shot, and close-ups on `out/<name>.png`. Fix flat backs, floating parts, overhanging seats, or flush overlap warnings from `jig describe`.
5. **Final check**: zero warnings from `jig describe`, within triangle budget, `jig build` writes clean deterministic `.glb`.

## Craft Rules

- **Real, not toy**: proportions, fastener sizes, indicator lamps, and wall thicknesses should read as engineered objects, not oversized cartoon props.
- **Colour restraint**: neutrals carry the form; 1–2 tones for body/frame, dark recessed trim/ink for contrast, one accent color used sparingly, glow only where light has a purpose.
- **Use semantic faces, not raw math**: write `bezel.on(shell.face("front").at(0, -0.02))` and `p.mirror("x")` instead of hand-computing trig and offsets.
- **Use `QuadCage` for compound curves**: don't approximate organic shells or rounded canopies with stacked boxes — sculpt a `QuadCage`, crease openings, `.subdivide(1)`, and `.thicken(..., bevel=...)`, then add hard-surface CAD hardware with `p.put(...)`.
- **Inspect warnings**: if `jig describe` prints `lies flush over line N` or `stands X mm past the face it sits on`, fix the sink/lift/inset directly in the script.
