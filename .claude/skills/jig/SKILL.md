---
name: jig
description: Model a 3D object (.glb) using jig's procedural hard-surface CAD and QuadCage SubD API in a fast multi-view visual feedback loop (jig view / jig build / jig describe).
argument-hint: <description of the 3D object to model>
---

Model this 3D object using `jig`: $ARGUMENTS

## Workflow

1. Read `SURFACE.md` and `AGENTS.md` (or run `uv run jig words` and `uv run jig materials`) if you haven't yet.
2. Scaffold or create the script (`uv run jig new <path>.py`) defining `def model(p):` with `from jig import *`.
3. Iterate in at least **4–6 visual rounds** using:
   - `uv run jig view <path>.py -o out/<name>.png` — open and inspect the rendered contact sheet after EVERY edit.
   - `uv run jig describe <path>.py [LINE]` — check bounding dimensions, `soft`/`deep`/`plain` metrics, face words/sizes, and zero flush/overhang warnings.
   - `uv run jig cost <path>.py` — keep triangle counts disciplined (trim wasteful chamfers on tiny repeated parts).
   - `uv run jig look out/<name>.glb [other.glb] --views 3q,front,side,back,top --slots` — check scale, silhouettes, and slots.
4. Build discipline across rounds:
   - **Silhouette first**: break the box with `body()`, `extrude(..., draft=)`, `rect(..., cut=)`, `loft()`, `lathe()`, or organic `QuadCage`.
   - **Secondary form**: sunk panels (`p.inset`), frames, collars, ribs, and `.chamfer(b)` on every visible outer edge (target `soft` 0.55–0.75 on hero models).
   - **Function & mechanics**: fasteners (`bolt().on(face.corners(...))`), vents (`p.slots`), cables (`sweep`), supports/feet, and restrained emissive indicators (`glow`/`lamp`/`led_*`).
   - **Check all four sides**: fix flat backs, floating parts, and every warning reported by `jig describe`.
5. Finish with `uv run jig build <path>.py -o out/<name>.glb` and report the `.glb` path, final preview PNG path, dimensions, triangle count, `soft`/`deep`/`plain` metrics, and rounds completed.
