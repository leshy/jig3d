"""jig — Agent-native procedural 3D solid & SubD modelling API for deterministic glTF 2.0 (.glb)."""

from __future__ import annotations

import math
from math import pi

import numpy as np

from ._places import Face, Frame, around, floor, frame
from .glb import pack_glb, write_glb
from .materials import BUILTIN_MATERIALS, THEMES, Material, Palette
from .metrics import measure, measure_file
from .scene import Model, load_script, run_script
from .sculpt import QuadCage, project_cage
from .solid import (
    Shape,
    annulus,
    between,
    bevelled,
    cyl,
    flat_poly,
    frustum,
    half,
    plane,
    prism,
    revolved,
    rod,
    rounded,
    rounded_box,
    slab,
    sym,
)
from .words import (
    Outline,
    Solid,
    block,
    body,
    bolt,
    box,
    extrude,
    hull,
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

__all__ = [
    # Stdlib / numeric convenience
    "math",
    "np",
    "pi",
    # Outlines
    "Outline",
    "rect",
    "ngon",
    "poly",
    "stripes",
    # Hard-surface solids
    "Solid",
    "Shape",
    "extrude",
    "body",
    "loft",
    "block",
    "box",
    "hull",
    "lathe",
    "sweep",
    "ring",
    "bolt",
    # Extra half-space primitives
    "plane",
    "sym",
    "slab",
    "half",
    "prism",
    "between",
    "cyl",
    "annulus",
    "revolved",
    "rounded",
    "rounded_box",
    "rod",
    "frustum",
    "flat_poly",
    "bevelled",
    # Places & frames
    "Frame",
    "Face",
    "frame",
    "floor",
    "around",
    # Organic SubD sculpting
    "QuadCage",
    "project_cage",
    # Model & materials & export
    "Model",
    "Material",
    "Palette",
    "BUILTIN_MATERIALS",
    "THEMES",
    "run_script",
    "load_script",
    "pack_glb",
    "write_glb",
    # Measurement
    "tris",
    "volume",
    "measure",
    "measure_file",
]
