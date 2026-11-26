"""Domain-neutral PBR materials and palette overrides for deterministic glTF 2.0 export.

Paint surfaces by built-in material name (`"body"`, `"panel"`, `"trim"`, `"steel"`, `"copper"`,
`"glass"`, `"glow"`, `"wood"`, `"rubber"`, ...), hex colour (`"#d4552a"`), `(r, g, b)` tuple,
or explicit `Material(...)`. Projects can optionally pass a JSON/dict colour-override palette.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Material:
    """A PBR glTF 2.0 material definition."""

    name: str
    rgb: tuple[int, int, int]
    metallic: float = 0.15
    roughness: float = 0.55
    emissive: bool = False
    emissive_strength: float = 1.0
    alpha: float = 1.0
    double_sided: bool = False

    def gltf_dict(self, emissive_texture_index: int | None = None) -> dict:
        """Convert to a glTF 2.0 material object."""
        lin = srgb_to_linear(self.rgb)
        mat: dict = {
            "name": self.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [lin[0], lin[1], lin[2], float(self.alpha)],
                "metallicFactor": float(self.metallic),
                "roughnessFactor": float(self.roughness),
            },
            "alphaMode": "BLEND" if self.alpha < 0.999 else "OPAQUE",
        }
        if self.double_sided or self.alpha < 0.999:
            mat["doubleSided"] = True
        if self.emissive:
            mat["emissiveFactor"] = [
                min(1.0, lin[0] * self.emissive_strength),
                min(1.0, lin[1] * self.emissive_strength),
                min(1.0, lin[2] * self.emissive_strength),
            ]
            if emissive_texture_index is not None:
                mat["emissiveTexture"] = {"index": emissive_texture_index}
        return mat


def _hex_to_rgb(h: int | str) -> tuple[int, int, int]:
    if isinstance(h, int):
        return ((h >> 16) & 255, (h >> 8) & 255, h & 255)
    s = h.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise ValueError(f"invalid hex colour: {h!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def srgb_to_linear(rgb: tuple[int, int, int]) -> list[float]:
    out = []
    for v in rgb:
        c = v / 255.0
        out.append(round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 6))
    return out


# Clean, general-purpose PBR material defaults (canonically ordered for deterministic glTF indices).
BUILTIN_MATERIALS: tuple[Material, ...] = (
    # Structural neutrals & accents
    Material("body", _hex_to_rgb(0x64686E), metallic=0.28, roughness=0.52),
    Material("body2", _hex_to_rgb(0x52575E), metallic=0.30, roughness=0.50),
    Material("panel", _hex_to_rgb(0xA6ABB2), metallic=0.18, roughness=0.50),
    Material("panel2", _hex_to_rgb(0x8C9198), metallic=0.20, roughness=0.55),
    Material("trim", _hex_to_rgb(0x2D3035), metallic=0.22, roughness=0.56),
    Material("ink", _hex_to_rgb(0x111214), metallic=0.08, roughness=0.72),
    Material("slate", _hex_to_rgb(0x49667E), metallic=0.20, roughness=0.48),
    Material("accent", _hex_to_rgb(0xE89B3C), metallic=0.10, roughness=0.44),
    Material("hazard", _hex_to_rgb(0xEA9F16), metallic=0.08, roughness=0.50),
    # Metals
    Material("steel", _hex_to_rgb(0x8D949B), metallic=0.78, roughness=0.32),
    Material("iron", _hex_to_rgb(0x45494E), metallic=0.72, roughness=0.46),
    Material("aluminum", _hex_to_rgb(0xC5CCD2), metallic=0.84, roughness=0.26),
    Material("chrome", _hex_to_rgb(0xD8DEE4), metallic=0.92, roughness=0.14),
    Material("copper", _hex_to_rgb(0xB96423), metallic=0.76, roughness=0.34),
    Material("brass", _hex_to_rgb(0xC9A24B), metallic=0.82, roughness=0.28),
    # Non-metals & organics
    Material("plastic", _hex_to_rgb(0xD0D4D9), metallic=0.0, roughness=0.38),
    Material("rubber", _hex_to_rgb(0x202226), metallic=0.0, roughness=0.90),
    Material("ceramic", _hex_to_rgb(0xE8E6DF), metallic=0.0, roughness=0.22),
    Material("wood", _hex_to_rgb(0x8B5E3C), metallic=0.0, roughness=0.72),
    Material("leather", _hex_to_rgb(0x5A3A28), metallic=0.0, roughness=0.62),
    Material("cloth", _hex_to_rgb(0x565F69), metallic=0.0, roughness=0.90),
    Material("foliage", _hex_to_rgb(0x3B5935), metallic=0.0, roughness=0.78),
    # Optical & emissive
    Material("glass", _hex_to_rgb(0x8194A0), metallic=0.10, roughness=0.12, alpha=0.42, double_sided=True),
    Material("visor", _hex_to_rgb(0x38586E), metallic=0.15, roughness=0.10, alpha=0.50, double_sided=True),
    Material("screen", _hex_to_rgb(0x22453B), metallic=0.05, roughness=0.20, emissive=True, emissive_strength=0.65),
    Material("lamp", _hex_to_rgb(0xFFFFFF), metallic=0.0, roughness=0.25, emissive=True),
    Material("glow", _hex_to_rgb(0x00F0D8), metallic=0.0, roughness=0.25, emissive=True),
    Material("led_red", _hex_to_rgb(0xFF1F38), metallic=0.0, roughness=0.25, emissive=True),
    Material("led_amber", _hex_to_rgb(0xFFB300), metallic=0.0, roughness=0.25, emissive=True),
    Material("led_green", _hex_to_rgb(0x5CE62E), metallic=0.0, roughness=0.25, emissive=True),
    Material("led_blue", _hex_to_rgb(0x38A8FF), metallic=0.0, roughness=0.25, emissive=True),
)

DEFAULT_BY_NAME: dict[str, Material] = {m.name: m for m in BUILTIN_MATERIALS}

# Domain-neutral built-in colour themes (remap semantic material roles without touching geometry).
THEMES: dict[str, dict[str, str]] = {
    "default": {},
    "light": {
        "body": "#dce0e4",
        "body2": "#c3c9d0",
        "panel": "#eff2f5",
        "panel2": "#cdd3da",
        "trim": "#383e46",
        "ink": "#1a1d22",
        "slate": "#4e7ca0",
        "accent": "#2f90d8",
        "hazard": "#f0a418",
        "glow": "#5ee6ff",
        "screen": "#1f4b5e",
    },
    "dark": {
        "body": "#33373d",
        "body2": "#25282e",
        "panel": "#494f58",
        "panel2": "#3b4048",
        "trim": "#191b1f",
        "ink": "#0d0e10",
        "slate": "#2f4456",
        "accent": "#e8742a",
        "hazard": "#e59a18",
        "glow": "#ff9b28",
        "screen": "#2b3830",
    },
    "industrial": {
        "body": "#6b705c",
        "body2": "#555b48",
        "panel": "#9c9d89",
        "panel2": "#7f8370",
        "trim": "#2d2f2b",
        "ink": "#171816",
        "slate": "#4e5d52",
        "accent": "#e39a1c",
        "hazard": "#f2b705",
        "glow": "#ffd24c",
        "screen": "#2c422c",
    },
    "warm": {
        "body": "#b9ad99",
        "body2": "#9c8f7a",
        "panel": "#d8cebc",
        "panel2": "#c2b6a1",
        "trim": "#3d342d",
        "ink": "#1e1915",
        "slate": "#5b6968",
        "accent": "#c9522d",
        "hazard": "#df8c1e",
        "glow": "#6ef0a4",
        "screen": "#204638",
    },
}


def load_palette_overrides(source: str | Path | dict[str, str] | None) -> dict[str, str]:
    """Load `{material_name: "#rrggbb"}` overrides from a built-in theme name, dict, or JSON file."""
    if source is None:
        return {}
    if isinstance(source, dict):
        return {str(k): str(v) for k, v in source.items()}
    if isinstance(source, str) and source in THEMES:
        return dict(THEMES[source])
    p = Path(source)
    if not p.is_file():
        raise FileNotFoundError(
            f"unknown theme or missing palette file: {source!r} (built-in themes: {', '.join(THEMES)})"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"palette JSON must be a mapping of name -> '#rrggbb', got {type(data).__name__}")
    return {str(k): str(v) for k, v in data.items()}


class Palette:
    """Mutable palette mapping material specs (name, `#rrggbb`, `(r, g, b)`, or `Material`) to u16 ids.

    Index 0 is reserved for empty/cut; valid material ids start at 1.
    """

    def __init__(
        self,
        overrides: str | Path | dict[str, str] | None = None,
        extra: dict[str, Material] | None = None,
    ):
        self._materials: list[Material | None] = [None]
        self._by_name: dict[str, int] = {}
        ov = load_palette_overrides(overrides)

        for mat in BUILTIN_MATERIALS:
            if mat.name in ov:
                mat = Material(
                    name=mat.name,
                    rgb=_hex_to_rgb(ov[mat.name]),
                    metallic=mat.metallic,
                    roughness=mat.roughness,
                    emissive=mat.emissive,
                    emissive_strength=mat.emissive_strength,
                    alpha=mat.alpha,
                    double_sided=mat.double_sided,
                )
            self.register(mat)

        # Any extra colour override names not in built-ins become plain PBR materials.
        for name, hex_col in ov.items():
            if name not in self._by_name:
                self.register(Material(name=name, rgb=_hex_to_rgb(hex_col)))

        if extra:
            for name, mat in extra.items():
                if mat.name != name:
                    mat = Material(
                        name=name,
                        rgb=mat.rgb,
                        metallic=mat.metallic,
                        roughness=mat.roughness,
                        emissive=mat.emissive,
                        emissive_strength=mat.emissive_strength,
                        alpha=mat.alpha,
                        double_sided=mat.double_sided,
                    )
                self.register(mat)

    def register(self, mat: Material) -> int:
        if mat.name in self._by_name:
            idx = self._by_name[mat.name]
            self._materials[idx] = mat
            return idx
        idx = len(self._materials)
        self._materials.append(mat)
        self._by_name[mat.name] = idx
        return idx

    def resolve(self, spec: str | tuple[int, int, int] | Material) -> tuple[int, Material]:
        """Resolve a material name, `#rrggbb`, `(r, g, b)` tuple, or `Material` to `(id, Material)`."""
        if isinstance(spec, Material):
            idx = self.register(spec)
            return idx, spec
        if isinstance(spec, str):
            if spec in self._by_name:
                idx = self._by_name[spec]
                return idx, self._materials[idx]  # type: ignore[return-value]
            if spec.startswith("#"):
                rgb = _hex_to_rgb(spec)
                name = f"color_{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                mat = Material(name=name, rgb=rgb)
                idx = self.register(mat)
                return idx, mat
            raise KeyError(f"unknown material {spec!r}; built-ins: {', '.join(sorted(DEFAULT_BY_NAME))}")
        if (
            isinstance(spec, tuple)
            and len(spec) == 3
            and all(isinstance(c, (int, np.integer)) and 0 <= int(c) <= 255 for c in spec)
        ):
            rgb = (int(spec[0]), int(spec[1]), int(spec[2]))
            name = f"color_{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            mat = Material(name=name, rgb=rgb)
            idx = self.register(mat)
            return idx, mat
        raise TypeError(f"material must be a name, '#rrggbb', (r, g, b), or Material; got {spec!r}")

    def id_of(self, spec: str | tuple[int, int, int] | Material) -> int:
        return self.resolve(spec)[0]

    def material_of(self, idx: int) -> Material:
        m = self._materials[int(idx)]
        if m is None:
            raise KeyError(f"invalid material index {idx}")
        return m

    def name_of(self, idx: int) -> str:
        return self.material_of(idx).name

    def __getitem__(self, spec) -> int:
        return self.id_of(spec)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def rgb_table(self) -> np.ndarray:
        rows = [(0, 0, 0)] + [m.rgb for m in self._materials[1:] if m is not None]
        return np.array(rows, dtype=np.uint8)

    def items(self):
        return self._by_name.items()
