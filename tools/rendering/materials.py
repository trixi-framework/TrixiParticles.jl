"""Independent Blender material choices; no material bundles a whole scene.

The v03 appearance uses turbulent water, copper solids, low-iron glass,
dark-metal floor, whitewater froth/spray/bubbles, and a separate stress ghost.
These are distinct selectors. Geometry, camera, light and render settings are
not included in material dictionaries. Explicit property flags take precedence.
"""

from __future__ import annotations

# Neutral *scene* defaults, not material parameters. Materials must be selected
# by role or supplied explicitly through per-property flags.
SCENE_DEFAULTS = {
    "solid_floor_extension": 0.0,
    "solid_bevel": 0.0,
    "solid_bevel_segments": 3,
    "solid_bevel_angle": 24.0,
    "floor_bevel": 0.0,
    "pedestal_margin": 0.0,
    "pedestal_thickness": 0.08,
    "wall_cap_extension": 0.0,
    "instance_seed": 0,
    "droplet_subdivisions": 2,
}

SOLID_MATERIALS = {
    "anodized-copper": {
        "solid_color": (0.62, 0.10, 0.018),
        "solid_metallic": 0.92,
        "solid_roughness": 0.30,
        "solid_coat": 0.16,
    },
    "steel-uncoated": {
        "solid_color": (0.53, 0.58, 0.63),
        "solid_metallic": 1.0,
        "solid_roughness": 0.28,
        "solid_coat": 0.0,
        "solid_noise_scale": 240.0,
        "solid_bump_strength": 0.06,
        "solid_bump_distance": 0.0003,
    },
    "steel-white-semigloss": {
        # A dielectric white paint layer over steel; the coating is not metallic.
        "solid_color": (0.83, 0.86, 0.88),
        "solid_metallic": 0.0,
        "solid_roughness": 0.34,
        "solid_coat": 0.25,
    },
    "concrete": {
        "solid_color": (0.065, 0.075, 0.073),
        "solid_metallic": 0.0,
        "solid_roughness": 0.99,
        "solid_coat": 0.0,
        "solid_specular_level": 0.03,
        "solid_noise_scale": 90.0,
        "solid_bump_strength": 0.65,
        "solid_bump_distance": 0.008,
        "solid_inclusion_color": (0.028, 0.033, 0.035),
        "solid_inclusion_scale": 18.0,
        "solid_inclusion_threshold": 0.28,
        "solid_inclusion_transition": 0.10,
        "solid_inclusion_warp_scale": 36.0,
        "solid_inclusion_distortion": 0.025,
    },
}

GLASS_MATERIALS = {
    "low-iron-glass": {
        "glass_color": (0.06, 0.42, 0.34),
        "glass_opacity": 0.04,
        "glass_roughness": 0.30,
        "glass_ior": 1.36,
        "glass_transmission": 0.90,
        "glass_specular": 0.04,
    },
}

LIQUID_MATERIALS = {
    "turbulent-water": {
        # v03: Pope and Fry (1997) absorption scaled by 2.5, with low scattering.
        "water_color": (0.35, 0.72, 1.0),
        "water_roughness": 0.06,
        "water_transmission": 1.0,
        "water_ior": 1.333,
        "water_absorption": (0.85, 0.14125, 0.02305),
        "water_scattering": (0.02, 0.04, 0.08),
        "water_scattering_anisotropy": 0.35,
    },
    "clear-water": {
        # Unscaled measured pure-water absorption; intended for non-v03 cases.
        "water_color": (1.0, 1.0, 1.0),
        "water_roughness": 0.02,
        "water_transmission": 1.0,
        "water_ior": 1.333,
        "water_absorption": (0.34, 0.0565, 0.00922),
        "water_scattering": (0.0, 0.0, 0.0),
        "water_scattering_anisotropy": 0.0,
    },
    "heavy-oil": {
        # Illustrative optical appearance; fluid density/viscosity are solver inputs.
        "water_color": (0.30, 0.17, 0.055),
        "water_roughness": 0.19,
        "water_transmission": 1.0,
        "water_ior": 1.47,
        "water_absorption": (1.1, 2.5, 5.5),
        "water_scattering": (0.012, 0.006, 0.002),
        "water_scattering_anisotropy": 0.15,
    },
    "melted-plastic": {
        # Muted opaque polymer, not an emissive or thermal/rheological model.
        "water_color": (0.13, 0.045, 0.015),
        "water_roughness": 0.42,
        "water_transmission": 0.0,
        "water_specular_level": 0.10,
        "water_ior": 1.48,
        "water_absorption": (0.45, 1.2, 2.8),
        "water_scattering": (0.0, 0.0, 0.0),
        "water_scattering_anisotropy": 0.0,
    },
}

FLOOR_MATERIALS = {
    "dark-metal": {
        "floor_color": (0.012, 0.018, 0.028),
        "floor_metallic": 0.35,
        "floor_roughness": 0.30,
        "floor_coat": 0.0,
    },
    "neutral-stress": {
        "floor_color": (0.020, 0.032, 0.050),
        "floor_metallic": 0.30,
        "floor_roughness": 0.40,
        "floor_coat": 0.10,
    },
}

FOAM_MATERIALS = {
    "whitewater-froth": {
        "foam_color": (0.72, 0.82, 0.86),
        "foam_roughness": 0.46,
        "foam_transmission": 0.04,
        "foam_subsurface": 0.035,
        "foam_point_radius": 0.006,
        "foam_voxel_size": 0.004,
        "foam_threshold": 0.5,
        "foam_adaptivity": 0.12,
        "foam_noise_scale": 220.0,
        "foam_noise_detail": 3.0,
        "foam_noise_roughness": 0.7,
        "foam_micro_bump": 0.00045,
        "foam_bump_strength": 0.22,
    },
}

SPRAY_MATERIALS = {
    "water-droplet": {
        "spray_color": (0.82, 0.94, 1.0),
        "spray_roughness": 0.025,
        "spray_radius": 0.0008,
        "spray_scale_range": (0.55, 1.15),
        "spray_exposure": 0.001,
        "spray_max_stretch": 4.0,
    },
}

BUBBLE_MATERIALS = {
    "submerged-air": {
        "bubble_color": (0.9, 0.97, 1.0),
        "bubble_roughness": 0.025,
        "bubble_radius": 0.0014,
        "bubble_scale_range": (0.35, 1.75),
        "bubble_relative_ior": 1.0 / 1.333,
    },
}

GHOST_MATERIALS = {
    "v03-ghost-water": {
        "water_color": (0.015, 0.28, 0.46),
        "water_roughness": 0.18,
        "water_ior": 1.333,
        "stress_water_transmission": 0.72,
        "stress_water_specular": 0.18,
        "stress_water_opacity": 0.005,
    },
}

MATERIAL_SELECTORS = (
    ("liquid_material", "--liquid-material", LIQUID_MATERIALS),
    ("solid_material", "--solid-material", SOLID_MATERIALS),
    ("glass_material", "--glass-material", GLASS_MATERIALS),
    ("floor_material", "--floor-material", FLOOR_MATERIALS),
    ("foam_material", "--foam-material", FOAM_MATERIALS),
    ("spray_material", "--spray-material", SPRAY_MATERIALS),
    ("bubble_material", "--bubble-material", BUBBLE_MATERIALS),
    ("ghost_material", "--ghost-material", GHOST_MATERIALS),
)


def describe_materials() -> str:
    lines = ["Independent material choices (explicit flags override each):", ""]
    for _, selector, choices in MATERIAL_SELECTORS:
        for name, values in choices.items():
            lines.append(f"  {selector} {name}")
            for key in sorted(values):
                lines.append(f"    {key} = {values[key]!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
