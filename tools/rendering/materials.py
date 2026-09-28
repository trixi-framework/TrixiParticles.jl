"""Named material presets for the Blender render stage.

Three scene presets use the reviewed v03 material parameters:
``liquid`` (physical water, copper solids, glass tank, floor), ``foam``
(``liquid`` plus froth, spray, and bubbles), and ``stress`` (vertex-colored
stress surface with pedestal and optional ghost water).

Presets are selected with ``--material-preset``; any explicitly passed
material flag overrides the preset value for that flag. Camera, lights, tank
bounds, resolution, and other scene settings are never part of a preset.
The copper solid and low-iron glass materials are independently selectable
with ``--solid-material`` and ``--glass-material``. Clear water is a second
liquid material for other cases; v03 uses the turbulent-water selection.
"""

from __future__ import annotations

# Historical per-flag defaults, used when neither a preset nor an explicit
# flag supplies a value. Kept separate so presets stay reviewable.
MATERIAL_DEFAULTS = {
    "solid_floor_extension": 0.0,
    "solid_bevel": 0.0,
    "solid_bevel_segments": 3,
    "solid_bevel_angle": 24.0,
    "floor_bevel": 0.0,
    "pedestal_margin": 0.0,
    "pedestal_thickness": 0.08,
    "glass_specular": 0.04,
    "floor_coat": 0.0,
    "wall_cap_extension": 0.0,
    "stress_water_opacity": 0.0,
    "stress_normal_light": 0.0,
    "stress_light_direction": (0.35, -0.45, 0.82),
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
        "water_ior": 1.333,
        "water_absorption": (0.85, 0.14125, 0.02305),
        "water_scattering": (0.02, 0.04, 0.08),
        "water_scattering_anisotropy": 0.35,
    },
    "clear-water": {
        # Unscaled measured pure-water absorption; intended for non-v03 cases.
        "water_color": (1.0, 1.0, 1.0),
        "water_roughness": 0.02,
        "water_ior": 1.333,
        "water_absorption": (0.34, 0.0565, 0.00922),
        "water_scattering": (0.0, 0.0, 0.0),
        "water_scattering_anisotropy": 0.0,
    },
}

_LIQUID = {
    **LIQUID_MATERIALS["turbulent-water"],
    **SOLID_MATERIALS["anodized-copper"],
    # Render-only mesh offsets remain scene settings.
    "solid_floor_extension": 0.02,
    "solid_bevel": 0.005,
    # Tank base.
    "floor_color": (0.012, 0.018, 0.028),
    "floor_metallic": 0.35,
    "floor_roughness": 0.30,
    "floor_coat": 0.0,
    # Reusable named tank-glass material.
    **GLASS_MATERIALS["low-iron-glass"],
    # Tank construction.
    "wall_thickness": 0.018,
    "floor_thickness": 0.018,
    "visible_wall_height": 0.16,
}

_FOAM = {
    # Whitewater froth surface.
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
    # Velocity-aligned spray.
    "spray_color": (0.82, 0.94, 1.0),
    "spray_roughness": 0.025,
    "spray_radius": 0.0008,
    "spray_scale_range": (0.55, 1.15),
    "spray_exposure": 0.001,
    "spray_max_stretch": 4.0,
    # Submerged bubble interfaces (relative IOR air over water).
    "bubble_color": (0.9, 0.97, 1.0),
    "bubble_roughness": 0.025,
    "bubble_radius": 0.0014,
    "bubble_scale_range": (0.35, 1.75),
    "bubble_relative_ior": 1.0 / 1.333,
}

_STRESS = {
    # The stress view colors the surface from its own vertex colors; only the
    # pedestal, edge treatment, and optional ghost water use materials.
    "water_color": (0.015, 0.28, 0.46),
    "water_roughness": 0.18,
    "water_ior": 1.333,
    "stress_water_transmission": 0.72,
    "stress_water_specular": 0.18,
    "stress_water_opacity": 0.005,
    "stress_normal_light": 0.28,
    "stress_light_direction": (0.35, -0.45, 0.82),
    "solid_floor_extension": 0.02,
    "solid_bevel": 0.004,
    "floor_color": (0.020, 0.032, 0.050),
    "floor_metallic": 0.3,
    "floor_roughness": 0.4,
    "floor_coat": 0.10,
}

MATERIAL_PRESETS = {
    "liquid": dict(_LIQUID),
    "foam": {**_LIQUID, **_FOAM},
    "stress": dict(_STRESS),
}

PRESET_DESCRIPTIONS = {
    "liquid": "Physical water, copper solids, glass tank, floor (clean surface view)",
    "foam": "The liquid set plus froth, spray, and bubble materials (whitewater view)",
    "stress": "Vertex-colored stress surface with pedestal and optional ghost water",
}


def describe_presets() -> str:
    lines = ["Material presets (explicit flags override preset values):", ""]
    for name in MATERIAL_PRESETS:
        lines.append(f"  {name:<8} {PRESET_DESCRIPTIONS[name]}")
        for key in sorted(MATERIAL_PRESETS[name]):
            lines.append(f"    {key} = {MATERIAL_PRESETS[name][key]!r}")
        lines.append("")
    lines.append("Component materials (may be selected alongside a scene preset):")
    for selector, choices in (("--liquid-material", LIQUID_MATERIALS),
                              ("--solid-material", SOLID_MATERIALS),
                              ("--glass-material", GLASS_MATERIALS)):
        for name, values in choices.items():
            lines.append(f"  {selector} {name}")
            for key in sorted(values):
                lines.append(f"    {key} = {values[key]!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
