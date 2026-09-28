#!/usr/bin/env python3
"""FoamGenerator and Blender tools; all case settings are command-line parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from data import axis_permutation, bounds
from materials import BLADE_MATERIALS, GLASS_MATERIALS, MATERIAL_DEFAULTS, MATERIAL_PRESETS


def positive(value: str) -> float:
    from math import isfinite
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def fraction(value: str) -> float:
    from math import isfinite
    number = float(value)
    if not isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return number


def rgb(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3 or any(not 0 <= value <= 1 for value in values):
        raise ValueError("RGB colors must have three components in [0, 1]")
    return tuple(values)


def light_specification(value: str):
    """name:x:y:z:energy:r:g:b:size for one area light."""
    fields = value.split(":")
    if len(fields) != 9 or not fields[0]:
        raise argparse.ArgumentTypeError("light must be name:x:y:z:energy:r:g:b:size")
    try:
        x, y, z, energy, red, green, blue, size = map(float, fields[1:])
        rgb([red, green, blue])
        from math import isfinite
        if not all(isfinite(number) for number in (x, y, z, energy, red, green, blue, size)) or \
                energy <= 0 or size <= 0:
            raise ValueError("light coordinates, size and energy must be finite and physical")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return (fields[0], (x, y, z), energy, (red, green, blue), size)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    foam = commands.add_parser("foam", help="Generate sequential SPlisHSPlasH markers")
    foam.add_argument("--fluid-pvd", required=True, type=Path)
    foam.add_argument("--generator", required=True, type=Path)
    foam.add_argument("--generator-source", type=Path,
                      help="Optional patched main.cpp to fingerprint")
    foam.add_argument("--generator-patch", type=Path,
                      help="Optional build patch to fingerprint")
    foam.add_argument("--output", required=True, type=Path)
    foam.add_argument("--domain-min", nargs=3, type=float, required=True)
    foam.add_argument("--domain-max", nargs=3, type=float, required=True)
    foam.add_argument("--generator-radius", required=True, type=positive)
    foam.add_argument("--foam-scale", required=True, type=positive)
    foam.add_argument("--lifetime-min", required=True, type=positive)
    foam.add_argument("--lifetime-max", required=True, type=positive)
    foam.add_argument("--buoyancy", required=True, type=float)
    foam.add_argument("--drag", required=True, type=fraction)
    foam.add_argument("--drag-reference-step", required=True, type=positive)
    foam.add_argument("--seed", required=True, type=int)
    foam.add_argument("--generator-threads", required=True, type=positive)
    foam.add_argument("--velocity-array", default="velocity")
    foam.add_argument("--output-axis-order", choices=("xyz", "xzy"), required=True)
    foam.add_argument("--reuse", action="store_true",
                      help="Reuse only a complete, hash-validated whitewater cache")

    render = commands.add_parser("render", help="Render a surface/whitewater or stress scene")
    render.add_argument("--blender", required=True, type=Path)
    render.add_argument("--surface-dir", required=True, type=Path)
    source = render.add_mutually_exclusive_group(required=True)
    source.add_argument("--surface-pvd", type=Path,
                        help="Native PR #1329 VTP collection, with matching PLY stems")
    source.add_argument("--surface-metadata", type=Path,
                        help="Saved reconstruction frame inventory (data, not a case config)")
    render.add_argument("--foam-dir", type=Path, help="Complete whitewater-stage output")
    render.add_argument("--output", required=True, type=Path)
    selection = render.add_mutually_exclusive_group(required=True)
    selection.add_argument("--frame", type=int)
    selection.add_argument("--sequence", action="store_true")
    render.add_argument("--start", type=int, default=0)
    render.add_argument("--stop", type=int)
    render.add_argument("--stride", type=int, default=1)
    render.add_argument("--resume", action="store_true")
    render.add_argument("--solid-pattern", action="append", default=[],
                        help="PLY pattern with {frame:06d}, {iter} filename suffix, {time}")
    render.add_argument("--surface-axis-order", choices=("xyz", "xzy"), required=True)
    render.add_argument("--foam-axis-order", choices=("xyz", "xzy"), required=True)
    render.add_argument("--mode", choices=("liquid", "stress"), default="liquid")
    render.add_argument("--material-preset", choices=tuple(MATERIAL_PRESETS),
                        help="Named v03 material set; explicit material flags override it")
    render.add_argument("--blade-material", choices=tuple(BLADE_MATERIALS),
                        help="Named solid material, independently selectable")
    render.add_argument("--glass-material", choices=tuple(GLASS_MATERIALS),
                        help="Named tank-glass material, independently selectable")
    render.add_argument("--stress-mesh", action="append", type=Path, default=[],
                        help="Pre-colored stress PLY; repeat for each solid")
    render.add_argument("--stress-water-opacity", type=fraction)
    render.add_argument("--stress-water-transmission", type=fraction)
    render.add_argument("--stress-water-specular", type=fraction)
    render.add_argument("--stress-normal-light", type=fraction)
    render.add_argument("--stress-light-direction", nargs=3, type=float)

    render.add_argument("--camera-position", nargs=3, required=True, type=float)
    render.add_argument("--camera-target", nargs=3, required=True, type=float)
    render.add_argument("--camera-fov", required=True, type=positive)
    render.add_argument("--tank-min", nargs=3, required=True, type=float)
    render.add_argument("--tank-max", nargs=3, required=True, type=float)
    render.add_argument("--light", action="append", type=light_specification, required=True)
    render.add_argument("--width", required=True, type=int)
    render.add_argument("--height", required=True, type=int)
    render.add_argument("--samples", required=True, type=int)
    render.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    render.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    render.add_argument("--gpu-backend", choices=("OPTIX", "CUDA", "HIP", "ONEAPI"),
                        default="OPTIX")
    render.add_argument("--view-transform", default="AgX")
    render.add_argument("--look", default="AgX - Medium High Contrast")
    render.add_argument("--exposure", type=float, default=0.0)
    render.add_argument("--gamma", type=positive, default=1.0)
    render.add_argument("--max-bounces", type=int, default=10)
    render.add_argument("--transmission-bounces", type=int, default=8)
    render.add_argument("--transparent-bounces", type=int, default=8)
    render.add_argument("--png-compression", type=int, default=20)
    render.add_argument("--no-denoise", action="store_true")
    render.add_argument("--world-color", nargs=3, type=float, required=True)
    render.add_argument("--world-strength", type=positive, required=True)

    render.add_argument("--water-color", nargs=3, type=float)
    render.add_argument("--water-roughness", type=fraction)
    render.add_argument("--water-ior", type=positive)
    render.add_argument("--water-absorption", nargs=3, type=float)
    render.add_argument("--water-scattering", nargs=3, type=float)
    render.add_argument("--water-scattering-anisotropy", type=float)
    render.add_argument("--blade-color", nargs=3, type=float)
    render.add_argument("--blade-metallic", type=fraction)
    render.add_argument("--blade-roughness", type=fraction)
    render.add_argument("--blade-coat", type=fraction)
    render.add_argument("--blade-floor-extension", type=float, default=None)
    render.add_argument("--blade-bevel", type=float, default=None)
    render.add_argument("--blade-bevel-segments", type=int, default=None)
    render.add_argument("--blade-bevel-angle", type=float, default=None)
    render.add_argument("--floor-color", nargs=3, type=float)
    render.add_argument("--floor-metallic", type=fraction)
    render.add_argument("--floor-roughness", type=fraction)
    render.add_argument("--floor-coat", type=fraction)
    render.add_argument("--floor-bevel", type=float, default=None)
    render.add_argument("--pedestal-margin", type=float, default=None)
    render.add_argument("--pedestal-thickness", type=positive, default=None)
    render.add_argument("--glass-color", nargs=3, type=float)
    render.add_argument("--glass-opacity", type=fraction)
    render.add_argument("--glass-roughness", type=fraction)
    render.add_argument("--glass-ior", type=positive)
    render.add_argument("--glass-transmission", type=fraction)
    render.add_argument("--glass-specular", type=fraction, default=None)
    render.add_argument("--wall-thickness", type=positive)
    render.add_argument("--floor-thickness", type=positive)
    render.add_argument("--visible-wall-height", type=positive)
    render.add_argument("--wall-cap-extension", type=float, default=None)

    render.add_argument("--foam-color", nargs=3, type=float)
    render.add_argument("--foam-point-radius", type=positive)
    render.add_argument("--foam-voxel-size", type=positive)
    render.add_argument("--foam-threshold", type=positive)
    render.add_argument("--foam-adaptivity", type=fraction)
    render.add_argument("--foam-roughness", type=fraction)
    render.add_argument("--foam-transmission", type=fraction)
    render.add_argument("--foam-subsurface", type=fraction)
    render.add_argument("--foam-micro-bump", type=float)
    render.add_argument("--foam-noise-scale", type=positive)
    render.add_argument("--foam-noise-detail", type=positive)
    render.add_argument("--foam-noise-roughness", type=fraction)
    render.add_argument("--foam-bump-strength", type=fraction)
    render.add_argument("--spray-color", nargs=3, type=float)
    render.add_argument("--spray-roughness", type=fraction)
    render.add_argument("--spray-radius", type=positive)
    render.add_argument("--spray-scale-range", nargs=2, type=positive)
    render.add_argument("--spray-exposure", type=positive)
    render.add_argument("--spray-max-stretch", type=positive)
    render.add_argument("--bubble-color", nargs=3, type=float)
    render.add_argument("--bubble-roughness", type=fraction)
    render.add_argument("--bubble-radius", type=positive)
    render.add_argument("--bubble-scale-range", nargs=2, type=positive)
    render.add_argument("--bubble-relative-ior", type=positive)
    render.add_argument("--instance-seed", type=int, default=None)
    render.add_argument("--droplet-subdivisions", type=int, default=None)
    commands.add_parser("material-presets",
                        help="List named v03 material presets and exit")
    return result


def validate(args: argparse.Namespace) -> None:
    if args.command == "foam":
        bounds(args.domain_min, args.domain_max)
        if args.lifetime_max < args.lifetime_min or not 0 <= args.seed < 2 ** 64 or \
                args.generator_threads != int(args.generator_threads) or args.buoyancy < 0:
            raise ValueError("invalid whitewater lifetime, seed, threads, or buoyancy")
        axis_permutation(args.output_axis_order)
        return

    bounds(args.tank_min, args.tank_max)
    axis_permutation(args.surface_axis_order)
    axis_permutation(args.foam_axis_order)
    explicit = {key for key in (*BLADE_MATERIALS["anodized-copper"],
                                *GLASS_MATERIALS["low-iron-glass"])
                if getattr(args, key, None) is not None}
    if args.material_preset is not None:
        if args.material_preset not in MATERIAL_PRESETS:
            raise ValueError(f"unknown material preset: {args.material_preset}")
        if args.mode == "liquid" and args.material_preset == "stress":
            raise ValueError("the stress preset does not cover liquid-mode materials; "
                             "use --material-preset liquid or foam")
        if args.mode == "stress" and args.material_preset != "stress":
            raise ValueError("stress mode only uses the stress material preset")
        for key, value in MATERIAL_PRESETS[args.material_preset].items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    if args.mode == "stress" and (args.blade_material or args.glass_material):
        raise ValueError("stress mode uses vertex colors and no glass walls; "
                         "blade/glass materials are unused")
    # Explicit per-property values win; a selected component material replaces
    # the corresponding part of a scene preset without changing any other set.
    blade_material = args.blade_material or (
        "anodized-copper" if args.material_preset in ("liquid", "foam") else None)
    glass_material = args.glass_material or (
        "low-iron-glass" if args.material_preset in ("liquid", "foam") else None)
    for selection, catalog in ((blade_material, BLADE_MATERIALS),
                               (glass_material, GLASS_MATERIALS)):
        if selection is not None:
            for key, value in catalog[selection].items():
                if key not in explicit:
                    setattr(args, key, value)
    args.blade_material = blade_material
    args.glass_material = glass_material
    for key, value in MATERIAL_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    from math import isfinite
    if args.mode == "liquid":
        required = ("water_color", "water_roughness", "water_ior", "water_absorption",
                    "water_scattering", "water_scattering_anisotropy", "blade_color",
                    "blade_metallic", "blade_roughness", "blade_coat", "glass_color",
                    "glass_opacity", "glass_roughness", "glass_ior", "glass_transmission",
                    "wall_thickness",
                    "floor_thickness", "visible_wall_height")
    else:
        required = (("water_color", "water_roughness", "water_ior",
                     "stress_water_transmission", "stress_water_specular")
                    if args.stress_water_opacity else ())
        if args.foam_dir is not None:
            raise ValueError("stress rendering does not use whitewater materials")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("required render parameters: " + ", ".join(missing))
    if not all(isfinite(number) for number in
               (*args.camera_position, *args.camera_target,
                *(args.water_absorption or ()), *(args.water_scattering or ()),
                *args.stress_light_direction,
                args.exposure, args.camera_fov, args.blade_bevel_angle,
                args.world_strength, args.visible_wall_height or 0)):
        raise ValueError("nonfinite render or material parameter")
    for name in ("water_color", "blade_color", "glass_color", "world_color", "floor_color"):
        values = getattr(args, name)
        if values is not None:
            rgb(values)
    for name in ("water_absorption", "water_scattering"):
        if getattr(args, name) is not None and any(value < 0 for value in getattr(args, name)):
            raise ValueError(f"{name} cannot be negative")
    if args.width <= 0 or args.height <= 0 or args.samples <= 0 or args.stride <= 0 or \
            min(args.max_bounces, args.transmission_bounces,
                args.transparent_bounces) < 0 or not 0 <= args.png_compression <= 100 or \
            args.blade_bevel_segments < 1 or args.droplet_subdivisions < 0 or \
            (args.water_scattering_anisotropy is not None and
             not -1 <= args.water_scattering_anisotropy <= 1) or \
            args.blade_floor_extension < 0 or args.blade_bevel < 0 or \
            args.floor_bevel < 0 or args.pedestal_margin < 0 or \
            args.wall_cap_extension < 0 or not 0 < args.camera_fov < 180 or \
            not 0 <= args.blade_bevel_angle <= 180:
        raise ValueError("invalid render dimensions or material parameters")
    if args.mode == "stress" and not args.stress_mesh:
        raise ValueError("stress mode requires at least one --stress-mesh")
    if args.foam_dir is not None:
        required = ("foam_color", "foam_point_radius", "foam_voxel_size", "foam_threshold",
                    "foam_adaptivity", "foam_roughness", "foam_transmission",
                    "foam_subsurface", "foam_micro_bump",
                    "foam_noise_scale", "foam_noise_detail", "foam_noise_roughness",
                    "foam_bump_strength", "spray_color", "spray_roughness",
                    "spray_radius", "spray_scale_range",
                    "spray_exposure", "spray_max_stretch", "bubble_color", "bubble_radius",
                    "bubble_roughness", "bubble_scale_range", "bubble_relative_ior")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError("foam rendering requires: " + ", ".join(missing))
        rgb(args.foam_color)
        rgb(args.spray_color)
        rgb(args.bubble_color)
        if args.foam_micro_bump < 0:
            raise ValueError("foam micro-bump distance cannot be negative")
        for field in ("spray_scale_range", "bubble_scale_range"):
            values = getattr(args, field)
            if values[0] > values[1]:
                raise ValueError(f"{field} must be increasing")
    if args.frame is not None and (args.frame < 0 or args.resume):
        raise ValueError("single-frame selection cannot resume and must be nonnegative")
    if args.sequence and (args.start < 0 or args.stop is not None and args.stop <= args.start):
        raise ValueError("invalid sequence range")


def main(arguments: list[str] | None = None) -> None:
    values = sys.argv[1:] if arguments is None else arguments
    options = parser().parse_args(values)
    if options.command == "material-presets":
        from materials import describe_presets
        print(describe_presets(), end="")
        return
    try:
        validate(options)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if options.command == "foam":
        from foam import generate
        generate(options)
        return
    # Snap's /snap/bin/blender is a launcher symlink to /usr/bin/snap; resolving
    # it discards the application name that the launcher needs to dispatch.
    blender = options.blender.expanduser().absolute()
    if not blender.is_file():
        raise SystemExit(f"Blender executable not found: {blender}")
    worker = Path(__file__).with_name("blender_worker.py")
    args = [str(blender), "--background", "--python-exit-code", "1",
            "--python", str(worker), "--", *values]
    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
