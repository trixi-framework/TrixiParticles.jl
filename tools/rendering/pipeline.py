#!/usr/bin/env python3
"""FoamGenerator and Blender tools; all case settings are command-line parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from data import axis_permutation, bounds
from materials import MATERIAL_SELECTORS, SCENE_DEFAULTS


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
    for _, flag, choices in MATERIAL_SELECTORS:
        render.add_argument(flag, choices=tuple(choices),
                            help="Select this material independently; property flags override it")
    render.add_argument("--stress-mesh", action="append", type=Path, default=[],
                        help="Pre-colored stress PLY; repeat for each solid")
    render.add_argument("--stress-water-opacity", type=fraction)
    render.add_argument("--stress-water-transmission", type=fraction)
    render.add_argument("--stress-water-specular", type=fraction)
    render.add_argument("--stress-normal-light", type=fraction, default=0.0)
    render.add_argument("--stress-light-direction", nargs=3, type=float,
                        default=(0.35, -0.45, 0.82))

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
    render.add_argument("--water-transmission", type=fraction)
    render.add_argument("--water-specular-level", type=fraction)
    render.add_argument("--water-ior", type=positive)
    render.add_argument("--water-absorption", nargs=3, type=float)
    render.add_argument("--water-scattering", nargs=3, type=float)
    render.add_argument("--water-scattering-anisotropy", type=float)
    render.add_argument("--solid-color", nargs=3, type=float)
    render.add_argument("--solid-metallic", type=fraction)
    render.add_argument("--solid-roughness", type=fraction)
    render.add_argument("--solid-coat", type=fraction)
    render.add_argument("--solid-specular-level", type=fraction)
    render.add_argument("--solid-noise-scale", type=positive,
                        help="Optional local noise texture frequency for a solid")
    render.add_argument("--solid-bump-strength", type=fraction,
                        help="Optional procedural surface microstructure strength")
    render.add_argument("--solid-bump-distance", type=positive,
                        help="Optional microstructure bump distance in metres")
    render.add_argument("--solid-inclusion-color", nargs=3, type=float,
                        help="Optional aggregate/stone inclusion color")
    render.add_argument("--solid-inclusion-scale", type=positive,
                        help="Optional Voronoi inclusion frequency in inverse metres")
    render.add_argument("--solid-inclusion-threshold", type=fraction,
                        help="Voronoi inclusion core radius in cell units")
    render.add_argument("--solid-inclusion-transition", type=positive,
                        help="Transition from inclusion to matrix in cell units")
    render.add_argument("--solid-inclusion-warp-scale", type=positive,
                        help="Optional noise frequency that irregularizes inclusions")
    render.add_argument("--solid-inclusion-distortion", type=float,
                        help="Optional inclusion-boundary distortion in metres")
    render.add_argument("--solid-floor-extension", type=float, default=None)
    render.add_argument("--solid-bevel", type=float, default=None)
    render.add_argument("--solid-bevel-segments", type=int, default=None)
    render.add_argument("--solid-bevel-angle", type=float, default=None)
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
    commands.add_parser("materials", help="List independent material choices and exit")
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
    component_keys = {key for _, _, catalog in MATERIAL_SELECTORS
                      for component in catalog.values() for key in component}
    explicit = {key for key in component_keys
                if getattr(args, key, None) is not None}
    if args.mode == "stress" and any(getattr(args, key) is not None for key in
                                      ("liquid_material", "solid_material", "glass_material",
                                       "foam_material", "spray_material", "bubble_material")):
        raise ValueError("stress view uses a pre-colored solid and omits other materials")
    if args.mode == "liquid" and args.ghost_material is not None:
        raise ValueError("ghost material is only used in the stress view")
    if args.foam_dir is None and any(getattr(args, key) is not None for key in
                                     ("foam_material", "spray_material", "bubble_material")):
        raise ValueError("whitewater materials need --foam-dir")
    for selector, _, catalog in MATERIAL_SELECTORS:
        choice = getattr(args, selector)
        if choice is not None:
            for key, value in catalog[choice].items():
                if key not in explicit:
                    setattr(args, key, value)
    for key, value in SCENE_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.solid_bump_strength is not None and args.solid_bump_strength > 0:
        if args.solid_noise_scale is None or args.solid_bump_distance is None:
            raise ValueError("solid microstructure needs scale and bump distance")
    elif args.solid_noise_scale is not None or args.solid_bump_distance is not None:
        if args.solid_bump_strength is None:
            raise ValueError("solid microstructure needs bump strength")
    inclusions = ("solid_inclusion_scale", "solid_inclusion_threshold",
                  "solid_inclusion_transition", "solid_inclusion_warp_scale",
                  "solid_inclusion_distortion")
    if args.solid_inclusion_color is not None:
        missing = [name for name in inclusions if getattr(args, name) is None]
        if missing:
            raise ValueError("solid inclusions need: " + ", ".join(missing))
        if args.solid_inclusion_threshold + args.solid_inclusion_transition >= 1:
            raise ValueError("solid inclusion transition must end before 1")
        if args.solid_inclusion_distortion < 0:
            raise ValueError("solid inclusion distortion cannot be negative")
    elif any(getattr(args, name) is not None for name in inclusions):
        raise ValueError("solid inclusions need --solid-inclusion-color")
    if args.stress_water_opacity is None:
        args.stress_water_opacity = 0.0
    from math import isfinite
    if args.mode == "liquid":
        required = ("water_color", "water_roughness", "water_transmission", "water_ior",
                    "water_absorption",
                    "water_scattering", "water_scattering_anisotropy", "glass_color",
                    "glass_opacity", "glass_roughness", "glass_ior", "glass_transmission",
                    "glass_specular", "wall_thickness", "floor_thickness",
                    "visible_wall_height")
    else:
        required = (("water_color", "water_roughness", "water_ior",
                     "stress_water_transmission", "stress_water_specular")
                    if args.stress_water_opacity else ())
        if args.foam_dir is not None:
            raise ValueError("stress rendering does not use whitewater materials")
    required += ("floor_color", "floor_metallic", "floor_roughness", "floor_coat")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("required render parameters: " + ", ".join(missing))
    if not all(isfinite(number) for number in
               (*args.camera_position, *args.camera_target,
                *(args.water_absorption or ()), *(args.water_scattering or ()),
                *args.stress_light_direction,
                args.exposure, args.camera_fov, args.solid_bevel_angle,
                args.world_strength, args.visible_wall_height or 0)):
        raise ValueError("nonfinite render or material parameter")
    for name in ("water_color", "solid_color", "glass_color", "world_color",
                 "floor_color", "solid_inclusion_color"):
        values = getattr(args, name)
        if values is not None:
            rgb(values)
    for name in ("water_absorption", "water_scattering"):
        if getattr(args, name) is not None and any(value < 0 for value in getattr(args, name)):
            raise ValueError(f"{name} cannot be negative")
    if args.width <= 0 or args.height <= 0 or args.samples <= 0 or args.stride <= 0 or \
            min(args.max_bounces, args.transmission_bounces,
                args.transparent_bounces) < 0 or not 0 <= args.png_compression <= 100 or \
            args.solid_bevel_segments < 1 or args.droplet_subdivisions < 0 or \
            (args.water_scattering_anisotropy is not None and
             not -1 <= args.water_scattering_anisotropy <= 1) or \
            args.solid_floor_extension < 0 or args.solid_bevel < 0 or \
            args.floor_bevel < 0 or args.pedestal_margin < 0 or \
            args.wall_cap_extension < 0 or not 0 < args.camera_fov < 180 or \
            not 0 <= args.solid_bevel_angle <= 180:
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
    if options.command == "materials":
        from materials import describe_materials
        print(describe_materials(), end="")
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
