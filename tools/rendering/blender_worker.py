#!/usr/bin/env blender --background --python
"""Blender worker for package surfaces and optional SPlisHSPlasH whitewater."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from mathutils import Vector
import numpy as np

from data import file_sha256, iteration_token, png_dimensions, read_pvd, write_json
from pipeline import parser, validate


def selected_sources(args):
    if args.surface_pvd is not None:
        frames = [{"source_timestep": item.index, "simulation_time_s": item.time,
                   "water_path": item.path.with_suffix(".ply"),
                   "solids": [args.surface_dir /
                              pattern.format(frame=item.index,
                                             iter=iteration_token(item.path.name, item.index),
                                             time=item.time)
                              for pattern in args.solid_pattern]}
                  for item in read_pvd(args.surface_pvd)]
        metadata_path = args.surface_pvd
    else:
        metadata_path = args.surface_metadata
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source.get("status") != "complete" or not source.get("frames"):
            raise ValueError("surface inventory is incomplete")
        frames = [{"source_timestep": int(item["source_timestep"]),
                   "simulation_time_s": float(item["simulation_time_s"]),
                   "water_path": args.surface_dir / item["water_file"],
                   "solids": [args.surface_dir / filename
                              for filename in item.get("solid_files", item.get("blade_files", []))] +
                   [args.surface_dir / pattern.format(frame=item["source_timestep"],
                                                      time=item["simulation_time_s"])
                    for pattern in args.solid_pattern]}
                  for item in source["frames"]]
    indices = [record["source_timestep"] for record in frames]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("surface source indices must be unique and increasing")
    selected = ([item for item in frames if item["source_timestep"] == args.frame]
                if args.frame is not None else
                [item for item in frames if item["source_timestep"] >= args.start and
                 (args.stop is None or item["source_timestep"] < args.stop) and
                 (item["source_timestep"] - args.start) % args.stride == 0])
    if not selected:
        raise ValueError("surface selection is empty")
    return selected, metadata_path


def foam_frames(args):
    if args.foam_dir is None:
        return {}, None
    path = args.foam_dir / "foam_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("coordinate_space") != \
            f"source axes {args.foam_axis_order}":
        raise ValueError("whitewater cache is incomplete or uses another coordinate order")
    if metadata.get("frame_count") != len(metadata.get("frames", [])):
        raise ValueError("whitewater cache has an incomplete frame inventory")
    frames = {int(item["source_timestep"]): item for item in metadata["frames"]}
    if len(frames) != metadata["frame_count"]:
        raise ValueError("duplicate whitewater source timestep")
    return frames, path


def verified_inputs(args, selected, whitewater):
    """Fingerprint the exact selected meshes and particles before rendering."""
    output = []
    for frame in selected:
        paths = [frame["water_path"], *frame["solids"]]
        if args.mode == "stress":
            paths.extend(args.stress_mesh)
        if args.foam_dir is not None:
            record = whitewater.get(frame["source_timestep"])
            if record is None or not math.isclose(record["simulation_time_s"],
                                                  frame["simulation_time_s"], abs_tol=1e-9):
                raise ValueError("surface and whitewater source times differ")
            if sum(record["counts"][kind] for kind in ("foam", "spray", "bubbles")) != \
                    record["counts"]["total"]:
                raise ValueError("whitewater category counts differ from the total")
            for item in record["outputs"]:
                path = args.foam_dir / item["file"]
                if item["type"] not in ("foam", "spray", "bubbles") or \
                        not path.is_file() or path.stat().st_size != item["bytes"] or \
                        file_sha256(path) != item["sha256"]:
                    raise ValueError(f"whitewater output failed validation: {path}")
                paths.append(path)
        checksums = {}
        for path in paths:
            if not path.is_file():
                raise ValueError(f"missing render input: {path}")
            checksums[str(path.resolve())] = file_sha256(path)
        output.append(checksums)
    return output


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights, bpy.data.node_groups):
        for datablock in list(datablocks):
            datablocks.remove(datablock)


def principled(name, color, roughness, *, transmission=0, metallic=0, ior=1.45, coat=0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Transmission Weight"].default_value = transmission
    node.inputs["Metallic"].default_value = metallic
    node.inputs["IOR"].default_value = ior
    node.inputs["Coat Weight"].default_value = coat
    return material


def water_material(args):
    label = (args.liquid_material or "custom liquid").replace("-", " ").title()
    material = principled(label, args.water_color, args.water_roughness,
                          transmission=1, ior=args.water_ior)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    volume = nodes.new("ShaderNodeVolumeCoefficients")
    volume.inputs["Absorption Coefficients"].default_value = tuple(args.water_absorption)
    volume.inputs["Scatter Coefficients"].default_value = tuple(args.water_scattering)
    volume.inputs["Anisotropy"].default_value = args.water_scattering_anisotropy
    volume.phase = "HENYEY_GREENSTEIN"
    links.new(volume.outputs["Volume"], nodes.get("Material Output").inputs["Volume"])
    return material


def froth_material(args):
    material = principled("Whitewater froth", args.foam_color, args.foam_roughness,
                          transmission=args.foam_transmission, ior=args.water_ior)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Subsurface Weight"].default_value = args.foam_subsurface
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = args.foam_noise_scale
    noise.inputs["Detail"].default_value = args.foam_noise_detail
    noise.inputs["Roughness"].default_value = args.foam_noise_roughness
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = args.foam_micro_bump
    bump.inputs["Strength"].default_value = args.foam_bump_strength
    links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def bubble_material(args):
    material = bpy.data.materials.new("Submerged bubble interfaces")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (*args.bubble_color, 1)
    glass.inputs["IOR"].default_value = args.bubble_relative_ior
    glass.inputs["Roughness"].default_value = args.bubble_roughness
    material.node_tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def stress_material(args, color_attribute):
    material = bpy.data.materials.new("Stress vertex color")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = color_attribute
    normal = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = args.stress_light_direction
    nonnegative = nodes.new("ShaderNodeMath")
    nonnegative.operation = "MAXIMUM"
    nonnegative.inputs[1].default_value = 0
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = args.stress_normal_light
    base = nodes.new("ShaderNodeMath")
    base.operation = "ADD"
    base.inputs[1].default_value = 1 - args.stress_normal_light
    tint = nodes.new("ShaderNodeMixRGB")
    tint.blend_type = "MULTIPLY"
    tint.inputs[0].default_value = 1
    emission = nodes.new("ShaderNodeEmission")
    links = material.node_tree.links
    links.new(normal.outputs["Normal"], dot.inputs[0])
    links.new(dot.outputs["Value"], nonnegative.inputs[0])
    links.new(nonnegative.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], base.inputs[0])
    links.new(color.outputs["Color"], tint.inputs[1])
    links.new(base.outputs[0], tint.inputs[2])
    links.new(tint.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs["Surface"])
    return material


def import_mesh(path: Path, name: str, material, source_axis: str, *, smooth=True):
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(path.resolve()))
    imported = list(set(bpy.data.objects) - before)
    if len(imported) != 1:
        raise ValueError(f"expected one PLY object from {path}")
    obj = imported[0]
    obj.name = name
    if material is not None:
        obj.data.materials.append(material)
    if source_axis == "xyz":
        for vertex in obj.data.vertices:
            vertex.co.y, vertex.co.z = vertex.co.z, vertex.co.y
        obj.data.flip_normals()  # Swapping axes reflects the mesh.
        obj.data.update()
    if smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


def particles_ply(path, name, material, source_axis):
    with path.open("rb") as source:
        if source.readline() != b"ply\n" or source.readline() != \
                b"format binary_little_endian 1.0\n":
            raise ValueError("whitewater PLY must be binary little-endian")
        properties, count, reading_vertices = [], None, False
        while True:
            raw = source.readline()
            if not raw:
                raise ValueError("unterminated whitewater PLY header")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            words = line.split()
            if words[:2] == ["element", "vertex"]:
                count, reading_vertices = int(words[2]), True
            elif words[:1] == ["element"]:
                reading_vertices = False
            elif reading_vertices and words[:2] == ["property", "float"]:
                properties.append(words[2])
        payload = source.read()
    if properties != ["x", "y", "z", "velocity_x", "velocity_y", "velocity_z"]:
        raise ValueError("whitewater PLY has unexpected attributes")
    data = np.frombuffer(payload, dtype="<f4")
    if count is None or data.size != count * 6 or not np.isfinite(data).all():
        raise ValueError("whitewater PLY has invalid binary content")
    data = data.reshape(count, 6)
    if source_axis == "xyz":
        data = data[:, (0, 2, 1, 3, 5, 4)]
    mesh = bpy.data.meshes.new(f"{name} points")
    mesh.vertices.add(count)
    mesh.vertices.foreach_set("co", np.ascontiguousarray(data[:, :3]).ravel())
    velocity = mesh.attributes.new(name="velocity", type="FLOAT_VECTOR", domain="POINT")
    velocity.data.foreach_set("vector", np.ascontiguousarray(data[:, 3:]).ravel())
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def froth_nodes(obj, material, args):
    modifier = obj.modifiers.new("Whitewater surface", "NODES")
    group = bpy.data.node_groups.new("Whitewater froth reconstruction", "GeometryNodeTree")
    modifier.node_group = group
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    source = nodes.new("NodeGroupInput")
    target = nodes.new("NodeGroupOutput")
    markers = nodes.new("GeometryNodeMeshToPoints")
    markers.mode = "VERTICES"
    volume = nodes.new("GeometryNodePointsToVolume")
    volume.inputs["Resolution Mode"].default_value = "Size"
    volume.inputs["Voxel Size"].default_value = args.foam_voxel_size
    volume.inputs["Density"].default_value = 1
    volume.inputs["Radius"].default_value = args.foam_point_radius
    surface = nodes.new("GeometryNodeVolumeToMesh")
    surface.inputs["Resolution Mode"].default_value = "Grid"
    surface.inputs["Threshold"].default_value = args.foam_threshold
    surface.inputs["Adaptivity"].default_value = args.foam_adaptivity
    smoothing = nodes.new("GeometryNodeSetShadeSmooth")
    smoothing.inputs["Shade Smooth"].default_value = True
    assigned = nodes.new("GeometryNodeSetMaterial")
    assigned.inputs["Material"].default_value = material
    for out, inp in ((source.outputs["Geometry"], markers.inputs["Mesh"]),
                     (markers.outputs["Points"], volume.inputs["Points"]),
                     (volume.outputs["Volume"], surface.inputs["Volume"]),
                     (surface.outputs["Mesh"], smoothing.inputs["Mesh"]),
                     (smoothing.outputs["Mesh"], assigned.inputs["Geometry"]),
                     (assigned.outputs["Geometry"], target.inputs["Geometry"])):
        links.new(out, inp)


def instances(obj, material, kind, args):
    modifier = obj.modifiers.new(f"{kind} instances", "NODES")
    group = bpy.data.node_groups.new(f"{kind} droplets", "GeometryNodeTree")
    modifier.node_group = group
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    source, target = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    points = nodes.new("GeometryNodeMeshToPoints")
    points.mode = "VERTICES"
    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Radius"].default_value = (args.spray_radius if kind == "spray" else
                                               args.bubble_radius)
    sphere.inputs["Subdivisions"].default_value = args.droplet_subdivisions
    smoothing = nodes.new("GeometryNodeSetShadeSmooth")
    smoothing.inputs["Shade Smooth"].default_value = True
    colored = nodes.new("GeometryNodeSetMaterial")
    colored.inputs["Material"].default_value = material
    index = nodes.new("GeometryNodeInputIndex")
    random = nodes.new("FunctionNodeRandomValue")
    random.data_type = "FLOAT"
    random.inputs["Seed"].default_value = args.instance_seed + (1 if kind == "spray" else 2)
    lower, upper = (args.spray_scale_range if kind == "spray" else args.bubble_scale_range)
    random.inputs["Min"].default_value = lower
    random.inputs["Max"].default_value = upper
    scale = nodes.new("ShaderNodeCombineXYZ")
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    for out, inp in ((source.outputs["Geometry"], points.inputs["Mesh"]),
                     (points.outputs["Points"], instance.inputs["Points"]),
                     (sphere.outputs["Mesh"], smoothing.inputs["Mesh"]),
                     (smoothing.outputs["Mesh"], colored.inputs["Geometry"]),
                     (colored.outputs["Geometry"], instance.inputs["Instance"]),
                     (index.outputs["Index"], random.inputs["ID"]),
                     (random.outputs["Value"], scale.inputs["X"]),
                     (random.outputs["Value"], scale.inputs["Y"])):
        links.new(out, inp)
    if kind == "spray":
        velocity = nodes.new("GeometryNodeInputNamedAttribute")
        velocity.data_type = "FLOAT_VECTOR"
        velocity.inputs["Name"].default_value = "velocity"
        direction = nodes.new("FunctionNodeAlignRotationToVector")
        direction.axis = "Z"
        length = nodes.new("ShaderNodeVectorMath")
        length.operation = "LENGTH"
        exposure = nodes.new("ShaderNodeMath")
        exposure.operation = "MULTIPLY"
        exposure.inputs[1].default_value = args.spray_exposure / (2 * args.spray_radius)
        stretch = nodes.new("ShaderNodeMath")
        stretch.operation = "ADD"
        stretch.inputs[1].default_value = 1
        cap = nodes.new("ShaderNodeMath")
        cap.operation = "MINIMUM"
        cap.inputs[1].default_value = args.spray_max_stretch
        scaled = nodes.new("ShaderNodeMath")
        scaled.operation = "MULTIPLY"
        for out, inp in ((velocity.outputs["Attribute"], direction.inputs["Vector"]),
                         (velocity.outputs["Attribute"], length.inputs[0]),
                         (length.outputs["Value"], exposure.inputs[0]),
                         (exposure.outputs[0], stretch.inputs[0]),
                         (stretch.outputs[0], cap.inputs[0]),
                         (random.outputs["Value"], scaled.inputs[0]),
                         (cap.outputs[0], scaled.inputs[1]),
                         (scaled.outputs[0], scale.inputs["Z"]),
                         (direction.outputs["Rotation"], instance.inputs["Rotation"])):
            links.new(out, inp)
    else:
        links.new(random.outputs["Value"], scale.inputs["Z"])
    links.new(scale.outputs["Vector"], instance.inputs["Scale"])
    links.new(instance.outputs["Instances"], target.inputs["Geometry"])


def box(name, location, dimensions, material, bevel=0):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = tuple(value / 2 for value in dimensions)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel:
        modifier = obj.modifiers.new("Soft edge", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def tank_and_lighting(args):
    xmin, ymin, zmin = args.tank_min
    xmax, ymax, zmax = args.tank_max
    x, depth, height = xmax - xmin, zmax - zmin, ymax - ymin
    thickness = args.wall_thickness or 0
    floor = principled("Tank base", args.floor_color, args.floor_roughness,
                       metallic=args.floor_metallic, coat=args.floor_coat)
    box("Tank pedestal", ((xmin + xmax) / 2, (zmin + zmax) / 2,
                          ymin - args.pedestal_thickness / 2 - thickness),
        (x + 2 * args.pedestal_margin, depth + 2 * args.pedestal_margin,
         args.pedestal_thickness), floor, args.floor_bevel)
    if args.mode == "stress":
        # The reviewed stress close-up omits the tank walls.
        add_lighting(args)
        return
    glass_name = "Low-iron glass" if args.glass_material == "low-iron-glass" else "Tank glass"
    glass = principled(glass_name, args.glass_color, args.glass_roughness,
                       transmission=args.glass_transmission, ior=args.glass_ior)
    nodes, links = glass.node_tree.nodes, glass.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = args.glass_specular
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = args.glass_opacity
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(bsdf.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get("Material Output").inputs["Surface"])
    box("Glass floor", ((xmin + xmax) / 2, (zmin + zmax) / 2,
                        ymin - args.floor_thickness / 2),
        (x + 2 * thickness, depth + 2 * thickness, args.floor_thickness), glass)
    full_height = height + args.wall_cap_extension
    box("Upstream wall", (xmin - thickness / 2, (zmin + zmax) / 2,
                          ymin + full_height / 2),
        (thickness, depth + 2 * thickness, full_height), glass)
    box("Far wall", ((xmin + xmax) / 2, zmin - thickness / 2,
                     ymin + full_height / 2),
        (x + 2 * thickness, thickness, full_height), glass)
    visible = args.visible_wall_height
    box("Downstream wall", (xmax + thickness / 2, (zmin + zmax) / 2, ymin + visible / 2),
        (thickness, depth + 2 * thickness, visible), glass)
    box("Near wall", ((xmin + xmax) / 2, zmax + thickness / 2, ymin + visible / 2),
        (x + 2 * thickness, thickness, visible), glass)
    add_lighting(args)


def add_lighting(args):
    for name, position, energy, color, size in args.light:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.color, data.shape, data.size = energy, color, "DISK", size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (Vector(args.camera_target) - obj.location).to_track_quat(
            "-Z", "Y").to_euler()


def configure_render(args, target):
    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = not args.no_denoise
        scene.cycles.max_bounces = args.max_bounces
        scene.cycles.transmission_bounces = args.transmission_bounces
        scene.cycles.transparent_max_bounces = args.transparent_bounces
        if args.device == "gpu":
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = args.gpu_backend
            preferences.get_devices()
            devices = [device for device in preferences.devices if device.type == args.gpu_backend]
            if not devices:
                raise ValueError(f"no Blender {args.gpu_backend} device is available")
            for device in preferences.devices:
                device.use = device in devices
        scene.cycles.device = "GPU" if args.device == "gpu" else "CPU"
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = args.samples
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = args.png_compression
    scene.render.filepath = str(target.resolve())
    scene.view_settings.view_transform = args.view_transform
    scene.view_settings.look = args.look
    scene.view_settings.exposure = args.exposure
    scene.view_settings.gamma = args.gamma
    world = scene.world or bpy.data.worlds.new("Render world")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*args.world_color, 1)
    background.inputs["Strength"].default_value = args.world_strength
    camera = bpy.data.cameras.new("Render camera")
    camera.sensor_fit = "VERTICAL"
    camera.lens = camera.sensor_height / (2 * math.tan(math.radians(args.camera_fov) / 2))
    camera_object = bpy.data.objects.new("Render camera", camera)
    bpy.context.collection.objects.link(camera_object)
    camera_object.location = args.camera_position
    camera_object.rotation_euler = (Vector(args.camera_target) - camera_object.location).to_track_quat(
        "-Z", "Y").to_euler()
    scene.camera = camera_object


def build_scene(args, frame, whitewater):
    if args.mode == "liquid":
        import_mesh(frame["water_path"], "Water", water_material(args), args.surface_axis_order)
        if args.foam_dir is not None:
            entries = {item["type"]: item for item in whitewater["outputs"]}
            for kind in ("foam", "spray", "bubbles"):
                if kind not in entries:
                    continue
                material = (froth_material(args) if kind == "foam" else
                            principled("Water spray", args.spray_color, args.spray_roughness,
                                       transmission=1, ior=args.water_ior)
                            if kind == "spray" else bubble_material(args))
                obj = particles_ply(args.foam_dir / entries[kind]["file"], kind, material,
                                    args.foam_axis_order)
                if kind == "foam":
                    froth_nodes(obj, material, args)
                else:
                    instances(obj, material, kind, args)
        if frame["solids"]:
            required = ("solid_color", "solid_metallic", "solid_roughness", "solid_coat")
            missing = [name for name in required if getattr(args, name) is None]
            if missing:
                raise ValueError("solid meshes need a --solid-material or explicit values: " +
                                 ", ".join(missing))
            solid_name = ((args.solid_material or "custom").replace("-", " ").title() +
                          " solids")
            solid_shader = principled(solid_name, args.solid_color, args.solid_roughness,
                                      metallic=args.solid_metallic, coat=args.solid_coat)
            if args.solid_bump_strength:
                nodes = solid_shader.node_tree.nodes
                links = solid_shader.node_tree.links
                coordinates = nodes.new("ShaderNodeTexCoord")
                noise = nodes.new("ShaderNodeTexNoise")
                noise.inputs["Scale"].default_value = args.solid_noise_scale
                bump = nodes.new("ShaderNodeBump")
                bump.inputs["Strength"].default_value = args.solid_bump_strength
                bump.inputs["Distance"].default_value = args.solid_bump_distance
                links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
                links.new(noise.outputs["Fac"], bump.inputs["Height"])
                links.new(bump.outputs["Normal"],
                          nodes.get("Principled BSDF").inputs["Normal"])
            for index, path in enumerate(frame["solids"], 1):
                obj = import_mesh(path, f"Structural solid {index}", solid_shader,
                                  args.surface_axis_order)
                if args.solid_floor_extension:
                    ground = min(vertex.co.z for vertex in obj.data.vertices)
                    for vertex in obj.data.vertices:
                        if vertex.co.z <= ground + 1e-6:
                            vertex.co.z -= args.solid_floor_extension
                    obj.data.update()
                if args.solid_bevel:
                    bevel = obj.modifiers.new("Edge highlight", "BEVEL")
                    bevel.width, bevel.segments = args.solid_bevel, args.solid_bevel_segments
                    bevel.limit_method = "ANGLE"
                    bevel.angle_limit = math.radians(args.solid_bevel_angle)
    else:
        stress_materials = {}
        for index, mesh_path in enumerate(args.stress_mesh, 1):
            stress = import_mesh(mesh_path, f"Stress-colored solid {index}", None,
                                 args.surface_axis_order)
            stress.data.materials.clear()
            colors = list(stress.data.color_attributes)
            if not colors:
                raise ValueError(f"stress surface needs a vertex color attribute: {mesh_path}")
            attribute = colors[0].name
            if attribute not in stress_materials:
                stress_materials[attribute] = stress_material(args, attribute)
            stress.data.materials.append(stress_materials[attribute])
            if args.solid_floor_extension:
                ground = min(vertex.co.z for vertex in stress.data.vertices)
                for vertex in stress.data.vertices:
                    if vertex.co.z <= ground + 1e-6:
                        vertex.co.z -= args.solid_floor_extension
                stress.data.update()
            if args.solid_bevel:
                bevel = stress.modifiers.new("Stress surface edge", "BEVEL")
                bevel.width, bevel.segments = args.solid_bevel, args.solid_bevel_segments
                bevel.limit_method = "ANGLE"
                bevel.angle_limit = math.radians(args.solid_bevel_angle)
        if args.stress_water_opacity:
            ghost_material = principled("Ghost fluid", args.water_color,
                                        args.water_roughness,
                                        transmission=args.stress_water_transmission,
                                        ior=args.water_ior)
            nodes = ghost_material.node_tree.nodes
            links = ghost_material.node_tree.links
            ghost_bsdf = nodes.get("Principled BSDF")
            if "Specular IOR Level" in ghost_bsdf.inputs:
                ghost_bsdf.inputs["Specular IOR Level"].default_value = \
                    args.stress_water_specular
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            mix = nodes.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = args.stress_water_opacity
            links.new(transparent.outputs[0], mix.inputs[1])
            links.new(ghost_bsdf.outputs[0], mix.inputs[2])
            links.new(mix.outputs[0], nodes.get("Material Output").inputs["Surface"])
            ghost = import_mesh(frame["water_path"], "Ghost water",
                                ghost_material, args.surface_axis_order)
            ghost.visible_shadow = False
    tank_and_lighting(args)


def render_one(args, frame, whitewater, output):
    clear_scene()
    configure_render(args, output)
    build_scene(args, frame, whitewater)
    bpy.ops.render.render(write_still=True)
    if png_dimensions(output) != (args.width, args.height):
        raise RuntimeError(f"rendered PNG is incomplete or has wrong dimensions: {output}")
    return {"source_timestep": frame["source_timestep"],
            "simulation_time_s": frame["simulation_time_s"],
            "file": output.name, "bytes": output.stat().st_size,
            "sha256": file_sha256(output)}


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main():
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser().parse_args(values)
    if args.command != "render":
        raise ValueError("Blender worker needs the render subcommand")
    validate(args)
    selected, surface_metadata = selected_sources(args)
    whitewater, foam_metadata = foam_frames(args)
    inputs = verified_inputs(args, selected, whitewater)
    settings = {key: jsonable(value)
                for key, value in vars(args).items() if key not in ("output", "resume")}
    provenance = {"settings": settings, "input_sha256": inputs,
                  "surface_metadata_sha256": file_sha256(surface_metadata),
                  "foam_metadata_sha256": file_sha256(foam_metadata) if foam_metadata else None,
                  "renderer_sha256": file_sha256(Path(__file__)),
                  "blender": bpy.app.version_string}
    fingerprint = sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    if not args.sequence:
        if output.suffix.lower() != ".png" or output.exists():
            raise ValueError("single-frame output must be a new PNG file")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = render_one(args, selected[0],
                            whitewater.get(selected[0]["source_timestep"]), output)
        write_json(output.with_suffix(".png.json"),
                   {"status": "complete", "fingerprint": fingerprint,
                    "provenance": provenance, "frame": result})
        return

    progress_path = output / "render_progress.json"
    if args.resume:
        if not progress_path.is_file():
            raise ValueError("resume requires recorded render progress")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint or progress.get("status") != "in_progress":
            raise ValueError("resume input, settings or renderer do not match")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"render output must be new or empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        progress = {"status": "in_progress", "fingerprint": fingerprint,
                    "completed": [], "provenance": provenance}
        write_json(progress_path, progress)

    completed = progress["completed"]
    if len(completed) > len(selected):
        raise ValueError("existing sequence is longer than the current selection")
    for index, item in enumerate(completed):
        path = output / f"frame_{index:06d}.png"
        if item["file"] != path.name or item["source_timestep"] != \
                selected[index]["source_timestep"] or png_dimensions(path) != \
                (args.width, args.height) or file_sha256(path) != item["sha256"]:
            raise ValueError("existing render frame does not match progress metadata")
    existing = sorted(output.glob("frame_*.png"))
    if len(existing) > len(completed):
        if len(existing) != len(completed) + 1 or existing[-1].name != \
                f"frame_{len(completed):06d}.png":
            raise ValueError("unexpected loose frame during resume")
        orphan = existing[-1]
        if png_dimensions(orphan) != (args.width, args.height):
            raise ValueError("incomplete unrecorded frame")
        frame = selected[len(completed)]
        completed.append({"source_timestep": frame["source_timestep"],
                          "simulation_time_s": frame["simulation_time_s"],
                          "file": orphan.name, "bytes": orphan.stat().st_size,
                          "sha256": file_sha256(orphan)})
        write_json(progress_path, progress)
    for index in range(len(completed), len(selected)):
        frame = selected[index]
        item = render_one(args, frame, whitewater.get(frame["source_timestep"]),
                          output / f"frame_{index:06d}.png")
        completed.append(item)
        write_json(progress_path, progress)
        print(f"Rendered source timestep {frame['source_timestep']} ({index + 1}/{len(selected)})")
    record = {"status": "complete", "generated_at": datetime.now(timezone.utc).isoformat(),
              "fingerprint": fingerprint, "provenance": provenance,
              "frame_count": len(completed), "frames": completed}
    write_json(output / "render_metadata.json", record)
    progress["status"] = "complete"
    write_json(progress_path, progress)


if __name__ == "__main__":
    main()
