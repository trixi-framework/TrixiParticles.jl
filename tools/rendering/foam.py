"""Sequential SPlisHSPlasH foam, spray and bubble postprocessing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from data import axis_permutation, effective_drag, file_sha256, read_pvd, uniform_step, write_json

PARTICLE_TYPES = ("foam", "spray", "bubbles")


def write_partio(path: Path, destination: Path, *, velocity_array: str):
    # FoamGenerator uses a Partio fluid sequence. Delay these imports so that
    # render-only users do not need the FoamGenerator Python environment.
    import meshio
    import numpy as np
    import partio

    cloud = meshio.read(path)
    points = np.asarray(cloud.points, dtype="<f4")
    velocity = np.asarray(cloud.point_data.get(velocity_array), dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3 or velocity.shape != points.shape:
        raise ValueError(f"{path} needs matching 3D position and {velocity_array} arrays")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(velocity)):
        raise ValueError(f"nonfinite fluid points or velocities: {path}")
    particles = partio.create()
    position = particles.addAttribute("position", partio.VECTOR, 3)
    speed = particles.addAttribute("velocity", partio.VECTOR, 3)
    particles.addParticles(len(points))
    for index in range(len(points)):
        particles.set(position, index, tuple(map(float, points[index])))
        particles.set(speed, index, tuple(map(float, velocity[index])))
    partio.write(str(destination), particles)
    if not destination.is_file():
        raise RuntimeError(f"Partio did not write {destination}")
    return len(points)


def read_secondary(path: Path):
    import meshio
    import numpy as np

    if not path.is_file():
        return np.empty((0, 3), dtype="<f4"), np.empty((0, 3), dtype="<f4")
    cloud = meshio.read(path)
    points = np.asarray(cloud.points, dtype="<f4")
    velocity = np.asarray(cloud.point_data.get("velocity"), dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3 or velocity.shape != points.shape or \
            not np.all(np.isfinite(points)) or not np.all(np.isfinite(velocity)):
        raise ValueError(f"invalid secondary-particle VTK data: {path}")
    return points, velocity


def write_ply(path: Path, points, velocity, order: str) -> None:
    import numpy as np

    permutation = axis_permutation(order)
    values = np.column_stack((points[:, permutation], velocity[:, permutation]))
    values = np.asarray(values, dtype="<f4", order="C")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment coordinates are source axes {order}; velocity in metres per second\n"
        f"element vertex {len(values)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float velocity_x\nproperty float velocity_y\n"
        "property float velocity_z\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as output:
        output.write(header)
        output.write(values.tobytes(order="C"))


def package_frame(source: Path, destination: Path, frame_index: int, time: float,
                  lower, upper, order: str):
    import numpy as np

    minimum = np.asarray(lower, dtype="<f4")
    maximum = np.asarray(upper, dtype="<f4")
    records = []
    raw_counts = {}
    clipped_counts = {}
    for kind in PARTICLE_TYPES:
        points, velocity = read_secondary(source / f"secondary_{frame_index:04d}_{kind}.vtk")
        raw_counts[kind] = len(points)
        inside = np.all((points >= minimum) & (points <= maximum), axis=1)
        points, velocity = points[inside], velocity[inside]
        clipped_counts[kind] = raw_counts[kind] - len(points)
        if not len(points):
            continue
        name = f"frame_{frame_index:06d}_{kind}.ply"
        path = destination / name
        write_ply(path, points, velocity, order)
        records.append({"type": kind, "file": name, "count": len(points),
                        "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    counts = {item["type"]: item["count"] for item in records}
    return {"source_timestep": frame_index, "simulation_time_s": time,
            "counts": {**{kind: counts.get(kind, 0) for kind in PARTICLE_TYPES},
                       "total": sum(counts.values())},
            "raw_counts": raw_counts, "clipped_counts": clipped_counts,
            "outputs": records}


def fingerprint(options, frames, step, binary_hash):
    from hashlib import sha256

    inputs = {
        "collection_sha256": file_sha256(options.fluid_pvd),
        "foam_generator_sha256": binary_hash,
        "frame_times": [frame.time for frame in frames],
        "frame_paths": [str(frame.path) for frame in frames],
        "frame_sha256": [file_sha256(frame.path) for frame in frames],
        "domain_min": options.domain_min, "domain_max": options.domain_max,
        "generator_radius": options.generator_radius, "foam_scale": options.foam_scale,
        "lifetime": [options.lifetime_min, options.lifetime_max],
        "buoyancy": options.buoyancy, "drag": options.drag,
        "drag_reference_step": options.drag_reference_step,
        "effective_drag": effective_drag(options.drag, step, options.drag_reference_step),
        "seed": options.seed, "generator_threads": int(options.generator_threads),
        "velocity_array": options.velocity_array, "output_axis_order": options.output_axis_order,
    }
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest(), inputs


def validate_cache(output: Path, key: str, frames) -> None:
    metadata_path = output / "foam_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("only a complete whitewater cache can be reused")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("fingerprint") != key or \
            len(metadata.get("frames", [])) != len(frames):
        raise ValueError("whitewater cache provenance or frame count differs")
    log = output / "foam_generator.log"
    if not log.is_file() or file_sha256(log) != metadata["generator"]["log_sha256"]:
        raise ValueError("whitewater generator log changed")
    expected_files = {metadata_path.name, log.name}
    for frame, record in zip(frames, metadata["frames"]):
        if record["source_timestep"] != frame.index or \
                not math.isclose(record["simulation_time_s"], frame.time, abs_tol=1e-12):
            raise ValueError("whitewater cache source mapping differs")
        for item in record["outputs"]:
            path = output / item["file"]
            expected_files.add(path.name)
            if not path.is_file() or path.stat().st_size != item["bytes"] or \
                    file_sha256(path) != item["sha256"]:
                raise ValueError(f"whitewater cache file changed: {path}")
    if {path.name for path in output.iterdir()} != expected_files:
        raise ValueError("whitewater cache contains unrecorded files")


def generate(options) -> None:
    frames = read_pvd(options.fluid_pvd)
    step = uniform_step(frames)
    binary = options.generator.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"FoamGenerator executable not found: {binary}")
    help_result = subprocess.run([str(binary), "--help"], text=True, capture_output=True,
                                 check=True)
    if "--seed" not in help_result.stdout:
        raise ValueError("FoamGenerator must support the explicit --seed build patch")
    binary_hash = file_sha256(binary)
    key, source_settings = fingerprint(options, frames, step, binary_hash)
    output = options.output.resolve()
    if options.reuse:
        validate_cache(output, key, frames)
        print(f"Validated existing whitewater cache: {output}")
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"whitewater output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trixiparticles_foam_") as temporary:
        temp = Path(temporary)
        partio_dir, raw = temp / "partio", temp / "raw"
        partio_dir.mkdir()
        raw.mkdir()
        counts = []
        for frame in frames:
            count = write_partio(frame.path, partio_dir / f"fluid_{frame.index:04d}.bgeo",
                                 velocity_array=options.velocity_array)
            counts.append(count)
        if len(set(counts)) != 1:
            raise ValueError("the source fluid particle count changed between frames")
        drag = effective_drag(options.drag, step, options.drag_reference_step)
        command = [str(binary), "--splittypes", "--startframe", "0", "--endframe",
                   str(len(frames) - 1), "--radius", repr(options.generator_radius),
                   "--timestepsize", repr(step), "--foamscale", repr(options.foam_scale),
                   "--lifetime", f"{options.lifetime_min},{options.lifetime_max}",
                   "--buoyancy", repr(options.buoyancy), "--drag", repr(drag),
                   "--bbsize", ",".join(map(str, (*options.domain_min, *options.domain_max))),
                   "--bbtype", "kill", "--seed", str(options.seed),
                   "--input", str(partio_dir / "fluid_####.bgeo"),
                   "--output", str(raw / "secondary_####.vtk")]
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(int(options.generator_threads))
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=environment, check=False)
        log = output / "foam_generator.log"
        log.write_text(process.stdout, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"FoamGenerator failed ({process.returncode}): see {log}")
        if not re.search(r"Parameters: --limits", process.stdout):
            raise ValueError("FoamGenerator did not report global calibration limits")
        records = [package_frame(raw, output, frame.index, frame.time, options.domain_min,
                                 options.domain_max, options.output_axis_order)
                   for frame in frames]

    metadata = {
        "status": "complete", "generated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": key, "source": source_settings,
        "generator": {"binary": str(binary), "sha256": binary_hash,
                      "seed": options.seed, "threads": int(options.generator_threads),
                      "source_sha256": file_sha256(options.generator_source)
                      if options.generator_source else None,
                      "patch_sha256": file_sha256(options.generator_patch)
                      if options.generator_patch else None,
                      "log_sha256": file_sha256(log)},
        "input_particle_count": counts[0], "frame_count": len(frames),
        "coordinate_space": f"source axes {options.output_axis_order}",
        "representation_note": "Heuristic whitewater postprocess, not a resolved gas phase",
        "frames": records,
    }
    write_json(output / "foam_metadata.json", metadata)
    print(f"Generated {len(frames)} sequential whitewater frames in {output}")
