"""Shared, Blender-free time series and provenance utilities for render tools."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Frame:
    index: int
    time: float
    path: Path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, record: dict) -> None:
    """Do not expose a complete manifest until its output is fully written."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_pvd(path: Path) -> list[Frame]:
    if not path.is_file():
        raise ValueError(f"missing VTK collection: {path}")
    document = ET.parse(path)
    collection = document.getroot().find("Collection")
    if collection is None:
        raise ValueError(f"not a VTK collection: {path}")
    frames = []
    for index, entry in enumerate(collection.findall("DataSet")):
        frame = Frame(index, float(entry.attrib["timestep"]),
                      (path.parent / entry.attrib["file"]).resolve())
        if not frame.path.is_file():
            raise ValueError(f"missing VTK frame: {frame.path}")
        frames.append(frame)
    if not frames:
        raise ValueError(f"VTK collection is empty: {path}")
    if any(not next_frame.time > current.time
           for current, next_frame in zip(frames, frames[1:])):
        raise ValueError(f"VTK frame times are not strictly increasing: {path}")
    return frames


def uniform_step(frames: list[Frame]) -> float:
    """Accept the expected Float32 serialization noise in SPH PVD timestamps."""
    if len(frames) < 2:
        raise ValueError("whitewater generation needs at least two fluid frames")
    times = [frame.time for frame in frames]
    step = (times[-1] - times[0]) / (len(times) - 1)
    if step <= 0:
        raise ValueError("nonpositive source time step")
    float32_epsilon = 2 ** -23
    tolerance = max(1e-12, 8 * float32_epsilon * max(1.0, abs(times[0]), abs(times[-1])))
    if any(abs(time - (times[0] + index * step)) > tolerance
           for index, time in enumerate(times)):
        raise ValueError("source fluid frames must be uniformly spaced")
    return step


def effective_drag(reference_drag: float, dt: float, reference_dt: float) -> float:
    if not 0 <= reference_drag <= 1 or dt <= 0 or reference_dt <= 0:
        raise ValueError("drag must be in [0, 1] and time steps must be positive")
    if reference_drag in (0, 1):
        return float(reference_drag)
    from math import expm1, log1p
    return -expm1((dt / reference_dt) * log1p(-reference_drag))


def axis_permutation(order: str) -> tuple[int, int, int]:
    if len(order) != 3 or set(order) != set("xyz"):
        raise ValueError("axis order must be a permutation of xyz")
    return tuple("xyz".index(axis) for axis in order)


def is_reflection(order: str) -> bool:
    indices = axis_permutation(order)
    return sum(a > b for i, a in enumerate(indices) for b in indices[i + 1:]) % 2 == 1


def iteration_token(filename: str, fallback: int) -> str:
    """Preserve the exact zero-padded solver iteration from a native VTK stem."""
    match = re.search(r"_(\d+)$", Path(filename).stem)
    return match.group(1) if match else str(fallback)


def bounds(lower: list[float], upper: list[float]) -> None:
    from math import isfinite
    if len(lower) != 3 or len(upper) != 3 or any(
        not isfinite(a) or not isfinite(b) or a >= b for a, b in zip(lower, upper)
    ):
        raise ValueError("domain bounds must be finite and strictly increasing")


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(24)
        source.seek(-12, 2)
        footer = source.read(12)
    if not header.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR") or \
            footer != b"\x00\x00\x00\x00IEND\xaeB\x60\x82":
        raise ValueError(f"incomplete PNG: {path}")
    return struct.unpack(">II", header[16:24])
