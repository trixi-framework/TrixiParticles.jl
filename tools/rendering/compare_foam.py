#!/usr/bin/env python3
"""Validate and compare complete old/new whitewater caches, ignoring PLY comments."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from data import file_sha256

KINDS = ("foam", "spray", "bubbles")
PROPERTIES = ("x", "y", "z", "velocity_x", "velocity_y", "velocity_z")


def particle_payload(path: Path, count: int) -> str:
    """Hash the six interleaved little-endian Float32 attributes, not PLY comments."""
    digest = sha256()
    with path.open("rb") as source:
        if (source.readline(), source.readline()) != \
                (b"ply\n", b"format binary_little_endian 1.0\n"):
            raise ValueError(f"invalid binary whitewater PLY: {path}")
        properties = []
        vertices = None
        for _ in range(64):
            words = source.readline().decode("ascii").strip().split()
            if words == ["end_header"]:
                break
            if words[:2] == ["element", "vertex"]:
                vertices = int(words[2])
            elif words[:2] == ["property", "float"]:
                properties.append(words[2])
        else:
            raise ValueError(f"unterminated whitewater PLY header: {path}")
        if vertices != count or tuple(properties) != PROPERTIES:
            raise ValueError(f"whitewater PLY count/properties mismatch: {path}")
        bytes_read = 0
        for block in iter(lambda: source.read(1024 * 1024), b""):
            bytes_read += len(block)
            digest.update(block)
        if bytes_read != count * 6 * 4:
            raise ValueError(f"whitewater PLY payload size mismatch: {path}")
    return digest.hexdigest()


def load_cache(directory: Path) -> dict:
    metadata = json.loads((directory / "foam_metadata.json").read_text(encoding="utf-8"))
    records = metadata.get("frames", [])
    if metadata.get("status") != "complete" or metadata.get("frame_count") != len(records):
        raise ValueError(f"incomplete foam cache: {directory}")
    legacy = "foam_generator" in metadata and "generator" not in metadata
    coordinate = "xzy" if legacy else metadata.get("coordinate_space", "").removeprefix(
        "source axes ")
    frames = {}
    for item in records:
        index = int(item["source_timestep"])
        if index in frames:
            raise ValueError(f"duplicate foam frame {index}: {directory}")
        counts = item["particle_counts" if legacy else "counts"]
        raw = item["raw_particle_counts" if legacy else "raw_counts"]
        clipped = item["clipped_particle_counts" if legacy else "clipped_counts"]
        for kind in KINDS:
            if not all(isinstance(mapping.get(kind), int) and mapping[kind] >= 0
                       for mapping in (counts, raw, clipped)) or \
                    counts[kind] != raw[kind] - clipped[kind]:
                raise ValueError(f"inconsistent foam counts at frame {index}, {kind}")
        if counts["total"] != sum(counts[kind] for kind in KINDS):
            raise ValueError(f"incorrect foam total at frame {index}")
        outputs = {}
        for record in item["outputs"]:
            kind = record["particle_type" if legacy else "type"]
            name = record["filename" if legacy else "file"]
            if kind not in KINDS or kind in outputs or name != Path(name).name:
                raise ValueError(f"unexpected foam output in frame {index}: {name}")
            count = record["particle_count" if legacy else "count"]
            size = record["size_bytes" if legacy else "bytes"]
            path = directory / name
            if count != counts[kind] or not path.is_file() or \
                    size != path.stat().st_size or file_sha256(path) != record["sha256"]:
                raise ValueError(f"whitewater file or hash mismatch: {path}")
            outputs[kind] = particle_payload(path, count)
        if set(outputs) != {kind for kind in KINDS if counts[kind]}:
            raise ValueError(f"missing foam particle category at frame {index}")
        frames[index] = {"time": item["simulation_time_s"],
                         "counts": {kind: counts[kind] for kind in KINDS},
                         "raw": {kind: raw[kind] for kind in KINDS},
                         "clipped": {kind: clipped[kind] for kind in KINDS},
                         "payload_sha256": outputs}
    source = (metadata.get("source_fluid_collection_sha256") if legacy else
              metadata.get("source", {}).get("collection_sha256"))
    frame_hashes = ([frame["sha256"] for frame in metadata["source_frames"]]
                    if legacy and "source_frames" in metadata else
                    metadata.get("source", {}).get("frame_sha256"))
    seed = (metadata["foam_generator"].get("rng_seed") if legacy else
            metadata["generator"].get("seed"))
    return {"coordinate": coordinate, "frames": frames, "source_sha256": source,
            "source_frame_sha256": frame_hashes, "seed": seed}


def compare(reference: dict, candidate: dict) -> None:
    for key in ("source_sha256", "source_frame_sha256", "seed"):
        if reference[key] is not None and reference[key] != candidate[key]:
            raise ValueError(f"foam {key} differs from the accepted source")
    if reference["coordinate"] != candidate["coordinate"] or \
            reference["frames"].keys() != candidate["frames"].keys():
        raise ValueError("foam coordinate order or frame inventory differs")
    for index, expected in reference["frames"].items():
        actual = candidate["frames"][index]
        if expected != actual:
            differences = [key for key in expected if expected[key] != actual[key]]
            raise ValueError(f"foam frame {index} differs: {', '.join(differences)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    reference = load_cache(args.reference)
    print(f"Validated {len(reference['frames'])} reference whitewater frames")
    if args.candidate is not None:
        candidate = load_cache(args.candidate)
        compare(reference, candidate)
        print(f"All {len(candidate['frames'])} candidate frames match: times, counts, "
              "clipping, positions and velocities")


if __name__ == "__main__":
    main()
