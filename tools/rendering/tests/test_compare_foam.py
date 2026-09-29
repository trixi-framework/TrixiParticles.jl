"""Accepted v03 cache compatibility and scientific particle-payload checks."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_foam import compare, load_cache
from data import file_sha256


class FoamParityTests(unittest.TestCase):
    def test_legacy_header_is_ignored_but_particle_velocity_is_not(self):
        with tempfile.TemporaryDirectory() as root:
            old, new = Path(root) / "old", Path(root) / "new"
            old.mkdir()
            new.mkdir()

            def cache(directory, legacy, velocity):
                name = "frame_000174_foam.ply"
                path = directory / name
                comment = "comment Blender (X,Y,Z) = simulation (x,z,y)\n" if legacy \
                    else "comment coordinates are source axes xzy\n"
                header = ("ply\nformat binary_little_endian 1.0\n" + comment +
                          "element vertex 1\nproperty float x\nproperty float y\n"
                          "property float z\nproperty float velocity_x\n"
                          "property float velocity_y\nproperty float velocity_z\nend_header\n")
                path.write_bytes(header.encode() + struct.pack("<6f", 1, 2, 3,
                                                                velocity, 5, 6))
                output = {"sha256": file_sha256(path)}
                counts = {"foam": 1, "spray": 0, "bubbles": 0, "total": 1}
                zero = {"foam": 0, "spray": 0, "bubbles": 0, "total": 0}
                frame = {"source_timestep": 174, "simulation_time_s": 0.58}
                if legacy:
                    frame.update(particle_counts=counts, raw_particle_counts=counts,
                                 clipped_particle_counts=zero,
                                 outputs=[{**output, "particle_type": "foam",
                                           "filename": name, "particle_count": 1,
                                           "size_bytes": path.stat().st_size}])
                else:
                    frame.update(counts=counts, raw_counts=counts,
                                 clipped_counts=zero,
                                 outputs=[{**output, "type": "foam", "file": name,
                                           "count": 1, "bytes": path.stat().st_size}])
                manifest = {"status": "complete", "frame_count": 1, "frames": [frame]}
                manifest["foam_generator" if legacy else "generator"] = {}
                if legacy:
                    manifest["source_fluid_collection_sha256"] = "collection-hash"
                    manifest["source_frames"] = [{"sha256": "fluid-frame-hash"}]
                    manifest["foam_generator"]["rng_seed"] = 20260915
                else:
                    manifest["source"] = {"collection_sha256": "collection-hash",
                                          "frame_sha256": ["fluid-frame-hash"]}
                    manifest["generator"]["seed"] = 20260915
                if not legacy:
                    manifest["coordinate_space"] = "source axes xzy"
                (directory / "foam_metadata.json").write_text(json.dumps(manifest))

            cache(old, True, 4)
            cache(new, False, 4)
            compare(load_cache(old), load_cache(new))
            cache(new, False, 4.5)
            with self.assertRaisesRegex(ValueError, "payload_sha256"):
                compare(load_cache(old), load_cache(new))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                (new / "frame_000174_foam.ply").write_bytes(b"modified")
                load_cache(new)


if __name__ == "__main__":
    unittest.main()
