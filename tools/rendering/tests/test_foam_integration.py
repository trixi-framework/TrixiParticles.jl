"""Optional headless test with the patched FoamGenerator and real Partio."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import file_sha256


@unittest.skipUnless(os.environ.get("TRIXIPARTICLES_FOAM_GENERATOR"),
                     "set TRIXIPARTICLES_FOAM_GENERATOR to run the patched generator")
class FoamGeneratorTests(unittest.TestCase):
    def test_sequential_source_and_seeded_reuse(self):
        try:
            import meshio
            import numpy as np
        except ImportError as error:
            self.skipTest(f"foam conversion dependencies unavailable: {error}")
        script = Path(__file__).resolve().parents[1] / "pipeline.py"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "fluid"
            source.mkdir()
            random = np.random.default_rng(4)
            grid = np.stack(np.meshgrid(*([np.arange(8) * 0.06 + 0.2] * 3),
                                        indexing="ij"), axis=-1)
            positions = grid.reshape(-1, 3).astype(np.float32)
            speed = random.normal(0, 4, positions.shape).astype(np.float32)
            for i in range(3):
                points = positions + i * 0.002 * speed
                meshio.write(source / f"fluid_{i}.vtu",
                             meshio.Mesh(points, [("vertex", np.arange(len(points), dtype=np.int32)
                                                  .reshape(-1, 1))],
                                         point_data={"velocity": speed}))
            (source / "fluid.pvd").write_text('''<VTKFile type="Collection"><Collection>
            <DataSet timestep="0" file="fluid_0.vtu"/>
            <DataSet timestep="0.02" file="fluid_1.vtu"/>
            <DataSet timestep="0.04" file="fluid_2.vtu"/>
            </Collection></VTKFile>''')
            output = folder / "whitewater"
            command = [sys.executable, str(script), "foam", "--fluid-pvd",
                       str(source / "fluid.pvd"), "--generator",
                       os.environ["TRIXIPARTICLES_FOAM_GENERATOR"],
                       "--output", str(output), "--domain-min", "-1", "-1", "-1",
                       "--domain-max", "2", "2", "2", "--generator-radius", "0.06",
                       "--foam-scale", "10000", "--lifetime-min", "2",
                       "--lifetime-max", "5", "--buoyancy", "2", "--drag", "0.8",
                       "--drag-reference-step", "0.02", "--seed", "144",
                       "--generator-threads", "1", "--output-axis-order", "xyz"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            path = output / "foam_metadata.json"
            first_hash = file_sha256(path)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["frame_count"], 3)
            self.assertEqual(record["generator"]["seed"], 144)
            self.assertEqual([frame["source_timestep"] for frame in record["frames"]],
                             [0, 1, 2])
            self.assertEqual(record["source"]["effective_drag"], 0.8)
            reused = subprocess.run(command + ["--reuse"], capture_output=True, text=True)
            self.assertEqual(reused.returncode, 0, reused.stdout + reused.stderr)
            self.assertEqual(file_sha256(path), first_hash)
            self.assertGreater(sum(item["counts"]["total"] for item in record["frames"]), 0,
                               (output / "foam_generator.log").read_text()[-3000:])
            original = tuple(item["sha256"] for frame in record["frames"]
                             for item in frame["outputs"])
            repeat = folder / "same_seed"
            same = command.copy()
            same[same.index(str(output))] = str(repeat)
            result = subprocess.run(same, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            repeated = json.loads((repeat / "foam_metadata.json").read_text())
            self.assertEqual(original, tuple(item["sha256"] for frame in repeated["frames"]
                                             for item in frame["outputs"]))

            other = folder / "another_seed"
            changed = same.copy()
            changed[changed.index(str(repeat))] = str(other)
            changed[changed.index("144")] = "145"
            result = subprocess.run(changed, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            different = json.loads((other / "foam_metadata.json").read_text())
            self.assertNotEqual(original, tuple(item["sha256"] for frame in different["frames"]
                                                for item in frame["outputs"]))


if __name__ == "__main__":
    unittest.main()
