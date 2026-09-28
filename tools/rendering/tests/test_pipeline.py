"""Input timing, CLI, and secondary-particle payload regressions."""

from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
import pipeline


class TimeSeriesTests(unittest.TestCase):
    def test_float32_saved_times_and_actual_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("fluid_a.vtu", "fluid_b.vtu", "fluid_c.vtu"):
                (root / name).write_bytes(b"test VTK placeholder")
            pvd = root / "fluid.pvd"
            pvd.write_text('''<VTKFile type="Collection"><Collection>
            <DataSet timestep="0" file="fluid_a.vtu"/>
            <DataSet timestep="0.0033333334" file="fluid_b.vtu"/>
            <DataSet timestep="0.006666667" file="fluid_c.vtu"/>
            </Collection></VTKFile>''')
            frames = data.read_pvd(pvd)
            self.assertEqual([frame.index for frame in frames], [0, 1, 2])
            self.assertTrue(all(frame.path.is_file() for frame in frames))
            self.assertAlmostEqual(data.uniform_step(frames), 1 / 300, places=7)
            self.assertRaises(ValueError, data.uniform_step,
                              [data.Frame(0, 0, root), data.Frame(1, 0.02, root),
                               data.Frame(2, 0.041, root)])

    def test_drag_and_reflected_coordinates(self):
        step = data.effective_drag(0.8, 1 / 300, 0.02)
        self.assertAlmostEqual(1 - (1 - step) ** 6, 0.8, places=14)
        self.assertAlmostEqual(step, 0.23527550866827002, places=14)
        self.assertEqual(data.axis_permutation("xzy"), (0, 2, 1))
        self.assertTrue(data.is_reflection("xzy"))
        self.assertFalse(data.is_reflection("xyz"))
        self.assertEqual(data.iteration_token("surface_fluid_1_000100.vtp", 1), "000100")
        data.bounds([-1, -2, -3], [2, 3, 4])
        with self.assertRaises(ValueError):
            data.bounds([0, 0, 0], [0, 1, 1])

    def test_whitewater_requires_an_explicit_seed_and_domain(self):
        args = ("foam --fluid-pvd source.pvd --generator foam --output out "
                "--domain-min -1 0 0 --domain-max 2 3 4 --generator-radius 0.01 "
                "--foam-scale 25 --lifetime-min 1 --lifetime-max 3 "
                "--buoyancy 2 --drag 0.8 --drag-reference-step 0.02 "
                "--generator-threads 1 --output-axis-order xyz").split()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            pipeline.parser().parse_args(args)
        parsed = pipeline.parser().parse_args(args + ["--seed", "144"])
        pipeline.validate(parsed)
        self.assertEqual(parsed.seed, 144)
        self.assertEqual(parsed.domain_min, [-1, 0, 0])
        parsed.seed = -1
        with self.assertRaisesRegex(ValueError, "invalid whitewater"):
            pipeline.validate(parsed)
        with self.assertRaisesRegex(Exception, "light"):
            pipeline.light_specification("invalid-light")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            pipeline.parser().parse_args(args + ["--seed", "144", "--config", "case.json"])


class WhitewaterPayloadTests(unittest.TestCase):
    def test_packaged_types_clip_against_translated_domain_and_swap_velocities(self):
        try:
            import meshio
            import numpy as np
            import foam
        except ImportError as error:
            self.skipTest(f"foam conversion dependencies unavailable: {error}")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            cases = {"foam": [(-1.5, 0.5, 0.75), (2.01, 0.5, 0.75)],
                     "spray": [(1.5, 0.6, 0.8)],
                     "bubbles": [(1.75, 0.7, 0.9)]}
            for kind, points in cases.items():
                meshio.write(source / f"secondary_0000_{kind}.vtk",
                             meshio.Mesh(np.array(points, dtype=np.float32),
                                         [("vertex", np.arange(len(points), dtype=np.int32)
                                           .reshape(-1, 1))],
                                         point_data={"velocity": np.tile(
                                             np.array([[1, 2, 3]], dtype=np.float32),
                                             (len(points), 1))}))
            frame = foam.package_frame(source, output, 0, 0.25,
                                       [-2, 0, 0], [2, 2, 2], "xzy")
            self.assertEqual(frame["counts"], {"foam": 1, "spray": 1,
                                                "bubbles": 1, "total": 3})
            self.assertEqual(frame["clipped_counts"]["foam"], 1)
            spray = output / "frame_000000_spray.ply"
            data_bytes = spray.read_bytes().split(b"end_header\n", 1)[1]
            self.assertEqual(data_bytes,
                             struct.pack("<6f", 1.5, 0.8, 0.6, 1.0, 3.0, 2.0))
            self.assertEqual(frame["outputs"][1]["sha256"], data.file_sha256(spray))


if __name__ == "__main__":
    unittest.main()
