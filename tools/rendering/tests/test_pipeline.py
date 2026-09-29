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
import materials
import pipeline


def minimal_render(*extra):
    base = ("render --blender blender --surface-dir surface --surface-pvd surface.pvd "
            "--output out.png --frame 0 --surface-axis-order xyz --foam-axis-order xyz "
            "--camera-position 2 -2 2 --camera-target 0.5 0.5 0.4 --camera-fov 38 "
            "--tank-min -1 -1 -1 --tank-max 2 2 2 "
            "--light key:1:-1:2:400:1:0.9:0.8:2 --width 64 --height 64 --samples 2 "
            "--world-color 0.1 0.1 0.1 --world-strength 0.4").split()
    return pipeline.parser().parse_args(base + list(extra))


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

    def test_independent_materials_and_explicit_flags(self):
        base = ("--liquid-material", "turbulent-water",
                "--solid-material", "anodized-copper",
                "--glass-material", "low-iron-glass",
                "--floor-material", "dark-metal",
                "--wall-thickness", "0.018", "--floor-thickness", "0.018",
                "--visible-wall-height", "0.16")
        secondary = ("--foam-dir", "foam", "--foam-material", "whitewater-froth",
                     "--spray-material", "water-droplet",
                     "--bubble-material", "submerged-air")
        parsed = minimal_render(*base, *secondary)
        pipeline.validate(parsed)
        self.assertEqual(tuple(parsed.water_color), (0.35, 0.72, 1.0))
        self.assertEqual(tuple(parsed.floor_color), (0.012, 0.018, 0.028))
        self.assertEqual(parsed.floor_material, "dark-metal")
        self.assertEqual(parsed.glass_transmission, 0.90)
        self.assertEqual(parsed.foam_point_radius, 0.006)
        self.assertEqual(tuple(parsed.spray_scale_range), (0.55, 1.15))
        self.assertEqual(parsed.bubble_relative_ior, 1.0 / 1.333)
        self.assertEqual(parsed.instance_seed, 0)
        self.assertEqual(parsed.liquid_material, "turbulent-water")
        self.assertEqual(parsed.solid_material, "anodized-copper")
        self.assertEqual(parsed.glass_material, "low-iron-glass")

        components = minimal_render(*base, *secondary, "--solid-roughness", "0.18",
                                    "--glass-transmission", "0.72")
        pipeline.validate(components)
        self.assertEqual(components.solid_material, "anodized-copper")
        self.assertEqual(components.glass_material, "low-iron-glass")
        self.assertEqual(components.solid_roughness, 0.18)
        self.assertEqual(components.solid_color, (0.62, 0.10, 0.018))
        self.assertEqual(components.glass_transmission, 0.72)
        self.assertEqual(components.glass_color, (0.06, 0.42, 0.34))

        clear_base = ("--liquid-material", "clear-water", *base[2:])
        clear = minimal_render(*clear_base, *secondary, "--water-roughness", "0.12")
        pipeline.validate(clear)
        self.assertEqual(clear.liquid_material, "clear-water")
        self.assertEqual(tuple(clear.water_color), (1.0, 1.0, 1.0))
        self.assertEqual(tuple(clear.water_absorption), (0.34, 0.0565, 0.00922))
        self.assertEqual(tuple(clear.water_scattering), (0.0, 0.0, 0.0))
        self.assertEqual(clear.water_roughness, 0.12)
        self.assertEqual(clear.solid_material, "anodized-copper")

        overridden = minimal_render(*base, *secondary, "--water-roughness", "0.5",
                                    "--foam-threshold", "0.7", "--floor-color", "0.2",
                                    "0.3", "0.4")
        pipeline.validate(overridden)
        self.assertEqual(overridden.water_roughness, 0.5)
        self.assertEqual(overridden.foam_threshold, 0.7)
        self.assertEqual(tuple(overridden.water_color), (0.35, 0.72, 1.0))
        self.assertEqual(tuple(overridden.floor_color), (0.2, 0.3, 0.4))

        incomplete = minimal_render()
        with self.assertRaisesRegex(ValueError, "required render parameters"):
            pipeline.validate(incomplete)

        stress = minimal_render("--mode", "stress", "--stress-mesh", "stress.ply",
                                "--floor-material", "neutral-stress",
                                "--ghost-material", "v03-ghost-water",
                                "--stress-normal-light", "0.28",
                                "--solid-bevel", "0.004", "--solid-floor-extension", "0.02")
        pipeline.validate(stress)
        self.assertEqual(tuple(stress.water_color), (0.015, 0.28, 0.46))
        self.assertEqual(tuple(stress.floor_color), (0.020, 0.032, 0.050))
        self.assertEqual(stress.floor_coat, 0.10)
        self.assertEqual(stress.stress_water_opacity, 0.005)
        self.assertEqual(stress.stress_water_transmission, 0.72)
        self.assertEqual(stress.stress_normal_light, 0.28)
        self.assertEqual(stress.solid_bevel, 0.004)

        unused = minimal_render("--mode", "stress", "--stress-mesh", "stress.ply",
                                "--floor-material", "neutral-stress",
                                "--liquid-material", "turbulent-water")
        with self.assertRaisesRegex(ValueError, "pre-colored"):
            pipeline.validate(unused)
        unused_glass = minimal_render("--mode", "stress", "--floor-material", "neutral-stress",
                                      "--stress-mesh", "stress.ply", "--glass-material",
                                      "low-iron-glass")
        with self.assertRaisesRegex(ValueError, "pre-colored"):
            pipeline.validate(unused_glass)

        unused_foam = minimal_render(*base, "--foam-material", "whitewater-froth")
        with self.assertRaisesRegex(ValueError, "need --foam-dir"):
            pipeline.validate(unused_foam)

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            minimal_render("--material-preset", "foam")
        self.assertIn("--liquid-material clear-water", materials.describe_materials())
        self.assertIn("--solid-material anodized-copper", materials.describe_materials())
        self.assertIn("--glass-material low-iron-glass", materials.describe_materials())
        self.assertIn("--floor-material neutral-stress", materials.describe_materials())


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
