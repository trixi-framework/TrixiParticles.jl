"""Small translated-domain liquid/foam and stress scenes using real Blender."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import file_sha256, png_dimensions


def surface_ply(path, *, colored=False):
    vertices = [(0.4, 0.2, 0.3), (0.8, 0.2, 0.3), (0.4, 0.5, 0.3), (0.4, 0.2, 0.6)]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n")
    if colored:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for point in vertices:
            stream.write(struct.pack("<6f", *point, 0, 0, 1))
            if colored:
                stream.write(bytes((240, 90, 20)))
        for face in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
            stream.write(struct.pack("<Biii", 3, *face))


def particle_ply(path, position):
    content = ("ply\nformat binary_little_endian 1.0\n"
               "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
               "property float velocity_x\nproperty float velocity_y\n"
               "property float velocity_z\nend_header\n")
    path.write_bytes(content.encode("ascii") + struct.pack("<6f", *position, 1, 2, 3))


@unittest.skipUnless(os.environ.get("TRIXIPARTICLES_BLENDER"),
                     "set TRIXIPARTICLES_BLENDER to run headless Blender checks")
class BlenderRenderingTests(unittest.TestCase):
    def test_liquid_whitewater_stress_and_strict_resume(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            surface = path / "surface"
            secondary = path / "foam"
            surface.mkdir()
            secondary.mkdir()
            (surface / "water_000000.vtp").write_text("surface collection placeholder")
            (surface / "surface.pvd").write_text('''<VTKFile type="Collection"><Collection>
            <DataSet timestep="0" file="water_000000.vtp"/>
            </Collection></VTKFile>''')
            surface_ply(surface / "water_000000.ply")
            surface_ply(surface / "solid_000000.ply")
            surface_ply(surface / "stress.ply", colored=True)
            outputs = []
            for kind in ("foam", "spray", "bubbles"):
                filename = f"frame_000000_{kind}.ply"
                payload = secondary / filename
                particle_ply(payload, (0.45, 0.3, 0.35))
                outputs.append({"type": kind, "file": filename, "count": 1,
                                "bytes": payload.stat().st_size,
                                "sha256": file_sha256(payload)})
            (secondary / "foam_metadata.json").write_text(json.dumps({
                "status": "complete", "coordinate_space": "source axes xyz", "frame_count": 1,
                "frames": [{"source_timestep": 0, "simulation_time_s": 0,
                            "counts": {"foam": 1, "spray": 1, "bubbles": 1, "total": 3},
                            "outputs": outputs}]}))

            base = [sys.executable, str(root / "pipeline.py"), "render", "--blender",
                    os.environ["TRIXIPARTICLES_BLENDER"], "--surface-dir", str(surface),
                    "--surface-pvd", str(surface / "surface.pvd"),
                    "--solid-pattern", "solid_{frame:06d}.ply",
                    "--surface-axis-order", "xyz", "--foam-axis-order", "xyz",
                    "--camera-position", "2", "-2", "2", "--camera-target", "0.5", "0.5", "0.4",
                    "--camera-fov", "38", "--tank-min", "-1", "-1", "-1",
                    "--tank-max", "2", "2", "2",
                    "--light", "key:1:-1:2:400:1:0.9:0.8:2",
                    "--width", "64", "--height", "64", "--samples", "2", "--device", "cpu",
                    "--world-color", "0.1", "0.1", "0.1", "--world-strength", "0.4",
                    "--water-color", "0.35", "0.72", "1", "--water-roughness", "0.06",
                    "--water-ior", "1.333", "--water-absorption", "0.85", "0.14", "0.02",
                    "--water-scattering", "0.02", "0.04", "0.08",
                    "--water-scattering-anisotropy", "0.35",
                    "--solid-color", "0.62", "0.1", "0.018", "--solid-metallic", "0.92",
                    "--solid-roughness", "0.3", "--solid-coat", "0.16",
                    "--solid-floor-extension", "0.02", "--solid-bevel", "0.005",
                    "--floor-color", "0.02", "0.03", "0.05", "--floor-metallic", "0.3",
                    "--floor-roughness", "0.4", "--glass-color", "0.06", "0.42", "0.34",
                    "--glass-opacity", "0.04", "--glass-roughness", "0.3",
                    "--glass-ior", "1.36", "--glass-transmission", "0.9",
                    "--wall-thickness", "0.018",
                    "--floor-thickness", "0.018", "--visible-wall-height", "0.16"]
            whitewater = ["--foam-dir", str(secondary), "--foam-color", "0.72", "0.82", "0.86",
                          "--foam-point-radius", "0.006", "--foam-voxel-size", "0.004",
                          "--foam-threshold", "0.5", "--foam-adaptivity", "0.12",
                          "--foam-roughness", "0.46", "--foam-transmission", "0.04",
                          "--foam-subsurface", "0.035", "--foam-micro-bump", "0.00045",
                          "--foam-noise-scale", "220", "--foam-noise-detail", "3",
                          "--foam-noise-roughness", "0.7", "--foam-bump-strength", "0.22",
                          "--spray-color", "0.82", "0.94", "1", "--spray-roughness", "0.025",
                          "--spray-radius", "0.0008", "--spray-scale-range", "0.55", "1.15",
                          "--spray-exposure", "0.001", "--spray-max-stretch", "4",
                          "--bubble-color", "0.9", "0.97", "1", "--bubble-roughness", "0.025",
                          "--bubble-radius", "0.0014", "--bubble-scale-range", "0.35", "1.75",
                          "--bubble-relative-ior", "0.7501875"]
            movie = path / "sequence"
            result = subprocess.run(base + whitewater + ["--output", str(movie), "--sequence"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(movie / "frame_000000.png"), (64, 64))
            metadata = json.loads((movie / "render_metadata.json").read_text())
            self.assertEqual(metadata["frame_count"], 1)

            # The same scene through --material-preset: only scene flags remain.
            preset_base = [sys.executable, str(root / "pipeline.py"), "render", "--blender",
                           os.environ["TRIXIPARTICLES_BLENDER"], "--surface-dir",
                           str(surface), "--surface-pvd", str(surface / "surface.pvd"),
                           "--solid-pattern", "solid_{frame:06d}.ply",
                           "--surface-axis-order", "xyz", "--foam-axis-order", "xyz",
                           "--material-preset", "foam",
                           "--liquid-material", "turbulent-water",
                           "--solid-material", "anodized-copper",
                           "--glass-material", "low-iron-glass",
                           "--camera-position", "2", "-2", "2", "--camera-target",
                           "0.5", "0.5", "0.4", "--camera-fov", "38",
                           "--tank-min", "-1", "-1", "-1", "--tank-max", "2", "2", "2",
                           "--light", "key:1:-1:2:400:1:0.9:0.8:2",
                           "--width", "64", "--height", "64", "--samples", "2",
                           "--device", "cpu",
                           "--world-color", "0.1", "0.1", "0.1", "--world-strength", "0.4"]
            preset_still = path / "preset.png"
            result = subprocess.run(preset_base + ["--foam-dir", str(secondary),
                                                   "--output", str(preset_still),
                                                   "--frame", "0"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(preset_still), (64, 64))
            record = json.loads(preset_still.with_suffix(".png.json").read_text())
            self.assertEqual(record["provenance"]["settings"]["material_preset"], "foam")
            self.assertEqual(record["provenance"]["settings"]["liquid_material"],
                             "turbulent-water")
            self.assertEqual(record["provenance"]["settings"]["solid_material"],
                             "anodized-copper")
            self.assertEqual(record["provenance"]["settings"]["glass_material"],
                             "low-iron-glass")

            clear_water = path / "clear_water.png"
            clear_command = preset_base.copy()
            clear_command[clear_command.index("foam", clear_command.index("--material-preset"))] = \
                "liquid"
            clear_command[clear_command.index("turbulent-water")] = "clear-water"
            result = subprocess.run(clear_command + ["--output", str(clear_water),
                                                     "--frame", "0"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(clear_water), (64, 64))
            clear_settings = json.loads(clear_water.with_suffix(".png.json").read_text())[
                "provenance"]["settings"]
            self.assertEqual(clear_settings["liquid_material"], "clear-water")
            self.assertEqual(clear_settings["water_scattering"], [0, 0, 0])
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(base + whitewater + ["--output", str(movie), "--sequence",
                                                "--resume"], check=True,
                               capture_output=True, text=True)
            stress = path / "stress.png"
            result = subprocess.run(base + ["--output", str(stress), "--frame", "0",
                                            "--mode", "stress", "--stress-mesh",
                                            str(surface / "stress.ply"),
                                            "--stress-water-opacity", "0.005",
                                            "--stress-water-transmission", "0.72",
                                            "--stress-water-specular", "0.18"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(stress), (64, 64))

            minimal_stress = path / "stress_without_liquid_materials.png"
            stress_command = [
                sys.executable, str(root / "pipeline.py"), "render", "--blender",
                os.environ["TRIXIPARTICLES_BLENDER"], "--surface-dir", str(surface),
                "--surface-pvd", str(surface / "surface.pvd"),
                "--surface-axis-order", "xyz", "--foam-axis-order", "xyz",
                "--output", str(minimal_stress), "--frame", "0", "--mode", "stress",
                "--stress-mesh", str(surface / "stress.ply"),
                "--camera-position", "2", "-2", "2", "--camera-target", "0.5", "0.5", "0.4",
                "--camera-fov", "38", "--tank-min", "-1", "-1", "-1",
                "--tank-max", "2", "2", "2", "--light", "key:1:-1:2:400:1:0.9:0.8:2",
                "--width", "64", "--height", "64", "--samples", "2",
                "--world-color", "0.1", "0.1", "0.1", "--world-strength", "0.4",
                "--floor-color", "0.02", "0.03", "0.05", "--floor-metallic", "0.3",
                "--floor-roughness", "0.4"]
            result = subprocess.run(stress_command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(minimal_stress), (64, 64))

            stress_preset = path / "stress_with_preset.png"
            preset_command = stress_command.copy()
            preset_command[preset_command.index(str(minimal_stress))] = str(stress_preset)
            for flag, values in (("--floor-color", 3), ("--floor-metallic", 1),
                                 ("--floor-roughness", 1)):
                first = preset_command.index(flag)
                del preset_command[first:first + values + 1]
            second_blade = surface / "stress2.ply"
            second_blade.write_bytes((surface / "stress.ply").read_bytes())
            preset_command.extend(("--material-preset", "stress",
                                   "--stress-mesh", str(second_blade)))
            result = subprocess.run(preset_command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(stress_preset), (64, 64))
            settings = json.loads(stress_preset.with_suffix(".png.json").read_text())[
                "provenance"]["settings"]
            self.assertEqual(settings["stress_water_opacity"], 0.005)
            self.assertEqual(settings["stress_water_transmission"], 0.72)
            self.assertEqual(settings["stress_normal_light"], 0.28)
            self.assertEqual(settings["floor_color"], [0.02, 0.032, 0.05])
            self.assertEqual(len(settings["stress_mesh"]), 2)

            eevee = path / "eevee.png"
            result = subprocess.run(base + ["--output", str(eevee), "--frame", "0",
                                            "--engine", "eevee"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(eevee), (64, 64))

            # Continue the same case with a second saved state; a stopped
            # sequence must validate its PNG prefix and render only the rest.
            (surface / "water_000001.vtp").write_text("surface state 1")
            surface_ply(surface / "water_000001.ply")
            surface_ply(surface / "solid_000001.ply")
            (surface / "surface.pvd").write_text('''<VTKFile type="Collection"><Collection>
            <DataSet timestep="0" file="water_000000.vtp"/>
            <DataSet timestep="0.02" file="water_000001.vtp"/>
            </Collection></VTKFile>''')
            second_outputs = []
            for kind in ("foam", "spray", "bubbles"):
                name = f"frame_000001_{kind}.ply"
                payload = secondary / name
                particle_ply(payload, (0.46, 0.3, 0.35))
                second_outputs.append({"type": kind, "file": name, "count": 1,
                                       "bytes": payload.stat().st_size,
                                       "sha256": file_sha256(payload)})
            foam_metadata = json.loads((secondary / "foam_metadata.json").read_text())
            foam_metadata["frame_count"] = 2
            foam_metadata["frames"].append({"source_timestep": 1,
                                            "simulation_time_s": 0.02,
                                            "counts": {"foam": 1, "spray": 1,
                                                       "bubbles": 1, "total": 3},
                                            "outputs": second_outputs})
            (secondary / "foam_metadata.json").write_text(json.dumps(foam_metadata))
            resumable = path / "resumable"
            command = base + whitewater + ["--output", str(resumable), "--sequence"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (resumable / "frame_000001.png").unlink()
            (resumable / "render_metadata.json").unlink()
            progress_path = resumable / "render_progress.json"
            progress = json.loads(progress_path.read_text())
            progress["status"] = "in_progress"
            progress["completed"] = progress["completed"][:1]
            progress_path.write_text(json.dumps(progress))
            result = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(png_dimensions(resumable / "frame_000001.png"), (64, 64))
            self.assertEqual(json.loads((resumable / "render_metadata.json").read_text())
                             ["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
