"""Regressions for the parameterized FoamGenerator build script."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "build" / "build_foam_generator.py"


def load_build_script():
    specification = importlib.util.spec_from_file_location(
        "render_build_foam_generator", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class BuildScriptTests(unittest.TestCase):
    def test_pinned_defaults_and_bundled_patch(self):
        module = load_build_script()
        options = module.parser().parse_args([
            "--source-dir", "source", "--build-dir", "build"])
        self.assertEqual(options.repo_url, module.DEFAULT_REPO_URL)
        self.assertEqual(options.commit, "f3f677140761db7637b5443beb54f19f1f835ed4")
        self.assertEqual(Path(options.patch).resolve(),
                         BUILD_SCRIPT.with_name("splishsplash_foam_generator.patch"))
        self.assertTrue(Path(options.patch).is_file())
        self.assertTrue(options.double_precision)
        flags = module.cmake_flags(options)
        self.assertIn("-DFOAM_GENERATOR_ONLY=ON", flags)
        self.assertIn("-DUSE_DOUBLE_PRECISION=ON", flags)

    def test_patch_state_against_temporary_checkout(self):
        module = load_build_script()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email",
                            "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name",
                            "Test"], check=True)
            target = source / "file.txt"
            target.write_text("original\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(source), "add", "file.txt"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "base"],
                           check=True)
            target.write_text("patched\n", encoding="utf-8")
            patch = source / "change.patch"
            with patch.open("w", encoding="utf-8") as stream:
                subprocess.run(["git", "-C", str(source), "diff", "--", "file.txt"],
                               stdout=stream, check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "-q", "--",
                            "file.txt"], check=True)
            self.assertEqual(module.patch_state(source, patch), "applicable")
            subprocess.run(["git", "-C", str(source), "apply", str(patch)],
                           check=True)
            self.assertEqual(module.patch_state(source, patch), "applied")

    def test_help_documents_source_and_build_directories(self):
        completed = subprocess.run([sys.executable, str(BUILD_SCRIPT), "--help"],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--source-dir", completed.stdout)
        self.assertIn("--build-dir", completed.stdout)


if __name__ == "__main__":
    unittest.main()
