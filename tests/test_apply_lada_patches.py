import tempfile
import unittest
from pathlib import Path

import apply_lada_patches
from apply_lada_patches import (
    apply_patch_basicsr_torchvision_functional_tensor_compat,
    patch_basicsr_setup_py,
)


class ApplyLadaPatchesTests(unittest.TestCase):
    def test_patch_basicsr_setup_py_uses_exec_namespace(self):
        setup_py = """version_file = 'basicsr/version.py'

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_path = Path(tmp_dir) / "setup.py"
            setup_path.write_text(setup_py, encoding="utf-8")

            patched = patch_basicsr_setup_py(setup_path)

            self.assertTrue(patched)
            content = setup_path.read_text(encoding="utf-8")
            self.assertIn("namespace = {}", content)
            self.assertIn("exec(compile(f.read(), version_file, 'exec'), namespace)", content)
            self.assertIn("return namespace['__version__']", content)
            self.assertNotIn("return locals()['__version__']", content)

    def test_patch_basicsr_setup_py_is_idempotent(self):
        setup_py = """version_file = 'basicsr/version.py'

def get_version():
    namespace = {}
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'), namespace)
    return namespace['__version__']
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_path = Path(tmp_dir) / "setup.py"
            setup_path.write_text(setup_py, encoding="utf-8")

            patched = patch_basicsr_setup_py(setup_path)

            self.assertTrue(patched)
            self.assertEqual(setup_path.read_text(encoding="utf-8"), setup_py)

    def test_apply_patch_basicsr_torchvision_functional_tensor_compat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            site_packages = Path(tmp_dir)
            target_dir = site_packages / "basicsr" / "data"
            target_dir.mkdir(parents=True)
            degradations = target_dir / "degradations.py"
            degradations.write_text(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale\n",
                encoding="utf-8",
            )
            old_site_packages = apply_lada_patches.SITE_PACKAGES
            try:
                apply_lada_patches.SITE_PACKAGES = site_packages

                patched = apply_patch_basicsr_torchvision_functional_tensor_compat()

                self.assertTrue(patched)
                self.assertIn(
                    "from torchvision.transforms.functional import rgb_to_grayscale",
                    degradations.read_text(encoding="utf-8"),
                )
                self.assertTrue(list(target_dir.glob("degradations.backup_*")))
            finally:
                apply_lada_patches.SITE_PACKAGES = old_site_packages


if __name__ == "__main__":
    unittest.main()
