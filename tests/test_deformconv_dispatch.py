import types
import unittest
from unittest import mock

import torch

from lada.models.basicvsrpp import deformconv


class DeformConvDispatchTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(1, 4, 8, 8)
        self.offset = torch.randn(1, 18, 8, 8)
        self.weight = torch.randn(4, 4, 3, 3)
        self.bias = torch.randn(4)
        self.mask = torch.sigmoid(torch.randn(1, 9, 8, 8))

    def test_non_mps_uses_torchvision(self):
        with mock.patch.object(deformconv, "_torchvision_deform_conv2d", return_value="tv") as tv:
            with mock.patch.object(deformconv, "_mps_deform_conv2d", return_value="mps") as mps:
                result = deformconv.dispatch_deform_conv2d(
                    self.x, self.offset, self.weight, self.bias, 1, 1, 1, self.mask
                )
        self.assertEqual(result, "tv")
        tv.assert_called_once()
        mps.assert_not_called()

    def test_mps_uses_torchvision_native_before_legacy_extension(self):
        fake_x = types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))
        with mock.patch.object(deformconv, "_torchvision_deform_conv2d", return_value="tv") as tv:
            with mock.patch.object(deformconv, "_mps_deform_conv2d", return_value="mps") as mps:
                result = deformconv.dispatch_deform_conv2d(
                    fake_x, self.offset, self.weight, self.bias, 1, 1, 1, self.mask
                )
        self.assertEqual(result, "tv")
        tv.assert_called_once()
        mps.assert_not_called()

    def test_mps_deform_conv_backend_env_forces_legacy_extension(self):
        fake_x = types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))
        with mock.patch.dict("os.environ", {"LADA_DEFORM_CONV_BACKEND": "mps_deform_conv"}):
            with mock.patch.object(deformconv, "_torchvision_deform_conv2d", return_value="tv") as tv:
                with mock.patch.object(deformconv, "_mps_deform_conv2d", return_value="mps") as mps:
                    result = deformconv.dispatch_deform_conv2d(
                        fake_x, self.offset, self.weight, self.bias, 1, 1, 1, self.mask
                    )
        self.assertEqual(result, "mps")
        mps.assert_called_once()
        tv.assert_not_called()

    def test_mps_falls_back_to_legacy_extension_when_torchvision_fails(self):
        fake_x = types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))
        with mock.patch.object(deformconv, "_torchvision_deform_conv2d", side_effect=RuntimeError("missing")) as tv:
            with mock.patch.object(deformconv, "_mps_deform_conv2d", return_value="mps") as mps:
                result = deformconv.dispatch_deform_conv2d(
                    fake_x, self.offset, self.weight, self.bias, 1, 1, 1, self.mask
                )
        self.assertEqual(result, "mps")
        tv.assert_called_once()
        mps.assert_called_once()

    def test_mps_reraises_torchvision_error_when_all_native_paths_fail(self):
        fake_x = types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))
        with mock.patch.object(deformconv, "_torchvision_deform_conv2d", side_effect=RuntimeError("tv failed")):
            with mock.patch.object(
                deformconv,
                "_mps_deform_conv2d",
                side_effect=deformconv.MPSDeformConvUnavailableError("legacy failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "tv failed"):
                    deformconv.dispatch_deform_conv2d(
                        fake_x, self.offset, self.weight, self.bias, 1, 1, 1, self.mask
                    )


if __name__ == "__main__":
    unittest.main()
