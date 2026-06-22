import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.spynet import (
    avg_pool2d_nchw,
    interpolate_bilinear_nchw,
    interpolate_bilinear_nchw_to_size,
    spynet_basic_module_forward,
    spynet_compute_flow,
    spynet_forward,
)


class MLXSPyNetTests(unittest.TestCase):
    def test_spynet_basic_module_matches_pytorch(self):
        rng = np.random.default_rng(1101)
        x = rng.normal(size=(1, 8, 8, 9)).astype(np.float32)
        tensors = _random_spynet_basic_module_tensors(rng)

        expected = _torch_spynet_basic_module_forward(
            torch.from_numpy(x),
            {name: torch.from_numpy(value) for name, value in tensors.items()},
        ).numpy()
        actual = np.array(
            spynet_basic_module_forward(
                mx.array(x),
                {name: mx.array(value) for name, value in tensors.items()},
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)

    def test_avg_pool2d_nchw_matches_pytorch(self):
        rng = np.random.default_rng(1102)
        x = rng.normal(size=(1, 3, 8, 10)).astype(np.float32)

        expected = F.avg_pool2d(torch.from_numpy(x), kernel_size=2, stride=2, count_include_pad=False).numpy()
        actual = np.array(avg_pool2d_nchw(mx.array(x), kernel_size=2, stride=2))

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_interpolate_bilinear_nchw_matches_pytorch(self):
        rng = np.random.default_rng(1103)
        x = rng.normal(size=(1, 2, 5, 6)).astype(np.float32)

        expected = F.interpolate(torch.from_numpy(x), scale_factor=2, mode="bilinear", align_corners=True).numpy()
        actual = np.array(interpolate_bilinear_nchw(mx.array(x), scale_factor=2, align_corners=True))

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_interpolate_bilinear_nchw_to_size_matches_pytorch_align_corners_false(self):
        rng = np.random.default_rng(1105)
        x = rng.normal(size=(1, 2, 5, 7)).astype(np.float32)

        expected = F.interpolate(torch.from_numpy(x), size=(8, 11), mode="bilinear", align_corners=False).numpy()
        actual = np.array(interpolate_bilinear_nchw_to_size(mx.array(x), size=(8, 11), align_corners=False))

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_spynet_compute_flow_matches_pytorch_for_multiple_of_32_input(self):
        rng = np.random.default_rng(1104)
        ref = rng.normal(size=(1, 3, 64, 64)).astype(np.float32)
        supp = rng.normal(size=(1, 3, 64, 64)).astype(np.float32)
        tensors = _random_spynet_tensors(rng)

        expected = _torch_spynet_compute_flow(
            torch.from_numpy(ref),
            torch.from_numpy(supp),
            {name: torch.from_numpy(value) for name, value in tensors.items()},
        ).numpy()
        actual = np.array(
            spynet_compute_flow(
                mx.array(ref),
                mx.array(supp),
                {name: mx.array(value) for name, value in tensors.items()},
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)

    def test_spynet_forward_matches_pytorch_for_non_multiple_of_32_input(self):
        rng = np.random.default_rng(1106)
        ref = rng.normal(size=(1, 3, 65, 70)).astype(np.float32)
        supp = rng.normal(size=(1, 3, 65, 70)).astype(np.float32)
        tensors = _random_spynet_tensors(rng)

        expected = _torch_spynet_forward(
            torch.from_numpy(ref),
            torch.from_numpy(supp),
            {name: torch.from_numpy(value) for name, value in tensors.items()},
        ).numpy()
        actual = np.array(
            spynet_forward(
                mx.array(ref),
                mx.array(supp),
                {name: mx.array(value) for name, value in tensors.items()},
            )
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=7e-4)


def _torch_spynet_basic_module_forward(x, tensors):
    channels = [8, 32, 64, 32, 16, 2]
    out = x
    for layer in range(5):
        out = F.conv2d(
            out,
            tensors[f"basic_module.{layer}.conv.weight"],
            tensors[f"basic_module.{layer}.conv.bias"],
            padding=3,
        )
        if layer < 4:
            out = F.relu(out)
    return out


def _torch_spynet_compute_flow(ref, supp, tensors):
    ref = [(ref - tensors["mean"]) / tensors["std"]]
    supp = [(supp - tensors["mean"]) / tensors["std"]]
    for _ in range(5):
        ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
        supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
    ref = ref[::-1]
    supp = supp[::-1]
    n, _, h, w = ref[-1].shape
    flow = torch.zeros(n, 2, h // 32, w // 32)
    for level in range(6):
        if level == 0:
            flow_up = flow
        else:
            flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
        module_tensors = {
            key.removeprefix(f"basic_module.{level}."): value
            for key, value in tensors.items()
            if key.startswith(f"basic_module.{level}.")
        }
        warped = _torch_flow_warp_border(supp[level], flow_up.permute(0, 2, 3, 1))
        flow = flow_up + _torch_spynet_basic_module_forward(torch.cat([ref[level], warped, flow_up], dim=1), module_tensors)
    return flow


def _torch_spynet_forward(ref, supp, tensors):
    _, _, h, w = ref.shape
    w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
    h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
    ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
    supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
    flow = F.interpolate(_torch_spynet_compute_flow(ref_up, supp_up, tensors), size=(h, w), mode="bilinear", align_corners=False)
    flow[:, 0, :, :] *= float(w) / float(w_up)
    flow[:, 1, :, :] *= float(h) / float(h_up)
    return flow


def _torch_flow_warp_border(x, flow):
    _, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype),
        torch.arange(0, w, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(x, grid_flow, mode="bilinear", padding_mode="border", align_corners=True)


def _random_spynet_basic_module_tensors(rng):
    scale = 0.02
    channels = [8, 32, 64, 32, 16, 2]
    tensors = {}
    for layer in range(5):
        tensors[f"basic_module.{layer}.conv.weight"] = (
            rng.normal(size=(channels[layer + 1], channels[layer], 7, 7)) * scale
        ).astype(np.float32)
        tensors[f"basic_module.{layer}.conv.bias"] = (rng.normal(size=(channels[layer + 1],)) * scale).astype(np.float32)
    return tensors


def _random_spynet_tensors(rng):
    tensors = {
        "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1),
        "std": np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1),
    }
    for level in range(6):
        for name, value in _random_spynet_basic_module_tensors(rng).items():
            tensors[f"basic_module.{level}.{name}"] = value
    return tensors


if __name__ == "__main__":
    unittest.main()
