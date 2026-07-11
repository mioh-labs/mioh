# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.apple import export_basicvsrpp_coreai as exporter


class IdentityGenerator(torch.nn.Module):
    def forward(self, frames):
        return frames


class BasicVSRPPCoreAIArgumentTests(unittest.TestCase):
    def test_defaults_define_t18_fp16_contract(self):
        args = exporter.parse_args([])

        self.assertEqual(args.frames, 18)
        self.assertEqual(args.imgsz, 256)
        self.assertEqual(args.model, exporter.DEFAULT_MODEL)
        self.assertEqual(args.output, exporter.DEFAULT_OUTPUT)
        self.assertFalse(args.skip_reference_inference)

    def test_reference_inference_can_be_skipped_for_conversion_diagnostics(self):
        args = exporter.parse_args(["--skip-reference-inference"])

        self.assertTrue(args.skip_reference_inference)

    def test_t36_contract_uses_t36_default_output(self):
        args = exporter.parse_args(["--frames", "36"])

        self.assertEqual(args.frames, 36)
        self.assertEqual(
            args.output,
            Path("model_weights/basicvsrpp-v1.2-t36-fp16.aimodel"),
        )

    def test_t90_contract_uses_t90_default_output(self):
        args = exporter.parse_args(["--frames", "90"])

        self.assertEqual(args.frames, 90)
        self.assertEqual(
            args.output,
            Path("model_weights/basicvsrpp-v1.2-t90-fp16.aimodel"),
        )

    def test_report_path_is_derived_from_aimodel_output(self):
        output = Path("model_weights/basicvsrpp-t18-fp16.aimodel")

        self.assertEqual(
            exporter.derive_report_path(output),
            Path("model_weights/basicvsrpp-t18-fp16.report.json"),
        )

    def test_checkpoint_identity_records_size_and_sha256(self):
        payload = b"basicvsrpp-checkpoint"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            path.write_bytes(payload)

            identity = exporter.checkpoint_identity(path)

        self.assertEqual(identity["path"], str(path))
        self.assertEqual(identity["size_bytes"], len(payload))
        self.assertEqual(identity["sha256"], hashlib.sha256(payload).hexdigest())

    def test_failure_report_extracts_torch_operator_names(self):
        report = exporter.new_report(exporter.parse_args([]))

        exporter.record_failure(
            report,
            "coreai_convert",
            RuntimeError(
                "unsupported aten.grid_sampler_2d.default. "
                "torchvision::deform_conv2d and coreai::tensor.scatter"
            ),
        )

        self.assertEqual(
            report["error"]["operators"],
            [
                "aten.grid_sampler_2d.default",
                "coreai::tensor.scatter",
                "torchvision::deform_conv2d",
            ],
        )

    def test_failure_report_is_json_serializable(self):
        args = exporter.parse_args([])
        report = exporter.new_report(args)
        exporter.record_failure(report, "preflight", RuntimeError("missing dependency"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            exporter.write_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(saved["success"])
        self.assertEqual(saved["failed_stage"], "preflight")
        self.assertEqual(saved["error"]["type"], "RuntimeError")


class BasicVSRPPCoreAIContractTests(unittest.TestCase):
    def test_example_input_is_deterministic_fp16_t18(self):
        first = exporter.make_example_input(18, 256, seed=7)
        second = exporter.make_example_input(18, 256, seed=7)

        self.assertEqual(tuple(first.shape), (1, 18, 3, 256, 256))
        self.assertEqual(first.dtype, torch.float16)
        self.assertTrue(torch.equal(first, second))

    def test_wrapper_preserves_fixed_contract(self):
        example = torch.zeros((1, 18, 3, 16, 16), dtype=torch.float16)
        wrapper = exporter.BasicVSRPPExportWrapper(IdentityGenerator()).eval()

        output = wrapper(example)
        summary = exporter.validate_output(output, example)

        self.assertEqual(summary["shape"], [1, 18, 3, 16, 16])
        self.assertEqual(summary["dtype"], "float16")

    def test_validate_output_rejects_shape_changes(self):
        example = torch.zeros((1, 18, 3, 16, 16), dtype=torch.float16)

        with self.assertRaisesRegex(ValueError, "output shape"):
            exporter.validate_output(example[:, :-1], example)

    def test_model_loader_exports_ema_generator_used_for_inference(self):
        loaded_model = mock.Mock()
        loaded_model.generator = IdentityGenerator()
        loaded_model.generator_ema = torch.nn.Sequential(IdentityGenerator())

        with mock.patch(
            "lada.models.basicvsrpp.inference.load_model",
            return_value=loaded_model,
        ) as load_model:
            wrapper = exporter.load_generator(Path("checkpoint.pth"))

        self.assertIsInstance(wrapper, exporter.BasicVSRPPExportWrapper)
        self.assertIs(wrapper.generator, loaded_model.generator_ema)
        load_model.assert_called_once_with(
            None,
            "checkpoint.pth",
            device="cpu",
            fp16=True,
        )

    def test_exported_operator_summary_counts_call_function_nodes(self):
        class AddModel(torch.nn.Module):
            def forward(self, value):
                return value + value

        exported = torch.export.export(AddModel(), (torch.ones(1),))

        operators = exporter.summarize_exported_operators(exported)

        self.assertEqual(operators["aten.add.Tensor"], 1)

    def test_program_asset_is_saved_with_path_object(self):
        program = mock.Mock()
        output = Path("model.aimodel")

        exporter.save_program_asset(program, output)

        program.save_asset.assert_called_once_with(output)

    def test_converter_registers_custom_kernels_before_program(self):
        coreai_torch = mock.Mock()
        converter = coreai_torch.TorchConverter.return_value
        exported = mock.sentinel.exported
        kernel = mock.sentinel.kernel

        result = exporter.convert_exported_program(
            exported,
            coreai_torch,
            [kernel],
        )

        self.assertIs(result, converter.to_coreai.return_value)
        self.assertEqual(
            converter.method_calls,
            [
                mock.call.register_custom_kernels([kernel]),
                mock.call.add_exported_program(
                    exported,
                    input_names=["frames"],
                    output_names=["restored"],
                ),
                mock.call.to_coreai(),
            ],
        )


class BasicVSRPPCoreAIProbeTests(unittest.TestCase):
    def test_grid_sample_override_is_scoped_to_export(self):
        flow_warp_module = importlib.import_module(
            "lada.models.basicvsrpp.mmagic.flow_warp"
        )
        original = flow_warp_module.safe_mps_grid_sample
        image = torch.zeros((1, 2, 3, 4), dtype=torch.float16)
        grid = torch.zeros((1, 3, 4, 2), dtype=torch.float16)
        expected = torch.ones_like(image)
        kernel = mock.Mock(return_value=expected)

        with exporter.use_grid_sample_metal_kernel(kernel):
            actual = flow_warp_module.safe_mps_grid_sample(
                image,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

        self.assertIs(actual, expected)
        self.assertIs(flow_warp_module.safe_mps_grid_sample, original)
        kernel.assert_called_once()

    def test_deform_conv_override_is_scoped_to_export(self):
        basicvsrpp_module = importlib.import_module(
            "lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net"
        )
        original = basicvsrpp_module.dispatch_deform_conv2d
        image = torch.zeros((1, 2, 4, 4), dtype=torch.float16)
        offset = torch.zeros((1, 18, 4, 4), dtype=torch.float16)
        weight = torch.zeros((3, 2, 3, 3), dtype=torch.float16)
        bias = torch.zeros(3, dtype=torch.float16)
        mask = torch.ones((1, 9, 4, 4), dtype=torch.float16)
        expected = torch.ones((1, 3, 4, 4), dtype=torch.float16)
        kernel = mock.Mock(
            return_value=torch.ones((16, 3), dtype=torch.float16)
        )

        with exporter.use_deform_conv_metal_kernel(kernel):
            actual = basicvsrpp_module.dispatch_deform_conv2d(
                image,
                offset,
                weight,
                bias,
                1,
                1,
                1,
                mask,
            )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIs(basicvsrpp_module.dispatch_deform_conv2d, original)
        kernel.assert_called_once()

    def test_missing_coreai_fails_in_preflight_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "checkpoint.pth"
            model.write_bytes(b"checkpoint")
            output = Path(temp_dir) / "model.aimodel"
            report_path = Path(temp_dir) / "probe.json"
            args = exporter.parse_args(
                [
                    "--model",
                    str(model),
                    "--output",
                    str(output),
                    "--report",
                    str(report_path),
                ]
            )

            with mock.patch.object(
                exporter,
                "import_coreai",
                side_effect=RuntimeError("install coreai-torch"),
            ):
                exit_code = exporter.run_probe(args)

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["failed_stage"], "preflight")
        self.assertIn("preflight", report["stage_seconds"])
        self.assertEqual(
            report["checkpoint"]["sha256"],
            hashlib.sha256(b"checkpoint").hexdigest(),
        )

    def test_existing_output_requires_allow_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "checkpoint.pth"
            model.write_bytes(b"checkpoint")
            output = Path(temp_dir) / "model.aimodel"
            output.mkdir()
            report_path = Path(temp_dir) / "probe.json"
            args = exporter.parse_args(
                [
                    "--model",
                    str(model),
                    "--output",
                    str(output),
                    "--report",
                    str(report_path),
                ]
            )

            exit_code = exporter.run_probe(args)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["failed_stage"], "preflight")
        self.assertIn("allow-overwrite", report["error"]["message"])

    def test_skip_reference_inference_reaches_torch_export_directly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "checkpoint.pth"
            model.write_bytes(b"checkpoint")
            report_path = Path(temp_dir) / "probe.json"
            args = exporter.parse_args(
                [
                    "--model",
                    str(model),
                    "--output",
                    str(Path(temp_dir) / "model.aimodel"),
                    "--report",
                    str(report_path),
                    "--skip-reference-inference",
                ]
            )
            wrapper = mock.Mock()

            with (
                mock.patch.object(
                    exporter,
                    "import_coreai",
                    return_value=(mock.Mock(), mock.Mock()),
                ),
                mock.patch.object(exporter, "load_generator", return_value=wrapper),
                mock.patch.object(
                    exporter.torch.export,
                    "export",
                    side_effect=RuntimeError("torch export reached"),
                ),
            ):
                exit_code = exporter.run_probe(args)

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["failed_stage"], "torch_export")
        self.assertNotIn("reference_inference", report["stage_seconds"])
        wrapper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
