# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "training" / "prepare-basicvsrpp-ema-finetune.py"
SPEC = importlib.util.spec_from_file_location("prepare_basicvsrpp_ema_finetune", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _checkpoint() -> dict:
    state = OrderedDict(
        [
            ("generator.layer.weight", torch.full((2, 2), -1.0)),
            ("generator.layer.bias", torch.full((2,), -2.0)),
            ("generator_ema.layer.weight", torch.full((2, 2), 3.0)),
            ("generator_ema.layer.bias", torch.full((2,), 4.0)),
            ("discriminator.weight", torch.full((1,), 5.0)),
            ("step_counter", torch.tensor(9000)),
        ]
    )
    state._metadata = {"": {"version": 1}, "generator": {"version": 2}}
    return {
        "meta": {
            "iter": 9000,
            "epoch": 0,
            "experiment_name": "source-run",
            "cfg": "resume-only config must not be copied",
        },
        "state_dict": state,
        "optimizer": {"generator": {"state": {1: "must disappear"}}},
        "message_hub": {"runtime_info": {"iter": 9000}},
        "param_schedulers": ["must disappear"],
    }


class PrepareBasicVSRPPEMAFinetuneTests(unittest.TestCase):
    def test_requires_explicit_trust_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            torch.save(_checkpoint(), source)

            with mock.patch.object(MODULE.torch, "load") as load:
                with self.assertRaisesRegex(PermissionError, "trust_checkpoint"):
                    MODULE.prepare_checkpoint(
                        source, output, trust_checkpoint=False
                    )
                load.assert_not_called()
            self.assertFalse(output.exists())

    def test_copies_ema_to_both_generators_and_strips_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            source_checkpoint = _checkpoint()
            torch.save(source_checkpoint, source)
            expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()

            provenance = MODULE.prepare_checkpoint(
                source, output, trust_checkpoint=True
            )
            prepared = torch.load(output, map_location="cpu", weights_only=True)

            self.assertEqual(set(prepared), {"meta", "state_dict"})
            state = prepared["state_dict"]
            torch.testing.assert_close(
                state["generator.layer.weight"],
                source_checkpoint["state_dict"]["generator_ema.layer.weight"],
            )
            torch.testing.assert_close(
                state["generator.layer.bias"],
                source_checkpoint["state_dict"]["generator_ema.layer.bias"],
            )
            torch.testing.assert_close(
                state["generator.layer.weight"],
                state["generator_ema.layer.weight"],
            )
            torch.testing.assert_close(
                state["discriminator.weight"],
                source_checkpoint["state_dict"]["discriminator.weight"],
            )
            self.assertEqual(state["step_counter"].item(), 0)
            self.assertEqual(state._metadata, source_checkpoint["state_dict"]._metadata)

            recorded = prepared["meta"][MODULE.PROVENANCE_KEY]
            self.assertEqual(recorded["source_sha256"], expected_sha)
            self.assertEqual(recorded["source_metadata"]["iter"], 9000)
            self.assertEqual(recorded["generator_suffix_count"], 2)
            self.assertEqual(recorded["reset_step_counters"], ["step_counter"])
            self.assertFalse(recorded["optimizer_state_preserved"])
            self.assertFalse(recorded["message_hub_preserved"])
            self.assertEqual(provenance, recorded)
            self.assertNotIn("iter", prepared["meta"])
            self.assertNotIn("cfg", prepared["meta"])

    def test_rejects_suffix_set_mismatch_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            checkpoint = _checkpoint()
            del checkpoint["state_dict"]["generator_ema.layer.bias"]
            torch.save(checkpoint, source)

            with self.assertRaisesRegex(ValueError, "suffix sets differ"):
                MODULE.prepare_checkpoint(source, output, trust_checkpoint=True)
            self.assertFalse(output.exists())

    def test_rejects_shape_mismatch_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            checkpoint = _checkpoint()
            checkpoint["state_dict"]["generator_ema.layer.weight"] = torch.zeros(3, 2)
            torch.save(checkpoint, source)

            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                MODULE.prepare_checkpoint(source, output, trust_checkpoint=True)
            self.assertFalse(output.exists())

    def test_refuses_overwrite_unless_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            torch.save(_checkpoint(), source)
            output.write_bytes(b"keep-me")

            with self.assertRaises(FileExistsError):
                MODULE.prepare_checkpoint(source, output, trust_checkpoint=True)
            self.assertEqual(output.read_bytes(), b"keep-me")

            MODULE.prepare_checkpoint(
                source, output, trust_checkpoint=True, overwrite=True
            )
            prepared = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(
                prepared["meta"][MODULE.PROVENANCE_KEY]["generator_suffix_count"],
                2,
            )

    def test_atomic_save_cleans_up_temporary_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "output.pth"
            torch.save(_checkpoint(), source)

            with mock.patch.object(MODULE.torch, "save", side_effect=OSError("full")):
                with self.assertRaisesRegex(OSError, "full"):
                    MODULE.prepare_checkpoint(source, output, trust_checkpoint=True)

            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
