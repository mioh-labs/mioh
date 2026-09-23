#!/usr/bin/env python3
"""Re-export the public PiperSR video Core ML package with FP16 I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
from coremltools.models.utils import change_input_output_tensor_type
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"destination already exists: {args.destination}")
    model = ct.models.MLModel(str(args.source), compute_units=ct.ComputeUnit.CPU_AND_NE)
    converted = change_input_output_tensor_type(
        ml_model=model,
        from_type=ArrayFeatureType.FLOAT32,
        to_type=ArrayFeatureType.FLOAT16,
        input_names=["*"],
        output_names=["*"],
    )
    converted.save(str(args.destination))
    spec = converted.get_spec()
    assert spec.description.input[0].type.multiArrayType.dataType == ArrayFeatureType.FLOAT16
    assert spec.description.output[0].type.multiArrayType.dataType == ArrayFeatureType.FLOAT16
    print(args.destination)


if __name__ == "__main__":
    main()
