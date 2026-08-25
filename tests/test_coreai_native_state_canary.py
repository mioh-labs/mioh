# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "scripts/apple/canary_coreai_native_state.py"
SWIFT_RUNNER = ROOT / "tests/swift/CoreAINativeStateCanary.swift"
AB_BENCHMARK = ROOT / "scripts/apple/benchmark_basicvsrpp_native_state.py"
AB_SWIFT_RUNNER = ROOT / "tests/swift/BasicVSRPPNativeStateBenchmark.swift"


def test_native_state_canary_covers_lowering_compilation_and_swift_runtime():
    canary = CANARY.read_text()
    runner = SWIFT_RUNNER.read_text()

    for contract in (
        'state_names=["acc"]',
        "program.optimize()",
        '"coreai-build",',
        '"compile",',
        "validate_inspection(source_inspection)",
        "validate_inspection(compiled_inspection)",
        "validate_runtime(source_runtime",
        "validate_runtime(compiled_runtime",
    ):
        assert contract in canary

    for contract in (
        'stateDescriptor(of: "acc")',
        "InferenceFunction.MutableViews()",
        'states.insert(&state, for: "acc")',
        "function.run(",
        "states: states",
        '"stateAfter": values(state)',
    ):
        assert contract in runner


def test_basicvsrpp_native_state_ab_keeps_a_control_and_adoption_gates():
    benchmark = AB_BENCHMARK.read_text()
    runner = AB_SWIFT_RUNNER.read_text()

    for contract in (
        "StatefulPropagationContinue6",
        'state_names=["state_n1", "state_n2", "flow_previous"]',
        "export_control(",
        'default=1.03',
        '"adoption_recommended"',
        '"keep-explicit-boundary-io"',
    ):
        assert contract in benchmark

    for contract in (
        'states.insert(&n1, for: "state_n1")',
        'states.insert(&n2, for: "state_n2")',
        'states.insert(&previousFlow, for: "flow_previous")',
        '"shippingVsControl"',
        '"controlVsStateful"',
        '"statefulSpeedup"',
    ):
        assert contract in runner
