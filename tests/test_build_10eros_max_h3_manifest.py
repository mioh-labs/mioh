from scripts.apple.build_10eros_max_h3_manifest import simple_flow_sigmas


def test_simple_flow_sigmas_matches_comfyui_six_step_table() -> None:
    assert simple_flow_sigmas(6, 12.0) == [
        1.0,
        0.9836839001376056,
        0.9600575746671467,
        0.9230769230769231,
        0.8575096277278561,
        0.7063799788508988,
        0.0,
    ]


def test_simple_flow_sigmas_is_strictly_descending_to_zero() -> None:
    sigmas = simple_flow_sigmas(20, 12.0)
    assert len(sigmas) == 21
    assert sigmas[-1] == 0.0
    assert all(left > right for left, right in zip(sigmas, sigmas[1:]))
