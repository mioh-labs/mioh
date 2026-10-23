from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "packaging" / "macOS" / "standalone"


def test_mioh_build_includes_native_range_selector():
    build = (STANDALONE / "build_app.sh").read_text()
    source = (STANDALONE / "MiohVideoRangeSelection.swift").read_text()
    assert '"$PACKAGE_DIR/MiohVideoRangeSelection.swift"' in build
    for label in ("ここを開始に", "ここを終了に", "選択時間（長さ）", "動画全体を選択"):
        assert label in source
    assert "VideoPlayer(player: preview.player)" in source
    assert "step: 0.01" in source
    assert 'Slider(value: binding, in: 0...safeDuration' in source


def test_mioh_selected_range_reaches_native_decoder_and_progress():
    app = (STANDALONE / "MiohApp.swift").read_text()
    pipeline = (STANDALONE / "NativePreviewPipeline.swift").read_text()
    assert "selectedDecodeStartNanoseconds" in app
    assert "outputCoreStartNanoseconds: selectedCoreStartNanoseconds" in app
    assert "outputCoreEndNanoseconds: selectedCoreEndNanoseconds" in app
    assert "startNanoseconds: selectedDecodeStartNanoseconds" in app
    assert "position - progressStart" in pipeline
    assert "progressEnd - progressStart" in pipeline


def test_mioh_partial_export_trims_audio_to_the_same_half_open_range():
    pipeline = (STANDALONE / "NativePreviewPipeline.swift").read_text()
    assert "sourceStartNanoseconds: config.outputCoreStartNanoseconds" in pipeline
    assert "sourceEndNanoseconds: config.outputCoreEndNanoseconds" in pipeline
    assert '"-ss", String(format: "%.9f", selectedRange.start)' in pipeline
    assert '"-t", String(format: "%.9f", selectedRange.duration)' in pipeline
    assert "selectedRangeDuration == nil" in pipeline


def test_cluster_refuses_to_silently_ignore_a_partial_range():
    cluster = (STANDALONE / "MiohClusterController.swift").read_text()
    assert "guard !runner.usesPartialVideoRange" in cluster
    assert "現在ローカル実行を選択してください" in cluster
