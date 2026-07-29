from unittest import mock

import process_video_parallel as pvp


def test_no_split_returns_original_path_without_copying(tmp_path):
    source = tmp_path / "source.mkv"
    source.touch()
    segments_dir = tmp_path / "segments"

    with mock.patch.object(pvp, "split_video") as split:
        result = pvp.prepare_processing_segments(
            source,
            segments_dir,
            no_split=True,
        )

    assert result == [source]
    split.assert_not_called()
    assert not segments_dir.exists()


def test_no_split_ignores_pre_fps_without_creating_segment_file(tmp_path):
    source = tmp_path / "source.mkv"
    source.touch()
    segments_dir = tmp_path / "segments"

    with mock.patch.object(pvp, "convert_fps_segments_parallel") as convert:
        result = pvp.prepare_processing_segments(
            source,
            segments_dir,
            no_split=True,
            pre_fps=30,
            encoder_options="-q:v 55",
        )

    assert result == [source]
    convert.assert_not_called()
    assert not segments_dir.exists()


def test_no_split_parser_option():
    args = pvp.build_arg_parser().parse_args(["--no-split"])

    assert args.no_split is True


def test_no_split_output_does_not_reuse_split_segment_output(tmp_path):
    processed_dir = tmp_path / "processed"

    split_output = pvp.processed_output_path(processed_dir, 0, no_split=False)
    direct_output = pvp.processed_output_path(processed_dir, 0, no_split=True)

    assert split_output.name == "processed_000.mp4"
    assert direct_output.name == "processed_nosplit_000.mp4"
    assert direct_output != split_output
