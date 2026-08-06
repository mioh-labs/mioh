from types import SimpleNamespace
from unittest import mock

import process_video_parallel as pvp


def test_batch_processes_only_immediate_regular_video_files(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    input_dir.mkdir()

    first = input_dir / "first.mp4"
    second = input_dir / "second.mkv"
    first.touch()
    second.touch()
    (input_dir / "notes.txt").touch()
    (input_dir / "directory.mov").mkdir()

    nested = input_dir / "nested"
    nested.mkdir()
    (nested / "nested.mp4").touch()

    outside = tmp_path / "outside.mp4"
    outside.touch()
    (input_dir / "linked.mp4").symlink_to(outside)

    processor = mock.Mock()
    args = SimpleNamespace(overwrite=True)
    with mock.patch.object(pvp, "ParallelVideoProcessor", return_value=processor):
        with mock.patch.object(pvp, "cleanup_resources"):
            with mock.patch.object(pvp.time, "sleep"):
                pvp.process_batch(input_dir, output_dir, temp_dir, args)

    processed = [call.args[0] for call in processor.process.call_args_list]
    assert processed == [first, second]
