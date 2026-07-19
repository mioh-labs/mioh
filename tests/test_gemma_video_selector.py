import json
from pathlib import Path

import numpy as np

from lada.datasetcreation.gemma_video_selector import (
    AnalysisWindow,
    build_analysis_windows,
    criteria_digest,
    make_contact_sheet,
    parse_json_object,
    record_key,
    sample_timestamps,
    write_selected_list,
)


def test_whole_video_builds_one_window():
    assert build_analysis_windows(120, 0, 0) == [AnalysisWindow(0, 120)]


def test_scan_windows_retain_the_tail():
    assert build_analysis_windows(125, 20, 60) == [
        AnalysisWindow(0, 20),
        AnalysisWindow(60, 80),
        AnalysisWindow(120, 125),
    ]


def test_sample_timestamps_use_cell_centres():
    assert sample_timestamps(AnalysisWindow(10, 18), 4) == [11, 13, 15, 17]


def test_parse_json_accepts_fenced_or_surrounded_output():
    assert parse_json_object('```json\n{"matches_criteria": true}\n```')["matches_criteria"]
    assert parse_json_object('answer: {"confidence": 0.75} done')["confidence"] == 0.75


def test_contact_sheet_is_encoded_jpeg():
    frames = [np.zeros((80, 120, 3), np.uint8), np.full((120, 80, 3), 255, np.uint8)]
    jpeg = make_contact_sheet(frames, [1.0, 2.0], columns=2)
    assert jpeg.startswith(b"\xff\xd8")


def test_selected_list_deduplicates_selected_videos(tmp_path: Path):
    output = tmp_path / "selected.txt"
    records = [
        {"status": "ok", "video": "/a.mp4", "selected": True},
        {"status": "ok", "video": "/a.mp4", "selected": True},
        {"status": "ok", "video": "/b.mp4", "selected": False},
        {"status": "error", "video": "/c.mp4", "selected": True},
    ]
    assert write_selected_list(records, output) == [Path("/a.mp4")]
    assert output.read_text() == "/a.mp4\n"


def test_record_key_and_criteria_digest_are_stable():
    record = {
        "video": "/a.mp4", "window_start": 1.23449, "window_end": 2.0,
        "criteria_id": criteria_digest("test"),
    }
    assert record_key(json.loads(json.dumps(record))) == (
        "/a.mp4", 1.234, 2.0, criteria_digest("test")
    )
