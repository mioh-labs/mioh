# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest


MERGER = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "training"
        / "merge-basicvsrpp-hf-manifests.py"
    )
)


def entry(
    target: str,
    *,
    source: str,
    start: int,
    marker: str,
) -> dict[str, object]:
    return {
        "name": marker,
        "target_video": target,
        "mask_video": target + ".mask.mkv",
        "start_frame": start,
        "bucket": 512,
        "origins": [[0, 0]] * 9,
        "mask_reliability": [1.0] * 9,
        "mosaic_block_size": 8.0,
        "source_video_id": source,
        "recoverability": {"score": 0.5},
        "payload_must_survive": marker,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_merge_preserves_existing_deduplicates_and_balances_caps(
    tmp_path: Path,
) -> None:
    existing_path = tmp_path / "existing" / "old.jsonl"
    new_path = tmp_path / "new" / "new.jsonl"
    output_a = tmp_path / "merged-a.jsonl"
    output_b = tmp_path / "merged-b.jsonl"
    old_rows = [
        entry("../videos/old-a.mp4", source="old-a", start=0, marker="old-0"),
        entry("../videos/old-b.mp4", source="old-b", start=8, marker="old-1"),
    ]
    write_jsonl(existing_path, old_rows)
    # First new record aliases old-0 via an absolute spelling.  source-a has
    # three unique records but the per-source cap is two.  A total cap of three
    # selects two source-a records and one source-b record in balanced rounds.
    old_alias = str((existing_path.parent / "../videos/old-a.mp4").resolve())
    new_rows = [
        entry(old_alias, source="renamed-old", start=0, marker="duplicate-old"),
        entry("a.mp4", source="source-a", start=0, marker="a-0"),
        entry("a.mp4", source="source-a", start=8, marker="a-1"),
        entry("a.mp4", source="source-a", start=16, marker="a-2"),
        entry("b.mp4", source="source-b", start=0, marker="b-0"),
        # Duplicate new identity with a different non-identity payload.
        entry("b.mp4", source="source-b", start=0, marker="b-duplicate"),
    ]
    write_jsonl(new_path, new_rows)

    def merge(output: Path):
        return MERGER["merge_manifests"](
            existing_manifest=existing_path,
            new_manifest=new_path,
            output=output,
            max_new_per_source=2,
            max_new_total=3,
        )

    report_a = merge(output_a)
    report_b = merge(output_b)
    rows_a = read_jsonl(output_a)
    rows_b = read_jsonl(output_b)

    assert rows_a[: len(old_rows)] == old_rows
    assert [row["name"] for row in rows_a] == [
        "old-0",
        "old-1",
        "a-0",
        "a-1",
        "b-0",
    ]
    assert rows_a == rows_b
    assert output_a.read_bytes() == output_b.read_bytes()
    assert report_a["counts"] == {
        "existing_preserved": 2,
        "new_input": 6,
        "new_duplicate_existing": 1,
        "new_duplicate_new": 1,
        "new_unique": 4,
        "new_selected": 3,
        "new_rejected_by_limits": 1,
        "output": 5,
    }
    assert report_a["selected_new_per_source"] == {
        "source-a": 2,
        "source-b": 1,
    }
    digest = hashlib.sha256(output_a.read_bytes()).hexdigest()
    assert report_a["output"]["sha256"] == digest
    provenance = Path(str(output_a) + ".provenance.json")
    sha_file = Path(str(output_a) + ".sha256")
    assert json.loads(provenance.read_text(encoding="utf-8")) == report_a
    assert sha_file.read_text(encoding="utf-8") == f"{digest}  {output_a.name}\n"
    assert report_a["inputs"]["existing"]["sha256"] == hashlib.sha256(
        existing_path.read_bytes()
    ).hexdigest()


def test_duplicate_in_existing_is_rejected_without_touching_outputs(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.jsonl"
    new = tmp_path / "new.jsonl"
    output = tmp_path / "merged.jsonl"
    duplicate = entry("old.mp4", source="old", start=0, marker="old")
    write_jsonl(existing, [duplicate, dict(duplicate, name="duplicate")])
    write_jsonl(new, [entry("new.mp4", source="new", start=0, marker="new")])
    output.write_text("previous output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="existing manifest contains duplicate"):
        MERGER["merge_manifests"](
            existing_manifest=existing,
            new_manifest=new,
            output=output,
            max_new_per_source=2,
            max_new_total=3,
        )

    assert output.read_text(encoding="utf-8") == "previous output\n"
    assert not Path(str(output) + ".provenance.json").exists()
    assert not Path(str(output) + ".sha256").exists()


def test_atomic_jsonl_preserves_previous_output_if_serialization_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "merged.jsonl"
    output.write_text("previous\n", encoding="utf-8")

    def broken_records():
        yield {"name": "valid"}
        yield {"not_json": object()}

    with pytest.raises(TypeError):
        MERGER["atomic_write_jsonl"](broken_records(), output)
    assert output.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".merged.jsonl.tmp-*")) == []


def test_zero_limits_keep_only_the_established_curriculum(tmp_path: Path) -> None:
    existing = tmp_path / "existing.jsonl"
    new = tmp_path / "new.jsonl"
    output = tmp_path / "merged.jsonl"
    old_rows = [entry("old.mp4", source="old", start=0, marker="old")]
    write_jsonl(existing, old_rows)
    write_jsonl(new, [entry("new.mp4", source="new", start=0, marker="new")])

    report = MERGER["merge_manifests"](
        existing_manifest=existing,
        new_manifest=new,
        output=output,
        max_new_per_source=0,
        max_new_total=10,
    )
    assert read_jsonl(output) == old_rows
    assert report["counts"]["new_selected"] == 0
    assert report["counts"]["new_rejected_by_limits"] == 1
