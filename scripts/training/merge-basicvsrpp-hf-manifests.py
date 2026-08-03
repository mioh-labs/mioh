#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Merge an established and a new BasicVSR++ Native-HF curriculum.

The established curriculum is authoritative: every existing record is kept in
its original order and with its complete payload.  New records whose stable
identity is already present are ignored.  Remaining new records are selected
in source-balanced rounds, preserving their input order within each source,
until either the per-source or total new-record limit is reached.

The JSONL is replaced atomically.  A deterministic provenance report and a
conventional SHA-256 file are emitted alongside it; the JSONL is published
last and therefore acts as the commit marker for the three-file update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


FORMAT = "mioh-basicvsrpp-hf-merge-v1"
IDENTITY_FORMAT = "resolved-target-video-and-start-frame-v1"
SELECTION_FORMAT = "source-balanced-input-order-v1"


@dataclass(frozen=True)
class ManifestRecord:
    value: dict[str, Any]
    identity: str
    source_video_id: str
    ordinal: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-per-source", type=int, required=True)
    parser.add_argument("--max-new-total", type=int, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        help="default: <output>.provenance.json",
    )
    parser.add_argument(
        "--sha256-file",
        type=Path,
        help="default: <output>.sha256",
    )
    args = parser.parse_args(argv)
    if args.max_new_per_source < 0:
        parser.error("--max-new-per-source cannot be negative")
    if args.max_new_total < 0:
        parser.error("--max-new-total cannot be negative")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    # Materialize every record before any output file is opened.  This is an
    # important part of the atomic replacement contract for generator input.
    return "".join(_canonical_json(record) + "\n" for record in records)


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def atomic_write_text(value: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
        # Also persist the directory entry where the platform supports it.
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> None:
    atomic_write_text(_jsonl_text(records), output)


def stable_identity(value: dict[str, Any], manifest_root: Path) -> str:
    """Return the same identity for aliases of one decoded nine-frame window.

    Names, scores, mask paths and crop origins may be regenerated while the
    underlying target window remains unchanged.  The upstream recoverability
    builder therefore also de-duplicates by resolved target plus start frame.
    """

    try:
        target_value = value["target_video"]
        start_value = value["start_frame"]
    except KeyError as error:
        raise ValueError(f"missing identity field: {error.args[0]}") from error
    if not isinstance(target_value, str) or not target_value.strip():
        raise ValueError("target_video must be a non-empty string")
    if isinstance(start_value, bool):
        raise ValueError("start_frame must be a non-negative integer")
    try:
        start = int(start_value)
    except (TypeError, ValueError) as error:
        raise ValueError("start_frame must be a non-negative integer") from error
    if start < 0 or str(start) != str(start_value):
        raise ValueError("start_frame must be a non-negative integer")
    target = Path(target_value).expanduser()
    if not target.is_absolute():
        target = manifest_root / target
    identity_value = {
        "start_frame": start,
        "target_video": str(target.resolve()),
    }
    return hashlib.sha256(_canonical_json(identity_value).encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL line {line_number} in {path}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL line {line_number} in {path} is not an object"
                )
            source_video_id = value.get("source_video_id")
            if not isinstance(source_video_id, str) or not source_video_id.strip():
                raise ValueError(
                    f"invalid source_video_id on line {line_number} in {path}"
                )
            try:
                identity = stable_identity(value, path.parent)
            except ValueError as error:
                raise ValueError(
                    f"invalid identity on line {line_number} in {path}: {error}"
                ) from error
            records.append(
                ManifestRecord(
                    value=value,
                    identity=identity,
                    source_video_id=source_video_id,
                    ordinal=len(records),
                )
            )
    if not records:
        raise ValueError(f"manifest contains no records: {path}")
    return records


def select_new_records(
    records: Sequence[ManifestRecord],
    *,
    max_per_source: int,
    max_total: int,
) -> list[ManifestRecord]:
    """Select deterministically without allowing one long source to dominate."""

    if max_per_source < 0 or max_total < 0:
        raise ValueError("new-record limits cannot be negative")
    if not max_per_source or not max_total:
        return []
    by_source: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_video_id].append(record)

    selected_identities: set[str] = set()
    for depth in range(max_per_source):
        for source_video_id in sorted(by_source):
            candidates = by_source[source_video_id]
            if depth >= len(candidates):
                continue
            selected_identities.add(candidates[depth].identity)
            if len(selected_identities) >= max_total:
                break
        if len(selected_identities) >= max_total:
            break

    # Output keeps the new manifest's original relative ordering, while the
    # selected *set* is balanced by source above.
    return [record for record in records if record.identity in selected_identities]


def _input_identity(path: Path, records: Sequence[ManifestRecord]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "entries": len(records),
        "sha256": file_sha256(path),
    }


def merge_manifests(
    *,
    existing_manifest: Path,
    new_manifest: Path,
    output: Path,
    max_new_per_source: int,
    max_new_total: int,
    provenance: Path | None = None,
    sha256_file: Path | None = None,
) -> dict[str, Any]:
    if max_new_per_source < 0 or max_new_total < 0:
        raise ValueError("new-record limits cannot be negative")
    provenance = provenance or Path(str(output) + ".provenance.json")
    sha256_file = sha256_file or Path(str(output) + ".sha256")
    destinations = [output.resolve(), provenance.resolve(), sha256_file.resolve()]
    if len(set(destinations)) != len(destinations):
        raise ValueError("output, provenance and SHA-256 paths must be distinct")

    existing = load_manifest(existing_manifest)
    new = load_manifest(new_manifest)

    existing_identities: set[str] = set()
    for record in existing:
        if record.identity in existing_identities:
            # Silently deleting an established entry would violate the main
            # contract.  Refuse the merge and leave all previous outputs alone.
            raise ValueError(
                "existing manifest contains duplicate stable identity: "
                f"{record.identity}"
            )
        existing_identities.add(record.identity)

    unique_new: list[ManifestRecord] = []
    seen_new: set[str] = set()
    skipped_existing = 0
    skipped_new = 0
    for record in new:
        if record.identity in existing_identities:
            skipped_existing += 1
            continue
        if record.identity in seen_new:
            skipped_new += 1
            continue
        seen_new.add(record.identity)
        unique_new.append(record)

    selected_new = select_new_records(
        unique_new,
        max_per_source=max_new_per_source,
        max_total=max_new_total,
    )
    selected_per_source = Counter(
        record.source_video_id for record in selected_new
    )
    merged_values = [record.value for record in existing]
    merged_values.extend(record.value for record in selected_new)

    manifest_text = _jsonl_text(merged_values)
    manifest_bytes = manifest_text.encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    selected_identity_text = "".join(
        record.identity + "\n" for record in selected_new
    )
    report: dict[str, Any] = {
        "format": FORMAT,
        "identity_format": IDENTITY_FORMAT,
        "selection_format": SELECTION_FORMAT,
        "inputs": {
            "existing": _input_identity(existing_manifest, existing),
            "new": _input_identity(new_manifest, new),
        },
        "limits": {
            "max_new_per_source": max_new_per_source,
            "max_new_total": max_new_total,
        },
        "counts": {
            "existing_preserved": len(existing),
            "new_input": len(new),
            "new_duplicate_existing": skipped_existing,
            "new_duplicate_new": skipped_new,
            "new_unique": len(unique_new),
            "new_selected": len(selected_new),
            "new_rejected_by_limits": len(unique_new) - len(selected_new),
            "output": len(merged_values),
        },
        "selected_new_per_source": dict(sorted(selected_per_source.items())),
        "selected_new_identities_sha256": hashlib.sha256(
            selected_identity_text.encode("ascii")
        ).hexdigest(),
        "output": {
            "path": str(output.resolve()),
            "bytes": len(manifest_bytes),
            "entries": len(merged_values),
            "sha256": manifest_digest,
        },
    }

    # Build every payload first.  Metadata is published before the JSONL, and
    # the atomic JSONL rename is the commit marker for this reproducible merge.
    provenance_text = _json_text(report)
    sha256_text = f"{manifest_digest}  {output.name}\n"
    atomic_write_text(provenance_text, provenance)
    atomic_write_text(sha256_text, sha256_file)
    atomic_write_text(manifest_text, output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = merge_manifests(
        existing_manifest=args.existing_manifest,
        new_manifest=args.new_manifest,
        output=args.output,
        max_new_per_source=args.max_new_per_source,
        max_new_total=args.max_new_total,
        provenance=args.provenance,
        sha256_file=args.sha256_file,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
