import json
import os
from pathlib import Path

import pytest

from lada.datasetcreation.gemma_representative_selector import (
    DatasetClip,
    category_token_similarity,
    classification_signature,
    group_consecutive_similar_clips,
    load_dataset_clip,
    materialize_representative_dataset,
    remove_legacy_flat_representative_links,
    parse_clip_index,
    representative_score,
    select_representatives,
    split_representatives_by_source,
    signature_similarity,
)


def make_clip(tmp_path: Path, index: int, source: str = "source.mp4", size=(600, 800)) -> DatasetClip:
    video = tmp_path / f"{source}-{index:06d}-.mp4"
    metadata = tmp_path / f"{source}-{index:06d}-.json"
    mask = tmp_path / f"{source}-{index:06d}-.mkv"
    for path in (video, metadata, mask):
        path.touch()
    return DatasetClip(video, metadata, mask, source, index, 96, 24, size[1], size[0])


def decision(activity="scene_a", orientation="front", pose="sitting", framing="medium", **scores):
    value = {
        "usable": True,
        "confidence": 0.9,
        "activity_key": activity,
        "body_orientation": orientation,
        "pose": pose,
        "framing": framing,
        "subject_scale": 0.8,
        "sharpness": 0.8,
        "clarity": 0.8,
        "occlusion": 0.1,
    }
    value.update(scores)
    return value


def test_parse_clip_index():
    assert parse_clip_index(Path("movie.mp4-000123-.mp4")) == 123


def test_load_dataset_clip_uses_source_metadata_and_relative_pairs(tmp_path: Path):
    root = tmp_path / "dataset"
    meta = root / "crop_unscaled_meta" / "source.mp4-000007-.json"
    video = root / "crop_unscaled_img" / "source.mp4-000007-.mp4"
    mask = root / "crop_unscaled_mask" / "source.mp4-000007-.mkv"
    for directory in (meta.parent, video.parent, mask.parent):
        directory.mkdir(parents=True)
    video.touch()
    mask.touch()
    meta.write_text(json.dumps({
        "name": "original-source.mp4", "fps": 24, "frames_count": 192,
        "scene_shape": [500, 700],
        "relative_nsfw_video_path": "../crop_unscaled_img/source.mp4-000007-.mp4",
        "relative_mask_video_path": "../crop_unscaled_mask/source.mp4-000007-.mkv",
    }))
    clip = load_dataset_clip(meta)
    assert clip.source_name == "original-source.mp4"
    assert clip.clip_index == 7
    assert clip.duration == 8
    assert (clip.scene_height, clip.scene_width) == (500, 700)


def test_signature_similarity_is_weighted_and_ignores_unclear():
    a = classification_signature(decision())
    b = classification_signature(decision(framing="close_up"))
    assert signature_similarity(a, b) == 0.8
    b["activity"] = "unclear"
    assert signature_similarity(a, b) == 0.4


def test_activity_key_allows_incidental_extra_tokens():
    assert category_token_similarity("adjusting_clothing", "hand_adjusting_clothing") == 2 / 3
    a = classification_signature(decision(activity="adjusting_clothing", pose="unclear", orientation="mixed"))
    b = classification_signature(decision(activity="hand_adjusting_clothing", pose="unclear", orientation="mixed"))
    assert signature_similarity(a, b) == pytest.approx(0.6)


def test_known_orientation_or_pose_change_is_not_a_duplicate():
    a = classification_signature(decision(orientation="front", pose="sitting"))
    assert signature_similarity(a, classification_signature(decision(orientation="profile"))) == 0
    assert signature_similarity(a, classification_signature(decision(pose="lying"))) == 0


def test_only_consecutive_similar_clips_are_grouped(tmp_path: Path):
    items = [
        (make_clip(tmp_path, 1), decision()),
        (make_clip(tmp_path, 2), decision(framing="close_up")),
        (make_clip(tmp_path, 9), decision()),
        (make_clip(tmp_path, 10), decision(activity="scene_b")),
    ]
    groups = group_consecutive_similar_clips(items, max_index_gap=3, min_similarity=0.6)
    assert [[clip.clip_index for clip, _ in group] for group in groups] == [[1, 2], [9], [10]]


def test_similar_clips_can_group_across_one_different_clip(tmp_path: Path):
    items = [
        (make_clip(tmp_path, 1), decision()),
        (make_clip(tmp_path, 2), decision(activity="scene_b")),
        (make_clip(tmp_path, 3), decision(activity="hand_scene_a")),
    ]
    groups = group_consecutive_similar_clips(items, max_index_gap=3, min_similarity=0.6)
    assert [[clip.clip_index for clip, _ in group] for group in groups] == [[1, 3], [2]]


def test_best_clear_large_clip_becomes_representative(tmp_path: Path):
    smaller = make_clip(tmp_path, 1, size=(400, 400))
    larger = make_clip(tmp_path, 2, size=(768, 768))
    weak = decision(sharpness=0.5, clarity=0.6, subject_scale=0.6)
    strong = decision(sharpness=0.95, clarity=0.95, subject_scale=0.95)
    summaries, selected = select_representatives([(smaller, weak), (larger, strong)])
    assert selected == [larger]
    assert summaries[0]["representative"] == str(larger.video_path)
    assert representative_score(larger, strong) > representative_score(smaller, weak)


def test_unusable_group_has_no_representative(tmp_path: Path):
    clip = make_clip(tmp_path, 1)
    summaries, selected = select_representatives([
        (clip, decision(usable=False, sharpness=0.2, clarity=0.2))
    ])
    assert selected == []
    assert summaries[0]["representative"] is None


def test_content_policy_rejection_does_not_override_good_technical_scores(tmp_path: Path):
    clip = make_clip(tmp_path, 1)
    summaries, selected = select_representatives([(clip, decision(usable=False))])
    assert selected == [clip]
    assert summaries[0]["ranking"][0]["model_usable"] is False
    assert summaries[0]["ranking"][0]["technically_usable"] is True


def test_source_cap_keeps_only_best_representatives(tmp_path: Path):
    items = [
        (
            make_clip(tmp_path, index),
            decision(activity=f"scene_{index}", sharpness=0.5 + index / 10),
        )
        for index in range(1, 6)
    ]
    summaries, selected = select_representatives(items, max_per_source=2)
    assert [clip.clip_index for clip in selected] == [4, 5]
    assert sum(summary["selected_after_source_cap"] for summary in summaries) == 2


def test_dataset_splits_never_mix_one_source(tmp_path: Path):
    clips = []
    for source_index in range(12):
        source = f"source-{source_index}.mp4"
        clips.extend(make_clip(tmp_path, index + 1, source=source) for index in range(3))
    splits = split_representatives_by_source(clips, train_ratio=0.8, validation_ratio=0.1)
    source_sets = {
        name: {clip.source_name for clip in values} for name, values in splits.items()
    }
    assert source_sets["train"].isdisjoint(source_sets["validation"])
    assert source_sets["train"].isdisjoint(source_sets["test"])
    assert source_sets["validation"].isdisjoint(source_sets["test"])
    assert sum(len(values) for values in splits.values()) == len(clips)


def test_materialize_links_all_training_triplets(tmp_path: Path):
    clip = make_clip(tmp_path, 1)
    output = tmp_path / "selected"
    materialize_representative_dataset([clip], output, "symlink")
    links = [
        output / "crop_unscaled_img" / clip.video_path.name,
        output / "crop_unscaled_mask" / clip.mask_path.name,
        output / "crop_unscaled_meta" / clip.metadata_path.name,
    ]
    assert all(path.is_symlink() for path in links)
    assert all(os.path.samefile(path, source) for path, source in zip(
        links, [clip.video_path, clip.mask_path, clip.metadata_path]
    ))


def test_remove_legacy_layout_only_removes_symlinks(tmp_path: Path):
    root = tmp_path / "selected"
    legacy = root / "crop_unscaled_img"
    legacy.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.touch()
    (legacy / "linked.mp4").symlink_to(source)
    real = legacy / "keep.mp4"
    real.touch()
    remove_legacy_flat_representative_links(root)
    assert not (legacy / "linked.mp4").exists()
    assert real.exists()
