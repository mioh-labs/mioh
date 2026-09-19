from pathlib import Path

import pytest

from lada.prompting.h3_gemma import build_user_content, load_h3_system_prompt


def _write_skill(root: Path) -> None:
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("skill body", encoding="utf-8")
    (root / "references" / "base-en.txt").write_text("base guide", encoding="utf-8")
    (root / "references" / "ref-en.txt").write_text("reference guide", encoding="utf-8")


def test_base_mode_loads_only_base_guide(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    prompt = load_h3_system_prompt(tmp_path, "i2va")
    assert "skill body" in prompt
    assert "base guide" in prompt
    assert "reference guide" not in prompt


def test_auto_mode_loads_both_guides(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    prompt = load_h3_system_prompt(tmp_path, "auto")
    assert "base guide" in prompt
    assert "reference guide" in prompt


def test_system_prompt_loads_mioh_music_video_guide(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    prompt = load_h3_system_prompt(tmp_path, "ref2va")
    assert "mioh_h3_music_video_prompting.md" in prompt
    assert "For continuation parts of this shot" in prompt
    assert "not a restart of" in prompt
    assert "the opening pose or action" in prompt
    assert "two-layer structure" in prompt
    assert "GLOBAL CONTINUITY" in prompt
    assert "stable identity/style/music/continuity rules" in prompt
    assert "LOCATION, FRAMING, ACTION, CAMERA" in prompt
    assert "interval-specific composition" in prompt
    assert "vary at least three" in prompt


def test_user_content_labels_ordered_images(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.png"
    first.write_bytes(b"jpeg")
    second.write_bytes(b"png")
    content = build_user_content("A seaside scene", "ref2va", 10, [first, second])
    assert content[0]["type"] == "text"
    assert "Reference images: 2" in content[0]["text"]
    assert content[1]["text"] == "<Picture 1> follows."
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[3]["text"] == "<Picture 2> follows."


def test_duration_is_limited_to_h3_range() -> None:
    with pytest.raises(ValueError, match="between 4 and 15"):
        build_user_content("request", "t2va", 16, [])
