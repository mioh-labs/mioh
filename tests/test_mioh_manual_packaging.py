from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_universal_build_enables_user_manual() -> None:
    script = (ROOT / "packaging/macOS/standalone/build_universal_app.sh").read_text()
    assert 'export INCLUDE_USER_MANUAL=1' in script


def test_dmg_build_copies_visible_japanese_manual() -> None:
    script = (ROOT / "packaging/macOS/standalone/build_app.sh").read_text()
    assert 'USER_MANUAL_PDF="${USER_MANUAL_PDF:-$ROOT/output/pdf/mioh-user-manual-ja.pdf}"' in script
    assert '"$DMG_ROOT/mioh ユーザーマニュアル.pdf"' in script


def test_manual_source_covers_every_tab() -> None:
    manual = (ROOT / "docs/mioh-user-manual-ja.md").read_text()
    for tab in ("基本", "分割", "復元", "検出", "出力", "メモリ", "設定", "再生", "ログ"):
        assert tab in manual
