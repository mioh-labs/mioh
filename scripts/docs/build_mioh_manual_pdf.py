#!/usr/bin/env python3
"""Render the Japanese mioh user manual Markdown into a distributable PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "docs/mioh-user-manual-ja.md"
DEFAULT_OUTPUT = ROOT / "output/pdf/mioh-user-manual-ja.pdf"
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
# Arial Unicode is used for both faces because macOS's Hiragino fonts use CFF
# outlines inside TTC files, which ReportLab cannot embed.  Heading size and
# colour still provide a clear hierarchy while keeping all Japanese glyphs.
FONT_BOLD = FONT_REGULAR
ICON = ROOT / "lada/gui/icons/mioh-icon.png"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def register_fonts() -> None:
    if not FONT_REGULAR.is_file() or not FONT_BOLD.is_file():
        raise FileNotFoundError("Arial Unicode is required to build the Japanese manual")
    pdfmetrics.registerFont(TTFont("MiohGothic", str(FONT_REGULAR), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MiohGothicBold", str(FONT_BOLD), subfontIndex=0))


def inline_markup(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = re.sub(r"`([^`]+)`", r'<font name="MiohGothic">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped.replace("  ", "<br/>")


def styles():
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyJP",
        parent=sample["BodyText"],
        fontName="MiohGothic",
        fontSize=9.2,
        leading=15,
        textColor=colors.HexColor("#20242B"),
        spaceAfter=5,
        wordWrap="CJK",
    )
    return {
        "body": body,
        "h1": ParagraphStyle(
            "H1JP", parent=body, fontName="MiohGothicBold", fontSize=19,
            leading=25, textColor=colors.HexColor("#172033"), spaceBefore=8,
            spaceAfter=10, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2JP", parent=body, fontName="MiohGothicBold", fontSize=15,
            leading=21, textColor=colors.HexColor("#234E9B"), spaceBefore=11,
            spaceAfter=7, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3JP", parent=body, fontName="MiohGothicBold", fontSize=11.5,
            leading=17, textColor=colors.HexColor("#263755"), spaceBefore=8,
            spaceAfter=4, keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "BulletJP", parent=body, leftIndent=14, firstLineIndent=-8,
            bulletIndent=4, spaceAfter=3,
        ),
        "number": ParagraphStyle(
            "NumberJP", parent=body, leftIndent=18, firstLineIndent=-13,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CodeJP", parent=body, fontName="MiohGothic", fontSize=8.1,
            leading=12, leftIndent=8, rightIndent=8, borderPadding=7,
            borderColor=colors.HexColor("#D5DAE4"), borderWidth=0.6,
            borderRadius=3, backColor=colors.HexColor("#F5F7FA"),
            spaceBefore=4, spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "TableJP", parent=body, fontSize=7.7, leading=11.2, spaceAfter=0,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderJP", parent=body, fontName="MiohGothicBold",
            fontSize=7.8, leading=11.2, textColor=colors.white, spaceAfter=0,
        ),
    }


def table_flowable(rows: list[list[str]], available_width: float, style_map) -> Table:
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    widths = [available_width / columns] * columns
    if columns == 2:
        widths = [available_width * 0.28, available_width * 0.72]
    elif columns == 3:
        widths = [available_width * 0.22, available_width * 0.18, available_width * 0.60]
    elif columns == 4:
        widths = [available_width * 0.18, available_width * 0.12,
                  available_width * 0.12, available_width * 0.58]
    cells = []
    for row_index, row in enumerate(normalized):
        cell_style = style_map["table_header"] if row_index == 0 else style_map["table"]
        cells.append([Paragraph(inline_markup(value), cell_style) for value in row])
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315FA8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8CFDA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def parse_markdown(source: str, available_width: float, style_map) -> list:
    lines = source.splitlines()
    flowables: list = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            flowables.append(Paragraph(inline_markup(" ".join(paragraph)), style_map["body"]))
            paragraph.clear()

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                flowables.append(Paragraph("<br/>".join(html.escape(x) for x in code), style_map["code"]))
                code.clear()
            in_code = not in_code
            index += 1
            continue
        if in_code:
            code.append(raw)
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            flush_paragraph()
            flowables.append(PageBreak())
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            rows: list[list[str]] = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not (candidate.startswith("|") and candidate.endswith("|")):
                    break
                values = [part.strip() for part in candidate.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", value) for value in values):
                    rows.append(values)
                index += 1
            if rows:
                flowables.append(table_flowable(rows, available_width, style_map))
                flowables.append(Spacer(1, 6))
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            if level == 1:
                # The document title is rendered on the cover.
                index += 1
                continue
            flowables.append(Paragraph(inline_markup(heading.group(2)), style_map[f"h{level}"]))
            if level == 2:
                flowables.append(HRFlowable(width="100%", thickness=0.6,
                                            color=colors.HexColor("#B9C8E2"),
                                            spaceAfter=5))
            index += 1
            continue
        bullet = re.match(r"^-\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            flowables.append(Paragraph("• " + inline_markup(bullet.group(1)), style_map["bullet"]))
            index += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if numbered:
            flush_paragraph()
            flowables.append(Paragraph(
                f"{numbered.group(1)}. {inline_markup(numbered.group(2))}",
                style_map["number"],
            ))
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return flowables


class ManualDocument(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="manual", frames=[frame], onPage=self.decorate_page))

    @staticmethod
    def decorate_page(canvas, document) -> None:
        if document.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DCE5"))
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, A4[1] - 15 * mm, A4[0] - 18 * mm, A4[1] - 15 * mm)
        canvas.setFont("MiohGothic", 7.5)
        canvas.setFillColor(colors.HexColor("#687182"))
        canvas.drawString(18 * mm, A4[1] - 11.5 * mm, "mioh ユーザーマニュアル 0.11.0")
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, str(document.page))
        canvas.restoreState()


def cover(style_map) -> list:
    result: list = [Spacer(1, 28 * mm)]
    if ICON.is_file():
        result.extend([Image(str(ICON), width=34 * mm, height=34 * mm), Spacer(1, 10 * mm)])
    title = ParagraphStyle(
        "CoverTitle", parent=style_map["h1"], fontSize=30, leading=38,
        alignment=TA_CENTER, textColor=colors.HexColor("#172033"),
    )
    subtitle = ParagraphStyle(
        "CoverSubtitle", parent=style_map["body"], fontSize=13, leading=20,
        alignment=TA_CENTER, textColor=colors.HexColor("#526078"),
    )
    note = ParagraphStyle(
        "CoverNote", parent=style_map["body"], fontSize=9, leading=15,
        alignment=TA_LEFT, leftIndent=16 * mm, rightIndent=16 * mm,
        borderColor=colors.HexColor("#C8D5EA"), borderWidth=0.8,
        borderPadding=10, backColor=colors.HexColor("#F3F6FB"),
    )
    result.extend([
        Paragraph("mioh", title),
        Paragraph("ユーザーマニュアル", title),
        Spacer(1, 7 * mm),
        Paragraph("mioh-universal 0.11.0 / macOS", subtitle),
        Spacer(1, 22 * mm),
        Paragraph(
            "動画の選択から、分割、モザイク検出、復元、合成、エンコード、"
            "VR再生、エラーからの再開までを説明します。", note,
        ),
        Spacer(1, 25 * mm),
        Paragraph("改訂日 2026年7月20日", subtitle),
        PageBreak(),
    ])
    return result


def build(source_path: Path, output_path: Path) -> None:
    register_fonts()
    style_map = styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = ManualDocument(
        str(output_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=21 * mm, bottomMargin=17 * mm,
        title="mioh ユーザーマニュアル",
        author="mioh",
        subject="mioh-universal 0.11.0 for macOS",
    )
    story = cover(style_map)
    story.extend(parse_markdown(source_path.read_text(encoding="utf-8"), document.width, style_map))
    document.build(story)
    print(output_path)


def main() -> int:
    args = arguments()
    build(args.source.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
