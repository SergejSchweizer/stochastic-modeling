"""Create the static PDF sidecar from the WQU template and README narrative."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "report-template.pdf"
README = ROOT / "README.md"
OUTPUT = ROOT / "outputs" / "pdf" / "Stochastic_Modeling_GWP1.pdf"

PARTICIPANTS = [
    ("Umuhoza Denyse Graine", "umuhozagraine2018@gmail.com"),
    ("Opeyemi Waliyilah Oladipupo", "walylad@gmail.com"),
    ("Sergej Schweizer", "sergej.schweizer@gmail.com"),
]
GROUP_NUMBER = "16855"

PURPLE = colors.HexColor("#443B63")
BLUE = colors.HexColor("#2E6E9E")
INK = colors.HexColor("#26323D")
LIGHT = colors.HexColor("#EEF3F7")


def _cover_overlay() -> PdfReader:
    """Create an overlay aligned with the first page of the supplied template."""
    stream = BytesIO()
    layer = canvas.Canvas(stream, pagesize=letter)
    layer.setFillColor(colors.white)
    layer.rect(215, 742, 45, 18, fill=True, stroke=False)
    layer.rect(168, 726, 115, 18, fill=True, stroke=False)
    layer.setFillColor(INK)
    layer.setFont("Helvetica-Bold", 11)
    layer.drawString(218, 746, GROUP_NUMBER)
    layer.setFont("Helvetica", 9)
    layer.drawString(171, 730, "Not provided")

    row_y = [646, 622, 598]
    for (name, email), y in zip(PARTICIPANTS, row_y, strict=True):
        layer.setFont("Helvetica", 7.5 if len(name) > 25 else 8.5)
        layer.drawString(76, y, name)
        layer.setFont("Helvetica", 8.5)
        layer.drawString(206, y, "Not provided")
        layer.setFont("Helvetica", 7.8)
        layer.drawString(346, y, email)

    integrity_y = [482, 456, 430]
    for (name, _), y in zip(PARTICIPANTS, integrity_y, strict=True):
        layer.setFont("Helvetica", 9)
        layer.drawString(181, y, name)

    layer.setFont("Helvetica-Bold", 10)
    layer.drawString(78, 316, "N/A - all members contributed.")
    layer.save()
    stream.seek(0)
    return PdfReader(stream)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=PURPLE,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=PURPLE,
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Question",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#52616B"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletReport",
            parent=styles["BodyReport"],
            leftIndent=14,
            firstLineIndent=-7,
            bulletIndent=4,
        )
    )
    return styles


def _image_flowable(path: Path, caption: str, caption_style: ParagraphStyle):
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image(str(path))
    max_width, max_height = 6.65 * inch, 3.65 * inch
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether([image, Spacer(1, 4), Paragraph(escape(caption), caption_style)])


def _report_story():
    """Convert the README narrative into report flowables, excluding submission notes."""
    styles = _styles()
    story = []
    skipping_submission = False
    plot_number = 0
    image_pattern = re.compile(r"!\[(?P<caption>.+)]\((?P<path>.+)\)")
    lines = README.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("## Step 4"):
            skipping_submission = True
            continue
        if line.startswith(("## References", "## Works Cited")):
            skipping_submission = False
        if skipping_submission or not line:
            continue
        if line.startswith("# "):
            # The supplied template is the title and integrity page.
            continue
        if line.startswith("## "):
            if story:
                story.append(PageBreak())
            story.append(Paragraph(escape(line[3:]), styles["Section"]))
            continue
        if line.startswith("### "):
            if line.startswith("### 3(ii)"):
                story.append(PageBreak())
            story.append(Paragraph(escape(line[4:]), styles["Question"]))
            continue
        image_match = image_pattern.fullmatch(line)
        if image_match:
            plot_number += 1
            image_path = ROOT / image_match.group("path")
            story.append(
                _image_flowable(
                    image_path,
                    f"Plot {plot_number}. {image_match.group('caption')}",
                    styles["Caption"],
                )
            )
            continue
        if line.startswith("- "):
            story.append(
                Paragraph(escape(line[2:]), styles["BulletReport"], bulletText="-"),
            )
            continue
        line = line.replace("This companion presents", "This report presents")
        story.append(Paragraph(escape(line), styles["BodyReport"]))
    return story


def _body_page(canvas_object: canvas.Canvas, document: SimpleDocTemplate) -> None:
    canvas_object.saveState()
    canvas_object.setStrokeColor(LIGHT)
    canvas_object.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas_object.setFillColor(colors.HexColor("#66737D"))
    canvas_object.setFont("Helvetica", 8)
    canvas_object.drawString(0.65 * inch, 0.35 * inch, "MScFE 622 | Group Work Project 1")
    canvas_object.drawRightString(7.85 * inch, 0.35 * inch, f"Report page {document.page}")
    canvas_object.restoreState()


def _build_body() -> PdfReader:
    stream = BytesIO()
    document = SimpleDocTemplate(
        stream,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.72 * inch,
        title="Stochastic Modeling Group Work Project 1",
        author=", ".join(name for name, _ in PARTICIPANTS),
    )
    document.build(_report_story(), onFirstPage=_body_page, onLaterPages=_body_page)
    stream.seek(0)
    return PdfReader(stream)


def generate() -> Path:
    """Generate the final static PDF, using only page one of the template."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    template = PdfReader(TEMPLATE)
    if len(template.pages) < 1:
        raise ValueError("report template has no pages")
    cover = template.pages[0]
    cover.merge_page(_cover_overlay().pages[0])
    body = _build_body()
    writer = PdfWriter()
    writer.add_page(cover)
    for page in body.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": "Stochastic Modeling Group Work Project 1",
            "/Author": ", ".join(name for name, _ in PARTICIPANTS),
            "/Subject": "MScFE 622 Group Work Project 1",
        }
    )
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    return OUTPUT


if __name__ == "__main__":
    print(generate())
