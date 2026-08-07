"""Create the static PDF sidecar from the WQU template and README narrative."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import nbformat
from matplotlib.mathtext import math_to_image
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
NOTEBOOK = ROOT / "notebooks" / "MScFE622_GWP1.ipynb"
OUTPUT = ROOT / "outputs" / "pdf" / "Stochastic_Modeling_GWP1.pdf"

PARTICIPANTS = [
    ("Umuhoza Denyse Graine", "umuhozagraine2018@gmail.com"),
    ("Opeyemi Waliyilah Oladipupo", "walylad@gmail.com"),
    ("Sergej Schweizer", "sergej.schweizer@gmail.com"),
]
GROUP_WORK_PROJECT = "1"
GROUP_NUMBER = "16855"

PURPLE = colors.HexColor("#443B63")
BLUE = colors.HexColor("#2E6E9E")
INK = colors.HexColor("#26323D")
LIGHT = colors.HexColor("#EEF3F7")

SECTION_EQUATIONS = {
    "1(i)": range(1, 8),
    "1(ii)": range(8, 10),
    "1(iii)": range(10, 14),
    "2(i)": range(14, 17),
    "2(ii)": range(8, 10),
    "2(iii)": range(17, 18),
    "3(i)": range(18, 24),
    "3(ii)": range(24, 26),
}


def _cover_overlay() -> PdfReader:
    """Create an overlay aligned with the first page of the supplied template."""
    stream = BytesIO()
    layer = canvas.Canvas(stream, pagesize=letter)
    layer.setFillColor(colors.white)
    layer.rect(215, 742, 45, 18, fill=True, stroke=False)
    layer.rect(168, 726, 115, 18, fill=True, stroke=False)
    layer.setFillColor(INK)
    layer.setFont("Helvetica-Bold", 11)
    layer.drawString(218, 746, GROUP_WORK_PROJECT)
    layer.setFont("Helvetica", 9)
    layer.drawString(171, 730, GROUP_NUMBER)

    row_y = [646, 622, 598]
    for (name, email), y in zip(PARTICIPANTS, row_y, strict=True):
        layer.setFont("Helvetica", 7.5 if len(name) > 25 else 8.5)
        layer.drawString(76, y, name)
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
            name="EquationLabel",
            parent=styles["BodyReport"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="EquationDescription",
            parent=styles["BodyReport"],
            fontSize=9,
            leading=12,
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


def _equation_references() -> list[tuple[int, str, str]]:
    """Extract each tagged equation and its parameter description from the notebook."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    description_pattern = re.compile(
        r"\*\*Parameters \(Equation(?:s)? (?P<start>\d+)"
        r"(?:-(?P<end>\d+))?\)\.\*\*\s*(?P<description>.*?)(?=\n\s*\n)",
        flags=re.DOTALL,
    )
    descriptions: dict[int, str] = {}
    for match in description_pattern.finditer(markdown):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        description = " ".join(match.group("description").split())
        descriptions.update({number: description for number in range(start, end + 1)})

    equation_pattern = re.compile(
        r"\$\$\s*(?P<latex>.*?)\s*"
        r"\\tag\*\{\$\\scriptstyle \((?P<number>\d+)\)\$\}\s*\$\$",
        flags=re.DOTALL,
    )
    references = [
        (
            int(match.group("number")),
            " ".join(match.group("latex").split()),
            descriptions[int(match.group("number"))],
        )
        for match in equation_pattern.finditer(markdown)
    ]
    if [number for number, _, _ in references] != list(range(1, 26)):
        raise ValueError("notebook must contain described equations numbered 1 through 25")
    return references


def _plain_description(description: str) -> str:
    """Convert inline mathematical markup into readable PDF text."""
    replacements = {
        r"\widehat{\mathbb{E}}": "estimated E",
        r"\mathbb{E}": "E",
        r"\mathbb{Q}": "Q",
        r"\operatorname{MSE}": "MSE",
        r"\operatorname{Re}": "Re",
        r"\sqrt{-1}": "sqrt(-1)",
        r"\Theta": "Theta",
        r"\kappa": "kappa",
        r"\theta": "theta",
        r"\sigma": "sigma",
        r"\rho": "rho",
        r"\varphi": "phi",
        r"\phi": "phi",
        r"\alpha": "alpha",
        r"\lambda": "lambda",
        r"\mu": "mu",
        r"\delta": "delta",
        r"\eta": "eta",
        r"\gamma": "gamma",
        r"\pi": "pi",
        r"\Delta": "Delta",
        r"\exp": "exp",
        r"\log": "log",
        r"\cdot": "dot",
    }
    text = description.replace("**", "").replace("$", "")
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = re.sub(r"\\(?:mathrm|text)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace(r"\!", "").replace(r"\,", " ")
    return escape(text.replace("{", "").replace("}", ""))


def _equation_flowable(
    number: int,
    latex: str,
    styles,
) -> list:
    stream = BytesIO()
    math_to_image(f"${latex}$", stream, dpi=180, format="png", color="#26323D")
    stream.seek(0)
    equation = Image(stream)
    scale = min(
        72 / 180,
        (6.4 * inch) / equation.imageWidth,
        (1.05 * inch) / equation.imageHeight,
    )
    equation.drawWidth = equation.imageWidth * scale
    equation.drawHeight = equation.imageHeight * scale
    equation.hAlign = "CENTER"
    return [
        Paragraph(f"Equation {number}", styles["EquationLabel"]),
        equation,
        Spacer(1, 5),
    ]


def _section_equation_story(section: str, styles) -> list:
    references = {
        number: (latex, description) for number, latex, description in _equation_references()
    }
    groups: list[list[tuple[int, str, str]]] = []
    for number in SECTION_EQUATIONS[section]:
        latex, description = references[number]
        if not groups or groups[-1][-1][2] != description:
            groups.append([])
        groups[-1].append((number, latex, description))

    story = []
    for group in groups:
        numbers = [number for number, _, _ in group]
        equation_word = "Equation" if len(numbers) == 1 else "Equations"
        number_label = str(numbers[0]) if len(numbers) == 1 else f"{numbers[0]}-{numbers[-1]}"
        flowables = [
            Paragraph(
                f"<b>Parameters ({equation_word} {number_label}).</b> "
                f"{_plain_description(group[0][2])}",
                styles["EquationDescription"],
            )
        ]
        for number, latex, _ in group:
            flowables.extend(_equation_flowable(number, latex, styles))
        story.append(KeepTogether(flowables))
    return story


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
    pending_equation_section = None
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
            pending_equation_section = None
            if story:
                story.append(PageBreak())
            story.append(Paragraph(escape(line[3:]), styles["Section"]))
            continue
        if line.startswith("### "):
            if line.startswith("### 3(ii)"):
                story.append(PageBreak())
            story.append(Paragraph(escape(line[4:]), styles["Question"]))
            section_match = re.match(r"### (?P<section>[123]\([^)]+\))", line)
            pending_equation_section = (
                section_match.group("section")
                if section_match and section_match.group("section") in SECTION_EQUATIONS
                else None
            )
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
        if pending_equation_section:
            story.extend(_section_equation_story(pending_equation_section, styles))
            pending_equation_section = None
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
