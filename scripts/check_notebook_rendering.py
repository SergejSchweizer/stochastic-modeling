"""Browser-level validation of the generated notebook HTML."""

from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs" / "MScFE622_GWP1.html"
EXPECTED_EQUATIONS = [f"({number})" for number in range(1, 26)]
EXPECTED_PLOT_DESCRIPTIONS = 9
EQUATION_LABEL_POINTS_IN_PIXELS = 8 * 96 / 72
PLOT_DESCRIPTION_POINTS_IN_PIXELS = 8 * 96 / 72
NORMAL_TEXT_POINTS_IN_PIXELS = 10 * 96 / 72


def _launch_browser(playwright):
    """Use project Chromium when installed, with system Edge as a Windows fallback."""
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError:
        return playwright.chromium.launch(channel="msedge", headless=True)


def _font_size(locator) -> float:
    value = locator.evaluate("element => getComputedStyle(element).fontSize")
    return float(value.removesuffix("px"))


def _all_have_font_size(locator, expected: float, tolerance: float = 0.05) -> bool:
    return all(
        abs(_font_size(locator.nth(index)) - expected) < tolerance
        for index in range(locator.count())
    )


def _all_have_normal_typography(locator, allowed_weights: tuple[str, ...] = ("400",)) -> bool:
    return all(
        locator.nth(index).evaluate(
            """(element, allowedWeights) => {
                const style = getComputedStyle(element);
                return style.fontFamily === 'Arial, Calibri, sans-serif'
                    && style.fontStyle === 'normal'
                    && allowedWeights.includes(style.fontWeight)
                    && Math.abs(parseFloat(style.fontSize) - 13.3333333333) < 0.05
                    && Math.abs(parseFloat(style.lineHeight) - 20.6666666667) < 0.05;
            }""",
            list(allowed_weights),
        )
        for index in range(locator.count())
    )


def main() -> None:
    if not HTML.exists():
        raise FileNotFoundError(HTML)

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(HTML.as_uri(), wait_until="networkidle", timeout=120_000)
        page.wait_for_selector(".MJXc-display", timeout=30_000)

        equations = page.locator(".MJXc-display")
        labels = page.locator('span.mjx-mtd[id^="mjx-eqn-"]')
        label_glyphs = labels.locator(".mjx-mstyle > .mjx-mrow")
        errors = page.locator(".MathJax_Error, mjx-merror")
        descriptions = page.locator(".plot-description")
        paragraphs = page.locator(".jp-RenderedMarkdown p:not(.plot-description)")
        lists = page.locator(".jp-RenderedMarkdown ul, .jp-RenderedMarkdown ol")
        list_items = page.locator(".jp-RenderedMarkdown li")
        headings = page.locator(
            ".jp-RenderedMarkdown h1, .jp-RenderedMarkdown h2, .jp-RenderedMarkdown h3, .jp-RenderedMarkdown h4"
        )
        tables = page.locator(".jp-RenderedHTMLCommon table")
        table_cells = page.locator(".jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td")

        assert equations.count() == len(EXPECTED_EQUATIONS)
        assert labels.all_text_contents() == EXPECTED_EQUATIONS
        assert label_glyphs.count() == len(EXPECTED_EQUATIONS)
        assert errors.count() == 0
        assert descriptions.count() == EXPECTED_PLOT_DESCRIPTIONS
        assert all(
            text.startswith(f"Plot {index}.")
            for index, text in enumerate(descriptions.all_text_contents(), start=1)
        )
        images = page.locator("img")
        assert images.count() == EXPECTED_PLOT_DESCRIPTIONS
        assert all(images.nth(index).get_attribute("alt") for index in range(images.count()))
        equation_label_tolerance = 0.25 * 96 / 72
        assert _all_have_font_size(
            label_glyphs,
            EQUATION_LABEL_POINTS_IN_PIXELS,
            tolerance=equation_label_tolerance,
        )
        assert _all_have_font_size(descriptions, PLOT_DESCRIPTION_POINTS_IN_PIXELS)
        assert all(
            descriptions.nth(index).evaluate("element => getComputedStyle(element).textAlign")
            == "center"
            for index in range(descriptions.count())
        )
        assert all(
            descriptions.nth(index).evaluate("element => getComputedStyle(element).textAlignLast")
            == "center"
            for index in range(descriptions.count())
        )
        assert paragraphs.count() > 0
        assert _all_have_normal_typography(paragraphs)
        assert list_items.count() > 0
        assert _all_have_normal_typography(list_items)
        assert lists.count() > 0
        assert all(
            lists.nth(index).evaluate("element => getComputedStyle(element).textAlign") == "center"
            for index in range(lists.count())
        )
        assert all(
            list_items.nth(index).evaluate("element => getComputedStyle(element).textAlign")
            == "center"
            for index in range(list_items.count())
        )
        assert all(
            list_items.nth(index).evaluate("element => getComputedStyle(element).listStylePosition")
            == "inside"
            for index in range(list_items.count())
        )
        assert all(
            paragraphs.nth(index).evaluate("element => getComputedStyle(element).textAlign")
            == "justify"
            for index in range(paragraphs.count())
        )
        assert all(
            paragraphs.nth(index).evaluate("element => getComputedStyle(element).textAlignLast")
            == "center"
            for index in range(paragraphs.count())
        )
        assert headings.count() > 0
        assert all(
            headings.nth(index).evaluate("element => getComputedStyle(element).textAlign")
            == "center"
            for index in range(headings.count())
        )
        assert tables.count() > 0
        assert all(
            tables.nth(index).evaluate("element => getComputedStyle(element).textAlign") == "center"
            for index in range(tables.count())
        )
        assert table_cells.count() > 0
        assert _all_have_normal_typography(table_cells, allowed_weights=("400", "700"))
        assert all(
            table_cells.nth(index).evaluate("element => getComputedStyle(element).textAlign")
            == "center"
            for index in range(table_cells.count())
        )
        assert r"\tag{" not in page.locator("body").inner_text()
        browser.close()

    print(
        "Browser rendering checks passed: centered headings and tables, justified text, and "
        "8 pt annotations."
    )


if __name__ == "__main__":
    main()
