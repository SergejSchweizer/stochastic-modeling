"""Browser-level validation of the generated notebook HTML."""

from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs" / "MScFE622_GWP1.html"
EXPECTED_EQUATIONS = [f"({number})" for number in range(1, 26)]
EXPECTED_PLOT_DESCRIPTIONS = 8
EIGHT_POINTS_IN_PIXELS = 8 * 96 / 72


def _launch_browser(playwright):
    """Use project Chromium when installed, with system Edge as a Windows fallback."""
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError:
        return playwright.chromium.launch(channel="msedge", headless=True)


def _font_size(locator) -> float:
    value = locator.evaluate("element => getComputedStyle(element).fontSize")
    return float(value.removesuffix("px"))


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
        errors = page.locator(".MathJax_Error, mjx-merror")
        descriptions = page.locator(".plot-description")

        assert equations.count() == len(EXPECTED_EQUATIONS)
        assert labels.all_text_contents() == EXPECTED_EQUATIONS
        assert errors.count() == 0
        assert descriptions.count() == EXPECTED_PLOT_DESCRIPTIONS
        images = page.locator("img")
        assert images.count() == EXPECTED_PLOT_DESCRIPTIONS
        assert all(images.nth(index).get_attribute("alt") for index in range(images.count()))
        assert all(
            abs(_font_size(labels.nth(index)) - EIGHT_POINTS_IN_PIXELS) < 0.05
            for index in range(labels.count())
        )
        assert all(
            abs(_font_size(descriptions.nth(index)) - EIGHT_POINTS_IN_PIXELS) < 0.05
            for index in range(descriptions.count())
        )
        assert r"\tag{" not in page.locator("body").inner_text()
        browser.close()

    print("Browser rendering checks passed: 25 equations and 8 plot descriptions at 8 pt.")


if __name__ == "__main__":
    main()
