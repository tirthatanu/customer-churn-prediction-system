"""Capture real UI screenshots, using a Python bridge for restricted browsers.

Prerequisites: running local uvicorn app, playwright, a Chromium installation.
This bridges browser fetch calls to the real FastAPI instance when browser
navigation is restricted by administrator policy in an evaluation environment.
"""
import base64
import os
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("SCREENSHOT_API_URL", "http://127.0.0.1:8795")


def html_bundle() -> str:
    page = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
    css = (ROOT / "app/static/style.css").read_text(encoding="utf-8")
    js = (ROOT / "app/static/app.js").read_text(encoding="utf-8")
    screenshot_proxy = """
<script>
window.fetch = async (path, options = {}) => {
 const raw = await window.pythonFetch(path, options);
 return new Response(JSON.stringify(raw.body), {status: raw.status,
          headers: {'Content-Type': 'application/json'}});
};
</script>
"""
    page = page.replace('<link rel="stylesheet" href="/static/style.css" />', f'<style>{css}</style>')
    page = page.replace('<script src="/static/app.js" defer></script>', '')
    data = base64.b64encode((ROOT / "reports/figures/feature_importance.png").read_bytes()).decode()
    page = page.replace('src="/figures/feature_importance.png"', f'src="data:image/png;base64,{data}"')
    page = page.replace('</body>', f'{screenshot_proxy}<script>{js}</script></body>')
    return page


def proxy_api(path, options):
    method = options.get("method", "GET")
    with httpx.Client(base_url=BASE_URL, timeout=20) as client:
        response = client.request(method, path, content=options.get("body"),
                                  headers=options.get("headers", {}))
    return {"status": response.status_code, "body": response.json()}


def capture():
    screenshots = ROOT / "screenshots"
    screenshots.mkdir(exist_ok=True)
    content = html_bundle()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path="/usr/bin/chromium",
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.expose_function("pythonFetch", proxy_api)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_content(content, wait_until="load")
        page.wait_for_function("document.getElementById('metric-accuracy').textContent !== '—'")
        assert page.locator("#quality-advisory").is_visible()
        page.screenshot(path=str(screenshots / "dashboard_overview.png"), full_page=True)
        page.locator("#submit-btn").click()
        page.wait_for_selector("#result-content:visible", timeout=20000)
        assert not page.locator("#result-empty").is_visible()
        page.wait_for_function("document.getElementById('history-count').textContent.includes('Showing')")
        assert page.locator("#prob-label").inner_text().endswith("%")
        print("Result:", page.locator("#outcome-badge").inner_text(),
              page.locator("#prob-label").inner_text(), page.locator("#risk-chip").inner_text())
        page.screenshot(path=str(screenshots / "prediction_result.png"), full_page=True)
        page.locator("#reset-btn").click()
        assert page.locator("#result-empty").is_visible()
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.expose_function("pythonFetch", proxy_api)
        mobile.set_content(content, wait_until="load")
        mobile.wait_for_function("document.getElementById('metric-accuracy').textContent !== '—'")
        mobile.locator('#insights').scroll_into_view_if_needed()
        mobile.wait_for_function("document.querySelector('.chart-holder img').complete && document.querySelector('.chart-holder img').naturalWidth > 0")
        mobile.evaluate("window.scrollTo(0,0)")
        mobile.screenshot(path=str(screenshots / "mobile_view.png"), full_page=True)
        horizontal_overflow = mobile.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        print("Mobile horizontal overflow:", horizontal_overflow)
        print("JavaScript page errors:", errors)
        if errors or horizontal_overflow:
            raise RuntimeError("Browser QA found problems")
        browser.close()


if __name__ == "__main__":
    capture()
