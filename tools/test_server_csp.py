"""The web app's Content-Security-Policy (serve/server.py): the pages load only their own files, so 'self' is
enough, and the policy must still allow what the app actually uses - its one inline script, the inline style
attributes of its progress bars, the images its file picker reads as data: URLs.  Nothing is served: the policy is
read from the handler's headers with a fake service, and the app's own files are checked against it.

    python -m unittest tools.test_server_csp
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as S  # noqa: E402

WEB = ROOT / "serve" / "web"
PAGES = ("index.html", "monitor.html")


def headers_for(path, api_monitor=True) -> str:
    """The response headers the handler writes for one GET, without a socket: only the HTML pages are reached this
    way, and that branch touches nothing but the file and the headers."""
    svc = SimpleNamespace(api_key="", api_monitor=api_monitor, host_names=[], allowed_hosts=[])
    handler = S.make_handler(svc)
    h = handler.__new__(handler)
    h.command, h.path, h.headers = "GET", path, {}
    h.requestline, h.request_version = f"GET {path} HTTP/1.0", "HTTP/1.0"
    h.client_address, h.wfile, h.record = ("127.0.0.1", 0), io.BytesIO(), None
    with contextlib.redirect_stderr(io.StringIO()):        # the access log line
        h.do_GET()
    return h.wfile.getvalue().decode("utf-8", "replace")


def policy_directives(policy: str) -> dict:
    parts = [p.strip() for p in policy.split(";") if p.strip()]
    return {p.split(" ", 1)[0]: p.split(" ", 1)[1] for p in parts}


class PolicyOnThePages(unittest.TestCase):
    def test_the_pages_carry_it(self):
        for path in ("/", "/api-monitor"):
            head = headers_for(path)
            self.assertIn("Content-Security-Policy:", head)
            self.assertIn(S.CSP, head)

    def test_the_monitor_page_is_a_page_too(self):
        # the same policy on both, so neither can be framed or load a script from elsewhere
        self.assertIn("frame-ancestors 'none'", headers_for("/api-monitor"))

    def test_only_the_page_itself_is_allowed(self):
        d = policy_directives(S.CSP)
        for name, sources in d.items():
            if name in ("script-src", "style-src", "img-src"):
                continue                                   # the app's own inline parts, checked below
            self.assertIn(sources, ("'self'", "'none'"), name)
        self.assertEqual(d["object-src"], "'none'")         # nothing may be embedded in the page
        self.assertEqual(d["frame-ancestors"], "'none'")    # and the page cannot be put in a frame

    def test_the_inline_parts_the_app_needs_are_allowed(self):
        d = policy_directives(S.CSP)
        self.assertIn("'unsafe-inline'", d["script-src"])          # index.html's theme before the first paint
        self.assertIn("'unsafe-inline'", d["style-src"])           # the progress bars' style="width:0%"
        self.assertIn("data:", d["img-src"])                       # app.js: readAsDataURL for picked images
        self.assertIn("blob:", d["img-src"])                       # app.js: the download link it makes

    def test_the_app_loads_nothing_outside_the_page(self):
        for page in PAGES:
            html = (WEB / page).read_text(encoding="utf-8")
            for tag in re.findall(r"<(?:script|link|img)\b[^>]*>", html):
                for ref in re.findall(r'(?:src|href)="([^"]+)"', tag):
                    self.assertTrue(ref.startswith(("web/", "/web/")), f"{page}: {ref}")
        for css in WEB.glob("*.css"):
            for url in re.findall(r'url\("([^"]+)"\)', css.read_text(encoding="utf-8")):
                self.assertFalse(url.startswith(("http:", "data:")), f"{css.name}: {url}")

    def test_no_inline_handler_is_used_so_the_policy_keeps_scripts_out(self):
        for f in list(WEB.glob("*.html")) + list(WEB.glob("*.js")):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("onclick=", text, f.name)
            self.assertNotIn("onerror=", text, f.name)


if __name__ == "__main__":
    unittest.main()
