"""Tests for setup.py's low-RAM messages and the paging question (#250): an explicit --low-ram off installs with
--yes instead of stopping at "Install it anyway?" (default n).  Mocked input - no GPU, no downloads.

    python -m unittest tools.test_setup_lowram
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(fn, *args, stdin=None):
    """-> (exit code or None, printed text); `stdin` answers input() (None: never asked)."""
    out = io.StringIO()
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        if stdin is None:
            raise AssertionError(f"asked {prompt!r} with --yes")
        return stdin

    code = None
    with contextlib.redirect_stdout(out), mock.patch("builtins.input", fake_input):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), asked


class PagingQuestion(unittest.TestCase):
    MODEL, RAM = "IQ3_XXS", 46                                       # 60 GB wanted, 46 here

    def test_yes_with_low_ram_off_installs(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "off", True)
        self.assertIsNone(code, out)
        self.assertEqual(asked, [])
        self.assertIn("installing IQ3_XXS with 46 GB of RAM, as you chose (--low-ram off)", out)

    def test_yes_with_auto_still_stops(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "auto", True)
        self.assertEqual(code, 1)
        self.assertIn("IQ3_XXS needs about 60 GB of RAM; this PC has 46 GB", out)
        self.assertIn("--low-ram off --yes", out)                    # the way to insist

    def test_interactive(self):
        for choice, answer, want in (("off", "", None), ("auto", "", 1), ("auto", "y", None), ("off", "n", 1)):
            with self.subTest(choice=choice, answer=answer):
                code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, choice, False, stdin=answer)
                self.assertEqual(code, want, out)
                self.assertEqual(len(asked), 1)
                self.assertIn("[y]" if choice == "off" else "[n]", asked[0])


if __name__ == "__main__":
    unittest.main()
