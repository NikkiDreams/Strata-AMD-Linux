"""#1594 as it stands in this fork: --gpus when starting an installed model names the AMD cards as setup lists them
(setup's KFD order, docs/AMD_HIP.md), since AMD is the only path here.  The numbers themselves are checked by
start_gpus; a card that cannot be used is named by amd_parse_gpus, which start() calls for the saved list.

    python -m unittest tools.test_setup_start_gpus_amd -v
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


def amd(index, arch="gfx1201", vram=32.0):
    return {"index": index, "name": f"Radeon {index}", "arch": arch, "vram_gb": vram}


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class StartGpusAmd(unittest.TestCase):
    def start(self, text, cards):
        with mock.patch.object(setup, "amd_gpus", return_value=list(cards)):
            return quiet(setup.start_gpus, text)[0]

    def test_amd_numbers_are_taken_as_setup_lists_them(self):
        self.assertEqual(self.start("1,0", [amd(0), amd(1)]), [1, 0])

    def test_all_still_means_every_supported_amd_card(self):
        self.assertEqual(sorted(self.start("all", [amd(0), amd(1)])), [0, 1])

    def test_all_orders_the_cards_as_they_share_the_model(self):
        # the most VRAM first (amd_rank), not the order the cards were named in
        self.assertEqual(self.start("all", [amd(0, vram=16.0), amd(1, vram=24.0)]), [1, 0])

    def test_all_needs_two_cards_strata_can_use(self):
        with self.assertRaises(SystemExit):
            self.start("all", [amd(0), amd(1, "gfx1010")])

    def test_a_card_that_cannot_be_used_is_named(self):
        # start_gpus only reads the numbers; the cards themselves are checked when the start uses them
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            setup.amd_parse_gpus("1,0", [amd(0), amd(1, "gfx1010")])
        self.assertIn("cannot be used", out.getvalue())
        self.assertIn("gfx1010", out.getvalue())

    def test_one_card_is_not_a_split(self):
        with self.assertRaises(SystemExit):
            self.start("0", [amd(0), amd(1)])

    def test_nothing_named_is_none(self):
        self.assertIsNone(self.start("", [amd(0)]))


if __name__ == "__main__":
    unittest.main()
