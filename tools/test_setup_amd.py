"""Tests for setup.py's AMD card detection on a mocked KFD topology (/sys/class/kfd + /sys/class/drm): the arch
names from gfx_target_version, the CPU node skipped, HIP numbering, product names, which cards are supported and
which TheRock index each family installs from.  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_amd
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def fake_sysfs(root: Path, nodes: list) -> None:
    """nodes: (gfx_target_version, simd_count, render_minor, product_name or None, vram_bytes)"""
    for i, (ver, simd, minor, name, vram) in enumerate(nodes):
        n = root / "class/kfd/kfd/topology/nodes" / str(i)
        n.mkdir(parents=True)
        (n / "properties").write_text(f"cpu_cores_count {0 if simd else 12}\nsimd_count {simd}\n"
                                      f"gfx_target_version {ver}\ndrm_render_minor {minor}\n")
        if simd:
            d = root / f"class/drm/renderD{minor}/device"
            d.mkdir(parents=True)
            (d / "mem_info_vram_total").write_text(str(vram))
            if name is not None:
                (d / "product_name").write_text(name + "\n")


class KfdDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_family(self):
        fake_sysfs(self.root, [
            (0, 0, 0, None, 0),                                   # the CPU node: skipped
            (110001, 120, 128, "", 16 << 30),                     # gfx1101 without a product name
            (120000, 64, 129, None, 16 << 30),                    # gfx1200, no product_name file
            (120001, 128, 130, "AMD Radeon AI PRO R9700", 32 << 30),
            (110002, 64, 131, None, 8 << 30),                     # gfx1102: supported, unvalidated (#938)
            (100306, 4, 132, None, 512 << 20),                    # an integrated gfx1036: listed, not supported
            (110000, 192, 133, "Radeon RX 7900 XTX", 24 << 30),
        ])
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([x["arch"] for x in g], ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1036", "gfx1100"])
        self.assertEqual([x["index"] for x in g], [0, 1, 2, 3, 4, 5])          # HIP numbers: GPU nodes only
        self.assertEqual(g[0]["name"], setup.AMD_NAMES["gfx1101"])
        self.assertEqual(g[1]["name"], setup.AMD_NAMES["gfx1200"])
        self.assertEqual(g[2]["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(g[3]["name"], setup.AMD_NAMES["gfx1102"])
        self.assertAlmostEqual(g[2]["vram_gb"], 32.0)
        ok = [x["arch"] for x in g if setup.amd_problem(x) is None]
        self.assertEqual(ok, ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1100"])
        self.assertIn("gfx1036", setup.amd_problem(g[4]))

    def test_no_kfd(self):
        self.assertEqual(setup.amd_gpus(str(self.root)), [])

    def test_rocm_index_per_family(self):
        for arch in setup.AMD_ARCHS:
            self.assertIn(arch, setup.ROCM_INDEXES)
        self.assertTrue(setup.ROCM_INDEXES["gfx1101"].endswith("/gfx110X-dgpu/"))
        self.assertTrue(setup.ROCM_INDEXES["gfx1200"].endswith("/gfx120X-all/"))
        self.assertEqual(setup.ROCM_INDEXES["gfx1101"], setup.ROCM_INDEXES["gfx1100"])
        self.assertEqual(setup.ROCM_INDEXES["gfx1200"], setup.ROCM_INDEXES["gfx1201"])
        # #524: the RX 6700 XT (gfx1031) takes the RDNA2 wheels, as the RX 6800 / 6900 (gfx1030)
        self.assertEqual(setup.ROCM_INDEXES["gfx1031"], setup.ROCM_INDEXES["gfx1030"])
        self.assertIsNone(setup.amd_problem({"arch": "gfx1031"}))


class GpuLists(unittest.TestCase):
    """--gpus with AMD cards: every chosen card must be supported; the first is the main one."""
    AMD = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "arch": "gfx1201"},
           {"index": 1, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"},
           {"index": 2, "name": "AMD Radeon (gfx1036)", "vram_gb": 0.5, "arch": "gfx1036"},
           {"index": 3, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0, "arch": "gfx1100"}]

    def setUp(self):
        self.say = setup.say
        setup.say = lambda *a, **k: None

    def tearDown(self):
        setup.say = self.say

    def test_list_in_order(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("1,0", self.AMD)], [1, 0])
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus(" 0, 3 ", self.AMD)], [0, 3])

    def test_all_is_every_supported_card_most_vram_first(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("all", self.AMD)], [1, 3, 0])

    def test_refusals(self):
        for text in ("1,2", "1,7", "1", "1,1", "x,y"):
            with self.assertRaises(SystemExit, msg=text):
                setup.amd_parse_gpus(text, self.AMD)

    def test_wheels_hold_one_family(self):
        with tempfile.TemporaryDirectory() as d:        # no system ROCm there: the wheels would be needed
            old = setup.os.environ.get("ROCM_PATH")
            setup.os.environ["ROCM_PATH"] = d
            try:
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
            finally:
                if old is None:
                    del setup.os.environ["ROCM_PATH"]
                else:
                    setup.os.environ["ROCM_PATH"] = old

    def test_runtime_only_system_rocm_falls_back_to_the_wheels(self):
        """#446: a system ROCm 7 with hipcc and libhipblas but no HIP development files (no hip-lang CMake package, no
        hip_runtime.h) is not used for the build: a warning says what is missing and the wheels path follows (here
        it stops at the two-family check, which only the wheels path makes); with the files it is used as before."""
        said = []
        setup.say = lambda msg="": said.append(msg)          # tearDown puts the real one back
        with tempfile.TemporaryDirectory() as d:
            sysroot = Path(d)
            for rel, text in (("bin/hipcc", ""), ("lib/libhipblas.so.3", ""),
                              ("include/rocm-core/rocm_version.h",
                               "#define ROCM_VERSION_MAJOR 7\n#define ROCM_VERSION_MINOR 14\n")):
                (sysroot / rel).parent.mkdir(parents=True, exist_ok=True)
                (sysroot / rel).write_text(text)
            with mock.patch.dict(setup.os.environ, {"ROCM_PATH": d}):
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
                self.assertIn(f"the ROCm in {sysroot} has no HIP development files (lib/cmake/hip-lang/hip-lang-"
                              "config.cmake, include/hip/hip_runtime.h): using AMD's wheels", "\n".join(said))
                self.assertIn("two GPU families", "\n".join(said))
                for lib in ("lib64", "lib"):                 # either place CMake looks
                    with self.subTest(lib=lib):
                        cfg = sysroot / lib / "cmake/hip-lang/hip-lang-config.cmake"
                        cfg.parent.mkdir(parents=True, exist_ok=True)
                        cfg.write_text("")
                        said.clear()
                        with self.assertRaises(SystemExit):
                            setup.rocm_root(["gfx1100", "gfx1201"])
                        self.assertIn("(include/hip/hip_runtime.h)", "\n".join(said))
                        (sysroot / "include/hip").mkdir(parents=True, exist_ok=True)
                        (sysroot / "include/hip/hip_runtime.h").write_text("")
                        self.assertEqual(setup.rocm_root(["gfx1100", "gfx1201"]), (sysroot, [str(sysroot / "lib")]))
                        cfg.unlink()
                        (sysroot / "include/hip/hip_runtime.h").unlink()

    def test_build_for_every_arch(self):
        """build_engine_hip compiles for the set of the chosen cards' archs and records it in BUILD.json."""
        calls = {}
        saved = {k: getattr(setup, k) for k in ("ROOT", "rocm_root", "cmake_build", "source_hash", "source_version",
                                                "ok", "shutil")}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def fake_build(src, bdir, target, defs):
                calls["defs"] = defs
                (bdir).mkdir(parents=True, exist_ok=True)
                (bdir / setup.EXE).write_text("engine")

            class Sh:
                which = staticmethod(lambda name: "/usr/bin/" + name)
                copy2 = staticmethod(lambda a, b: Path(b).write_text(Path(a).read_text()))
            setup.ROOT, setup.cmake_build, setup.ok, setup.shutil = root, fake_build, lambda *a: None, Sh
            setup.source_hash, setup.source_version = (lambda *a: "h"), (lambda: "0.1.31")
            setup.rocm_root = lambda archs: (calls.setdefault("archs", archs) and root, [str(root / "lib")])
            try:
                setup.build_engine_hip({"arch": "gfx1201", "archs": ["gfx1201", "gfx1100", "gfx1201"]}, root)
                self.assertEqual(calls["archs"], ["gfx1100", "gfx1201"])
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201", calls["defs"])
                import json
                self.assertEqual(json.loads((root / "engine" / "BUILD.json").read_text())["archs"],
                                 ["gfx1100", "gfx1201"])
                calls.clear()                            # one of those cards alone: already built, no compile
                setup.build_engine_hip({"arch": "gfx1100"}, root)
                self.assertNotIn("defs", calls)
                setup.build_engine_hip({"arch": "gfx1101"}, root)   # another arch: compiled again
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1101", calls["defs"])
            finally:
                for k, v in saved.items():
                    setattr(setup, k, v)


class CalibrationKey(unittest.TestCase):
    """#566: a calibration is saved and found again per PC and model (hardware_key).  A config's cards
    are the AMD cards, numbered as HIP numbers them (setup's KFD order)."""
    AMD = [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 23.98, "arch": "gfx1100"},
           {"index": 1, "name": "AMD Radeon RX 7900 XT", "vram_gb": 19.98, "arch": "gfx1100"},
           {"index": 2, "name": "AMD Radeon (gfx1201)", "vram_gb": 15.92, "arch": "gfx1201"},
           {"index": 3, "name": "AMD Radeon (gfx1036)", "vram_gb": 0.5, "arch": "gfx1036"}]

    def setUp(self):
        self.patches = [mock.patch.object(setup, "amd_gpus", lambda *a, **k: [dict(g) for g in self.AMD]),
                        mock.patch.object(setup, "cpu_info", lambda: ("AMD Ryzen 9 7950X", 16)),
                        mock.patch.object(setup, "ram_gb", lambda: 63.6)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    @staticmethod
    def cfg(gpu, backend="hip", ctx="32768"):
        c = {"model_name": "qwen3.8-flash-next-iq3_xxs", "args": ["--max-context", ctx, "--kv", "int8"], "gpu": gpu}
        if backend:
            c["backend"] = backend
        return c

    def test_hip_key_names_the_amd_card_and_its_arch(self):
        self.assertEqual(setup.hardware_key(self.cfg(0)),
                         "AMD Radeon RX 7900 XTX (gfx1100)|24GB|AMD Ryzen 9 7950X|64GB|qwen3.8-flash-next-iq3_xxs|"
                         "32768|text")
        # the arch is not repeated when the name already holds it (no product name in sysfs)
        self.assertTrue(setup.hardware_key(self.cfg(2)).startswith("AMD Radeon (gfx1201)|16GB|"))

    def test_two_cards_of_one_arch_are_two_keys(self):
        keys = {setup.hardware_key(self.cfg(i)) for i in (0, 1, 2)}
        self.assertEqual(len(keys), 3)

    def test_a_split_names_every_card(self):
        self.assertTrue(setup.hardware_key(self.cfg([1, 0])).startswith(
            "AMD Radeon RX 7900 XT (gfx1100) + AMD Radeon RX 7900 XTX (gfx1100)|44GB|"))

    def test_no_gpu_in_the_config_is_the_supported_card_with_the_most_vram(self):
        self.assertTrue(setup.hardware_key(self.cfg(None)).startswith("AMD Radeon RX 7900 XTX (gfx1100)|24GB|"))

    def test_a_card_that_is_gone_is_unknown(self):
        self.assertTrue(setup.hardware_key(self.cfg(7)).startswith("?|0GB|"))

    def test_setup_reuses_a_saved_calibration(self):
        cfg = self.cfg(1)
        saved = {"settings": {"pcie_frac": 0.0}, "tok_s": 40.1, "date": "2026-10-03"}
        other = {"settings": {"pcie_frac": 0.5}, "tok_s": 50.0, "date": "2026-10-02"}
        store = {"calibration": {setup.hardware_key(cfg): saved, setup.hardware_key(self.cfg(0)): other}}
        with mock.patch.object(setup, "load_settings", lambda: store):
            self.assertEqual(setup.setup_calibration(cfg), saved)         # its own card's, not the XTX's
            self.assertIsNone(setup.setup_calibration(self.cfg(2)))       # never tuned on that card
if __name__ == "__main__":
    unittest.main()
