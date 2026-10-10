"""Tests for setup.py's Unsloth choice (docs/UNSLOTH_Q4.md): UD-IQ4_XS - the three pinned shards with their sizes and
SHA-256, the RAM budget from the PC's RAM, the pack with --compat-bf16 (never experts.bin), the engine version it
needs, one GPU, the CPU image encoder.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_unsloth
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

M = "UD-IQ4_XS"
REV = "38bb39ee97821de2c9009abb7e93950eec396e66"
SOURCE_VER = ".".join(map(str, setup.source_ver()))   # the engine this source tree builds (ROOT is patched below)
CARD = {"index": 0, "name": "AMD Radeon RX 7600", "vram_gb": 11.9, "arch": "gfx1102"}


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Pins(unittest.TestCase):
    def test_the_three_shards(self):
        fam = setup.FAMILIES["unsloth"]
        names = [setup.model_file(fam, M, i) for i in range(1, setup.model_shards(fam) + 1)]
        self.assertEqual(names, list(setup.UNSLOTH_IQ4_XS_SHARDS))
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_IQ4_XS_SHARDS.values()), 93682584224)
        for n, (size, sha) in setup.UNSLOTH_IQ4_XS_SHARDS.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertEqual(fam["sha256"][n], (size, sha))
        self.assertEqual(fam["hf"].format(q=M),
                         f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-IQ4_XS/")
        self.assertEqual(setup.MODELS[M]["families"], ("unsloth",))    # no other family lists it
        self.assertFalse(setup.MODELS[M].get("experimental"))          # a regular choice since 0.1.39
        self.assertEqual(list(setup.FAMILIES)[0], "qwen")              # not the default choice

    def test_not_in_the_other_families(self):
        for f in ("qwen", "swift", "coder"):
            self.assertNotIn(M, [m for m in setup.MODELS if f in setup.MODELS[m].get("families", ("qwen", "swift"))])


class Budget(unittest.TestCase):
    def test_ram_less_24(self):
        self.assertEqual(setup.resident_budget_gib(M, 63.7), 40)       # this PC (64 GB): the measured setting
        self.assertEqual(setup.resident_budget_gib(M, 47.8), 24)
        self.assertEqual(setup.resident_budget_gib(M, 96), 55)         # at most all of them (59.5 GB = 55.4 GiB)
        self.assertEqual(setup.resident_budget_gib(M, 256), 55)
        self.assertEqual(setup.resident_budget_gib(M, 20), 8)          # the floor
        self.assertEqual(setup.resident_budget_gib(M, 63.7, 1.7), 38)  # a KV cache streamed to RAM comes out of it


class Sha256(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Path(self.tmp.name) / "x-00002-of-00003.gguf"
        self.f.write_bytes(b"strata" * 1000)
        self.sha = hashlib.sha256(self.f.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_file_marked_once(self):
        code, out = quiet(setup.verify_sha256, self.f, 6000, self.sha)
        self.assertIsNone(code, out)
        self.assertIn(f"sha256 {self.sha}", self.f.with_name(self.f.name + ".done").read_text())
        with mock.patch("hashlib.sha256", side_effect=AssertionError("hashed again")):
            code, out = quiet(setup.verify_sha256, self.f, 6000, self.sha)    # the mark says it was checked
        self.assertIsNone(code, out)

    def test_a_download_mark_alone_is_not_enough(self):
        setup.mark(self.f)                                             # the download's finish mark (a date)
        with mock.patch("hashlib.sha256", wraps=hashlib.sha256) as h:
            code, out = quiet(setup.verify_sha256, self.f, 6000, self.sha)
        self.assertIsNone(code, out)
        self.assertTrue(h.called)

    def test_wrong_hash_deletes_it(self):
        setup.mark(self.f)
        code, out = quiet(setup.verify_sha256, self.f, 6000, "0" * 64)
        self.assertEqual(code, 1)
        self.assertIn("wrong SHA-256", out)
        self.assertFalse(self.f.exists())
        self.assertFalse(self.f.with_name(self.f.name + ".done").exists())   # downloaded again next time

    def test_wrong_size(self):
        code, out = quiet(setup.verify_sha256, self.f, 6001, self.sha)
        self.assertEqual(code, 1)
        self.assertIn("6,000 bytes, not 6,001", out)
        self.assertTrue(self.f.exists())


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            for name, family, model in (("strata-unsloth-ud-iq4_xs.json", "unsloth", M),
                                        ("strata-swift-iq3_xxs.json", "swift", "IQ3_XXS"),
                                        ("strata-iq3_s.json", "qwen", "IQ3_S"),
                                        ("strata-coder-iq1_m.json", "coder", "IQ1_M")):
                p = Path(t) / name
                p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), (family, model), name)


class FakeGGUF:
    """gguf_reader.GGUFFile: shard 2 holds the PLE table."""
    def __init__(self, path):
        names = ["per_layer_token_embd.weight"] if "00002-of" in str(path) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


class Base(unittest.TestCase):
    """setup.main() for the Unsloth choice, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs, self.verified = [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, ram=63.7, version=SOURCE_VER, n_gpus=1, model=True, answers=None, family=True, free=500.0):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others), and every question
        asked is kept in self.asked.  family=False: no --family (the first menu is shown)."""
        self.asked = []

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        eng = self.t / "engine"

        def fake_build(gpu, llama, vision="none"):
            """build_engine_hip: the engine is always compiled here, so this is what its BUILD.json says."""
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local-hip", "backend": "hip", "version": version,
                                                         "archs": [gpu["arch"]], "vision": vision, "lib_dirs": []}),
                                            encoding="utf-8")
            return eng

        found = [{**CARD, "index": i} for i in range(n_gpus)]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "amd_gpus", lambda *a, **k: [dict(g) for g in found]),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),   # #642: not a hybrid CPU
            mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "rotational_disk", lambda p: None),   # #605: not the test PC's disk
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "build_engine_hip", fake_build),
            mock.patch.object(setup, "hipblaslt_table", lambda *a, **k: None),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"run-{tag}.sh"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", "unsloth"] if family else []),
                                            *(["--model", M] if model else []),
                                            *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = self.t / f"strata-unsloth-{M.lower()}.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)


class Unsloth(Base):
    """UD-IQ4_XS: the family's only model, a regular choice."""

    def test_install(self):
        code, out, cfg = self.main(["--context", "8192"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("EXPERIMENTAL", out)
        base = f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-IQ4_XS/"
        self.assertEqual(self.downloads, [base + n for n in setup.UNSLOTH_IQ4_XS_SHARDS])
        self.assertEqual([v[0] for v in self.verified], list(setup.UNSLOTH_IQ4_XS_SHARDS))
        self.assertEqual([v[1:] for v in self.verified], list(setup.UNSLOTH_IQ4_XS_SHARDS.values()))
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertIn("--compat-bf16", packs[0])
        self.assertNotIn("--experts-bin", packs[0])
        self.assertTrue(packs[0][packs[0].index("--gguf") + 1].endswith("UD-IQ4_XS-00001-of-00003.gguf"))
        args = cfg["args"]
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "40")
        self.assertTrue(args[args.index("--native") + 1].endswith("UD-IQ4_XS-00001-of-00003.gguf"))
        for flag in ("--ple-gguf", "--mmap-experts", "--resident-experts", "--vision", "--control-vector-scaled",
                     "--kv-resident"):
            self.assertNotIn(flag, args)
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-iq4_xs")
        self.assertNotIn("vision", cfg)
        pack = self.t / "data" / "packs" / "unsloth-ud-iq4_xs"                 # its own pack and tokenizer
        self.assertEqual(packs[0][packs[0].index("--out") + 1], str(pack))
        self.assertEqual(args[args.index("--pack") + 1], str(pack))
        self.assertEqual(cfg["tokenizer"], str(pack / "tokenizer"))

    def test_recommended_context_on_12_gb(self):
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "8192")

    def test_more_ram_bigger_budget(self):
        code, out, cfg = self.main(["--context", "8192"], ram=95.8)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "55")   # all of them

    def test_kv_streaming_comes_out_of_the_budget(self):
        code, out, cfg = self.main(["--context", "131072"])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertIn("--kv-resident", args)
        kv_gb = 131072 * 13 * 1056 / 1e9
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], str(40 - int(-(-kv_gb // 1))))

    def test_old_engine_stops_before_the_download(self):
        code, out, cfg = self.main(["--context", "8192"], version="0.1.37")
        self.assertEqual(code, 1)
        self.assertIn("needs engine 0.1.38 or newer; this one is 0.1.37", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)

    def test_too_little_ram(self):
        code, out, cfg = self.main(["--context", "8192"], ram=31.9, model=False)   # --yes alone: still a stop
        self.assertEqual(code, 1)
        self.assertIn("needs 48 GB of RAM or more", out)
        self.assertIn("--model UD-IQ4_XS --yes", out)       # the way to insist (the family's only size)
        self.assertEqual(self.downloads, [])

    def test_resumed_download_needs_room_for_the_rest_only(self):
        """#425: the disk check counts the finished shards and .part files already there (sizes scaled down: a
        2 MB "model" with 1.5 MB on the disk, 8 GB of other room needed, 8.001 GB free)."""
        d = self.t / "models" / "unsloth-UD-IQ4_XS"
        d.mkdir(parents=True)
        (d / (list(setup.UNSLOTH_IQ4_XS_SHARDS)[0] + ".part")).write_bytes(b"x" * 1_500_000)
        with mock.patch.dict(setup.MODELS[M], {"download_gb": 0.002}):
            code, out, cfg = self.main(["--context", "8192"], free=8.001)
        self.assertNotIn("not enough free disk space", out)
        self.assertEqual(code, 0, out)
        for f in d.iterdir():                                                         # the first run "downloaded" them
            f.unlink()
        shutil.rmtree(self.t / "data" / "mtp", ignore_errors=True)       # #897: a draft layer already there needs less room
        (d / (list(setup.UNSLOTH_IQ4_XS_SHARDS)[0] + ".part")).write_bytes(b"x" * 1_500_000)
        with mock.patch.dict(setup.MODELS[M], {"download_gb": 0.003}):              # 1.5 MB still missing: no room
            code, out, cfg = self.main(["--context", "8192"], free=8.001)
        self.assertEqual(code, 1)
        self.assertIn("not enough free disk space", out)

    def test_too_little_ram_with_an_explicit_model_installs(self):
        """S6 (the owner's rule): --model with --yes is the consent; the risk is said."""
        code, out, cfg = self.main(["--context", "8192"], ram=31.9)
        self.assertEqual(code, 0, out)
        self.assertIn("needs 48 GB of RAM or more; this PC has 32 GB", out)
        self.assertIn("installing UD-IQ4_XS with 32 GB of RAM, as you chose", out)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "8")

    def test_budget_flag(self):
        """S4: --resident-budget-gib N is kept; more than the recommendation says what it risks."""
        code, out, cfg = self.main(["--context", "8192", "--resident-budget-gib", "48"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "48")
        self.assertIn("more than setup recommends for this PC (40 GiB", out)
        self.assertIn("Kept as you chose", out)
        code, out, cfg = self.main(["--context", "8192", "--resident-budget-gib", "32.5"])
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "32.5")
        self.assertNotIn("more than setup recommends", out)

    def test_budget_flag_with_kv_streaming_is_kept(self):
        code, out, cfg = self.main(["--context", "131072", "--resident-budget-gib", "40"])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertIn("--kv-resident", args)
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "40")      # not 38: the user's number
        self.assertIn("come on top of your 40 GiB RAM budget", out)

    def test_budget_flag_needs_a_positive_number(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, out, cfg = self.main(["--context", "8192", "--resident-budget-gib", "0"])
        self.assertEqual(code, 2)                                                   # argparse: N > 0
        self.assertIsNone(cfg)

    def test_images_on_the_cpu(self):
        """#304: the AMD engine has no GPU image encoder, so `yes` says so and images stay off; --vision cpu turns
        the CPU encoder on, with the original model's encoder file."""
        code, out, cfg = self.main(["--context", "8192", "--vision", "yes"])
        self.assertEqual(code, 0, out)
        self.assertIn("no GPU image encoder yet", out)
        self.assertNotIn("vision", cfg)
        code, out, cfg = self.main(["--context", "8192", "--vision", "cpu"])
        self.assertEqual(code, 0, out)
        self.assertIn("images: on", out)
        self.assertEqual(self.downloads[-1], setup.hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
                         + "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        self.assertIn("--vision", cfg["args"])
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertTrue(cfg["vision"]["model"].endswith("UD-IQ4_XS-00001-of-00003.gguf"))
        self.assertIn("--resident-budget-gib", cfg["args"])

    def test_check_verdict(self):
        code, out, _ = self.main(["--check"])
        self.assertEqual(code, 0, out)
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(M))
        self.assertIn("fits with 40 GiB of its experts in RAM", line)
        self.assertNotIn("EXPERIMENTAL", line)

    def test_check_below_every_floor(self):
        # #977: 16 GB fits no size; the RAM line is not [ok] and the verdict says so
        code, out, _ = self.main(["--check"], ram=16.0)
        self.assertEqual(code, 1, out)
        self.assertNotIn("[ok] RAM", out)
        self.assertIn("[!]  RAM: 16 GB (less than", out)
        self.assertIn("This PC cannot run Strata yet", out)
        self.assertNotIn("This PC can run Strata", out)

    def test_yes_alone_starts_on_one_gpu(self):
        """#364: setup on AMD starts on one card; the RAM does not hold the model plus 24 GB, so the second card is
        not offered here."""
        code, out, cfg = self.main(["--context", "8192"], n_gpus=2, ram=165.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        self.assertNotIn("layer_split", cfg)

    def test_split_when_the_ram_holds_it(self):
        # #498: 165 GB and --gpus 0,1: the split without the RAM budget (the engine refuses the pair)
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1"], n_gpus=2, ram=165.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        for flag in ("--resident-budget-gib", "--mmap-experts", "--resident-experts"):
            self.assertNotIn(flag, cfg["args"])
        self.assertIn("as you chose (--gpus): no RAM budget", out)
        self.assertIn("OS file cache", out)
        self.assertNotIn("RAM budget: ", out)

    def test_split_without_the_ram_is_a_warning_not_a_wall(self):
        # #737: below the worst-case estimate (a 128 GB PC ran it) the split is still the user's call
        need = setup.unsloth_split_need_gb(M)
        self.assertAlmostEqual(need, setup.MODELS[M]["download_gb"] + setup.UNSLOTH_RAM_LEFT_GB)
        code, out, cfg = self.main(["--context", "8192"], n_gpus=2, ram=100.0)        # no --gpus: one card, the budget
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "55")   # all of them
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1"], n_gpus=2, ram=100.0)   # asked for by name
        self.assertEqual(code, 0, out)
        self.assertIn("worst case", out)
        self.assertIn("as you chose (--gpus), with 100 GB of RAM", out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--resident-budget-gib", cfg["args"])

    def test_an_explicit_budget_keeps_one_gpu(self):
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1", "--resident-budget-gib", "60"], n_gpus=2,
                                   ram=165.0)
        self.assertEqual(code, 0, out)
        self.assertIn("--resident-budget-gib has no layer split", out)
        self.assertNotIn("layer_split", cfg)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "60")


class LayerSplit(unittest.TestCase):
    """#498: UD-IQ4_XS across GPUs - setup on AMD starts on one card, --gpus shares it, and a start with --gpus no
    longer keeps the RAM budget the engine refuses with a split (it exited with code 2)."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from test_setup_golden import card
        self.found = [card(i, "AMD Radeon RX 7900 XTX", 24.0, "gfx1100") for i in range(2)]

    def test_setup_starts_on_one_card(self):
        from test_setup_golden import install
        argv = ["--family", "unsloth", "--model", M, "--context", "32768", "--no-start"]
        code, out, cfg, asked = install(165.0, self.found, argv)
        self.assertEqual(code, 0, out)
        self.assertEqual(asked, [])                                   # --yes: nothing asked
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, asked = install(165.0, self.found, argv + ["--gpus", "0,1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertNotIn("--resident-budget-gib", cfg["args"])

    def start(self, ram, gpu=(0, 1), offered=None):
        from test_setup_risk import run
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-unsloth-ud-iq4_xs.json"
            before = {"exe": str(exe), "args": ["--pack", "p", "--resident-budget-gib", "40", "--kv", "int8"],
                      "gpu": 0, "gpus_asked": offered is None}
            p.write_text(json.dumps(before))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "amd_gpus", lambda: self.found), \
                    mock.patch.object(setup, "ram_gb", lambda: ram), \
                    mock.patch.object(setup, "engine_runs_on", lambda g: True), \
                    mock.patch.object(setup, "engine_archs", lambda: ["gfx1100"]), \
                    mock.patch.object(setup.subprocess, "call", call):
                code, out, asked = run(setup.start, p, None, list(gpu) if gpu else None, False, offered is None,
                                       stdin=offered)
            return code, out, json.loads(p.read_text()), call.called

    def test_start_with_gpus_drops_the_budget(self):
        code, out, cfg, started = self.start(165.0)
        self.assertIsNone(code, out)
        self.assertTrue(started)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["args"], ["--pack", "p", "--kv", "int8"])   # #1447: no helper cache
        self.assertIn("no RAM budget", out)

    def test_start_with_gpus_without_the_ram_is_asked(self):
        # #737: a warning and a question (default no), not a wall; --gpus with --yes is the consent
        code, out, cfg, started = self.start(63.7, offered="n")          # asked: "n" stops, nothing saved
        self.assertEqual(code, 1)
        self.assertFalse(started)
        self.assertIn("cannot share its RAM budget across GPUs", out)
        self.assertIn("--gpu N", out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, started = self.start(63.7)                       # --gpus + --yes: goes on, no budget
        self.assertIsNone(code, out)
        self.assertTrue(started)
        self.assertIn("cannot share its RAM budget across GPUs", out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--resident-budget-gib", cfg["args"])

    def test_offered_at_a_start(self):
        code, out, cfg, _ = self.start(165.0, gpu=None, offered="")      # Enter: one GPU (the default here)
        self.assertIsNone(code, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, _ = self.start(165.0, gpu=None, offered="y")
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, _ = self.start(63.7, gpu=None, offered="y")      # not offered: the RAM does not hold it
        self.assertEqual(cfg["gpu"], 0)


if __name__ == "__main__":
    unittest.main()
