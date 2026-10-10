#!/usr/bin/env python3
"""Strata one-click setup and start (Linux, AMD GPUs).

    ./setup.sh      - it installs Python if needed and runs this file

The first time it asks which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune), which size, how much
context, and which KV cache - then installs everything and starts the model on http://127.0.0.1:8080 (OpenAI- and
Anthropic-compatible API; a small page there shows that it runs). Every later start skips straight to running the
model: nothing that is already downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: AMD GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ..., and AMD's ROCm wheels)
  4. compiles the Strata engine for your GPU (once, 10-20 minutes); a complete system ROCm 7 is used when it is
     there, else AMD's TheRock wheels are installed into .venv
  5. downloads the model from Hugging Face (resumable), and the image encoder's files if you ask for pictures
     (--vision cpu)
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.sh and starts the model

Runs on Linux with an AMD GPU only: on Windows, or on a PC whose card is not AMD, it stops with a clear message.

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S|IQ1_M|UD-IQ4_XS, --context 32768,
--rope-scaling none|linear|yarn (--rope-scale F; past the trained 262144 the setup adds yarn and the factor is the
final context over 262144, at least 1 - an explicit --rope-scaling none is refused for such a context),
--vision yes|no|cpu, --port 8080, --yes (recommended answers, no questions), --setup (install another model /
change settings instead of starting), --no-start, --host 0.0.0.0 --api-key KEY (reach it from other devices on
your network), --experimental-speed-projection on|off (EXPERIMENTAL, off by default), --models-dir DIR,
--gguf-dir DIR (use GGUF files you already have), --check (only check this PC), --resident-budget-gib N
(UD-IQ4_XS's experts in RAM), --kv-streaming on|off|auto.

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, more GPUs, a bigger RAM budget, a size it thinks will not fit - is kept, with
what it risks.  With --yes, an explicit flag (--model, --gpus, ...) is the consent to a risk setup would otherwise
stop at; --yes alone is not.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# #214: every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when this was pinned), so a checkout installs the same files on any
# day.  A revision the repository no longer has falls back to its current files, with a message (download()).
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "b22d729eae29b5796f76fb70f91aef549b9fc52c",   # 2026-09-24
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                   # 2026-09-30
}


HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


def hf_unpinned(url: str) -> str:
    """The same file at the repository's current revision (main)."""
    return re.sub(r"^(https?://[^/]+/.+?/resolve/)[0-9a-f]{40}/", r"\1main/", url, count=1)


# ModelScope (www.modelscope.cn) hosts every repository above under the same name, with the same file paths and
# sizes, and is reachable from mainland China where huggingface.co often is not.  It serves a repository's current
# files (no pinned revision), so a file from it is checked against the SHA-256 ModelScope publishes for it, and the
# MTP tensors against the pinned checkpoint's own hashes (tools/mtp_fetch.py).
# --source / STRATA_SOURCE: huggingface (what auto means), or modelscope: only when asked for (a different host
# serving current files, not our pinned revision: setup never switches to it by itself, it recommends it, #908).
MS_DEFAULT = "https://www.modelscope.cn"
SOURCES = ("auto", "modelscope", "huggingface")
HF_FILE = re.compile(r"^https?://[^/]+/(?P<repo>[^/]+/[^/]+)/resolve/[^/]+/(?P<path>.+)$")
_sources = {}                                      # model_source()'s answer per (STRATA_SOURCE, HF_ENDPOINT, host)
_ms_files = {}


def ms_endpoint() -> str:
    return (os.environ.get("MODELSCOPE_ENDPOINT") or "").strip().rstrip("/") or MS_DEFAULT


def reachable(url: str, timeout: float = 5.0) -> bool:
    """Whether `url` answers a HEAD request with success (2xx, after redirects) within `timeout` seconds."""
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"}),
                               timeout=timeout).close()
        return True
    except OSError:                                    # HTTPError too: an error page is not the file
        return False


def model_source() -> str:
    """"modelscope" or "huggingface".  ModelScope only when --source modelscope / STRATA_SOURCE=modelscope asks for it;
    everything else (auto, a HF_ENDPOINT mirror chosen on purpose, #495) is Hugging Face.  Setup does not switch hosts
    by itself (#908): ModelScope serves a repository's current files, not our pinned revision."""
    want = (os.environ.get("STRATA_SOURCE") or "auto").strip().lower()
    key = (want, os.environ.get("HF_ENDPOINT") or "", ms_endpoint())
    if key not in _sources:
        _sources[key] = "modelscope" if want in ("ms", "modelscope") else "huggingface"
    return _sources[key]


def source_hint(url: str) -> str:
    """The recommendation that goes with a failed download from Hugging Face: ModelScope, by an explicit choice."""
    if "huggingface" in url or "hf-mirror" in url:
        return ("; if huggingface.co does not reach you (mainland China), the same files are on ModelScope: run setup "
                "with --source modelscope (docs/INSTALL.md)")
    return ""


def ms_file(url: str):
    """(repo, path) when `url` is a Hugging Face file of a repository setup knows (HF_REVISIONS), else None."""
    m = HF_FILE.match(url)
    if m is None or m.group("repo") not in HF_REVISIONS:
        return None
    return m.group("repo"), m.group("path")


def ms_url(repo: str, path: str) -> str:
    return f"{ms_endpoint()}/models/{repo}/resolve/master/{path}"


def ms_meta(repo: str, path: str):
    """(size, sha256) ModelScope publishes for a file, or None when it cannot be asked."""
    if repo not in _ms_files:
        try:
            api = f"{ms_endpoint()}/api/v1/models/{repo}/repo/files?Recursive=true"
            with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "strata-setup"}),
                                        timeout=60) as r:
                files = json.loads(r.read())["Data"]["Files"]
            _ms_files[repo] = {f["Path"]: (int(f.get("Size") or 0), (f.get("Sha256") or "").lower()) for f in files}
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return _ms_files[repo].get(path)


HF = hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"
# The engine is compiled here for the AMD card (build_engine_hip): there is no ready-made engine in this fork.
# KV bytes per context token and attention layer: 8-bit 1056, rotated 4-bit 576, hybrid K8V4 (8-bit K, 4-bit V) 816
# KV bytes per context token and attention layer: 8-bit 1056, rotated 4-bit 576, hybrid K8V4 (8-bit K, 4-bit V) 816
KV_CELL_BYTES = {"q4_0": 576, "k8v4": 816}
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (#214)

MODELS = {
    # the original model only for now: Swift 1.5's Q2_0 files split one layer's experts across the two shards, which
    # the pack tool (tools/iq_pack.py) cannot prepare yet (#171)
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0, "families": ("qwen",)},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    # the original model only (Swift 1.5 has no IQ3_S): matches the full BF16 model on the published benchmarks
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen",)},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    # #621: Unsloth's UD-IQ4_XS - IQ3_S gate/up experts with IQ4_NL (43 layers) or Q8_0 (5) downs; three shards.
    # Its 59.5 GB of experts: a RAM budget of them, but far fewer read from the SSD on a 64 GB PC and none from
    # ~80 GB of RAM.  Images: the vision path has no restriction for this pack (the same base model and image
    # encoder), so setup asks.
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), the best quality here; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                   "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                   "engine": (0, 1, 38), "vision": True},
}
# UD-IQ4_XS's three shards at the pinned revision: name -> (bytes, sha256), checked after the download
# (setup trusts no other model file by name and size alone either: check_shards reads their directories).
UNSLOTH_IQ4_XS_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
}
UNSLOTH_ENGINE = (0, 1, 32)     # the first engine setup configures for Unsloth's packs (0.1.31 ran them by hand)
UNSLOTH_RAM_LEFT_GB = 24        # RAM beside the budget: the OS, the engine, and the file cache the rest is read through
# Contexts past 262144 (the model's trained length) extend it by rope scaling: for the context it will
# serve the setup resolves the method (yarn, or one question when interactive) and derives the factor
# from the final context (final / 262144, at least 1) itself (below), keeps an explicit
# --rope-scaling/--rope-scale, and refuses an explicit --rope-scaling none there - the stock angles past
# the trained range are out of spec.
# 204800 (200K) sits between 128K and 256K: it is inside the trained 262144, so it needs no rope scaling and
# costs ~2.8 GB of 8-bit KV with IQ3_S (vs ~3.6 GB at 256K) - a middle step for PCs that cannot hold 256K.
CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside coding",
              "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    # Unsloth's UD-IQ4_XS (three shards, #621; a regular choice from 0.1.39) of the original model
    # (docs/UNSLOTH_Q4.md)
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth's ~4-bit quantization (UD-IQ4_XS)",
                "about": "UD-IQ4_XS: a 94 GB download; with less than ~80 GB of RAM part of its experts are read from "
                         "the SSD",
                "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
                "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "shards": 3, "tag": "unsloth-",
                "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-unsloth",
                "pack_args": ["--compat-bf16"],
                "sha256": dict(UNSLOTH_IQ4_XS_SHARDS)},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# EXPERIMENTAL, off by default (setup asks): a control vector shipped with the repository, see its README
ESP_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
VISION = {"cpu": {"max_tokens": 300, "reserve_mib": 700}}   # the AMD engine has no GPU image encoder: the CPU one
EXE = "strata"
VEXE = "strata-vision"


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def flush_typed_ahead() -> None:
    """#841: keys pressed while setup was busy (a download, a build) are not answers to the next question: Enter pressed
    to "wake" a slow step answered "Go on anyway?" with its default.  Dropped before a question is asked, on a terminal
    only (a pipe or a test keeps its input)."""
    try:
        if not sys.stdin.isatty():
            return
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def ask(question, choices, default, yes):
    if yes:
        return default
    flush_typed_ahead()
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            fail("input ended before a setup answer was received",
                 "run setup in a terminal, or pass --yes to accept the recommended answers")
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    """Write the finish mark.  A folder that cannot be written (--gguf-dir on a read-only share, #570) only costs the
    mark: setup says so and goes on (the step is repeated on the next run), it does not stop."""
    try:
        path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    except OSError as e:
        warn(f"the finish mark of {path.name} cannot be written ({e}): the next run does this step again")


# ------------------------------------------------------------------------------------------------ the PC
def ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def cpu_cores():
    """#642: (performance cores, efficiency cores) of a hybrid CPU (Intel 12th gen+, AMD Zen 5 + Zen 5c), counted
    as the engine's pool counts them (detect_cpu_topology: physical cores, by the kernel's cpu_capacity); None on a
    CPU whose cores are all alike, or when the kernel does not say."""
    classes = linux_core_classes()
    if classes is None or max(classes) == min(classes):
        return None
    p = sum(1 for c in classes if c == max(classes))
    return p, len(classes) - p


def linux_core_classes(sys_root: str = "/sys"):
    """One class per physical core on Linux: 1 for a P-core, 0 for an E-core; None when the kernel does not say.
    #798: Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus) when they exist; else cpu_capacity, where an E-core is
    one under 90% of the largest - Turbo Boost Max 3.0 gives the "favored" P-cores a slightly higher capacity (1024
    against 1012), so "below the maximum" counted 2 of a Core Ultra 7 270K Plus's 8 P-cores."""
    pmu = linux_hybrid_pmu(sys_root + "/devices")
    seen = {}
    for cpu in sorted((Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cap = cpu / "cpu_capacity"
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        if not cap.exists() and pmu is None:
            return None
        key = (pkg.read_text(encoding="utf-8").strip(), core.read_text(encoding="utf-8").strip()) \
            if pkg.exists() and core.exists() else cpu.name
        if pmu is not None:
            seen.setdefault(key, 0 if int(cpu.name[3:]) in pmu[1] else 1)
        else:
            seen.setdefault(key, int(cap.read_text(encoding="utf-8").strip()))
    classes = list(seen.values())
    if pmu is None:
        top = max(classes, default=0)
        classes = [1 if c * 10 >= top * 9 else 0 for c in classes]
    return classes


def cpulist(text: str) -> set:
    """A kernel cpulist ("0-7,16") as a set of CPU numbers."""
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def linux_hybrid_pmu(root: str = "/sys/devices"):
    """#798: (P-core CPUs, E-core CPUs) from Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus), or None on a CPU
    that has no such lists."""
    try:
        p = cpulist((Path(root) / "cpu_core" / "cpus").read_text(encoding="utf-8"))
        e = cpulist((Path(root) / "cpu_atom" / "cpus").read_text(encoding="utf-8"))
    except OSError:
        return None
    return (p, e) if p and e else None


def hybrid_pool_workers(cores) -> int | None:
    """#642 (Hardin22's measurement): on a hybrid CPU the expert pool runs best on the P-cores but the host loop's one
    plus HALF of the E-cores - an E-core runs the expert kernels ~2.2x slower and each layer waits for its slowest
    part (i9-14900KF, 8P + 16E: 15 workers decoded 165 / 116 tok/s against 106 / 84 with all 23).  Only on a CPU with
    more E-cores than P-cores: on an i7-13700KF (8P + 8E, docs/AMD_HIP.md's gfx1030 report) all 15 workers decoded
    38-42 tok/s against 36 with 8, so there the engine's own count stays.  None: the engine's own default (one worker
    per physical core but the host's) stays."""
    if not cores:
        return None
    p, e = cores
    if e <= p:
        return None
    return max(1, p - 1 + e // 2)


def recommend_pool_workers(args: list) -> list:
    """`args` with setup's recommended `--pool-workers` for a hybrid CPU, unless they set one already (a calibration's
    measured count, or the user's own).  A recommendation: the config line can be edited or removed."""
    n = hybrid_pool_workers(cpu_cores())
    if n is None or "--pool-workers" in args:
        return args
    p, e = cpu_cores()
    ok(f"hybrid CPU ({p} performance + {e} efficiency cores): {n} CPU expert workers - the performance cores and half "
       "of the efficiency cores (--pool-workers in the config; START-HERE --calibrate measures it on this PC)")
    return [*args, "--pool-workers", str(n)]


def linux_sockets(sys_root: str = "/sys"):
    """(sockets, physical cores per socket) from the kernel's topology files, or None when they do not say."""
    cores = {}
    for cpu in (Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"):
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        try:
            cores.setdefault(pkg.read_text(encoding="utf-8").strip(), set()).add(core.read_text(encoding="utf-8").strip())
        except OSError:
            return None
    return (len(cores), min(len(c) for c in cores.values())) if cores else None


def cpu_sockets():
    """(sockets, physical cores per socket); None when the kernel's topology files do not say."""
    return linux_sockets()


def two_socket_note(sockets) -> list[str]:
    """Bench #674 #707 (2-socket Xeons: 17-18 workers beat 35 by ~20%): a tip to keep the expert pool on one socket's
    cores, one fewer for the host loop.  Nothing is written to the config."""
    if not sockets or sockets[0] < 2 or sockets[1] < 3:
        return []
    n = sockets[1] - 1
    return [f"tip: this PC has {sockets[0]} CPU sockets of {sockets[1]} cores. Expert workers on the other socket have "
            f"been measured slower than fewer, local ones (35 vs 17-18 on 2-socket Xeons): try --pool-workers {n} in "
            "the config's args (START-HERE --calibrate measures it on this PC)"]


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F; read from /proc/cpuinfo."""
    name, avx2, avx512 = "unknown CPU", False, False
    try:
        txt = open("/proc/cpuinfo").read()
        flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
        avx2 = "avx2" in flags
        avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
        m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
        name = m.group(1) if m else name
    except (OSError, AttributeError):
        pass
    return name, avx2, avx512


def cpu_floor(avx2: bool) -> str:
    """The experimental older-CPU build this PC needs (#394 #595 #623): "" with AVX2 (the normal engine), "avx" (Sandy /
    Ivy Bridge, AMD Bulldozer), "none" (SSE4.2 + POPCNT: Nehalem, Westmere), or "unsupported".  STRATA_ISA_FLOOR=avx|
    none asks for that build on any PC (testing it on a newer one); read from /proc/cpuinfo."""
    forced = os.environ.get("STRATA_ISA_FLOOR", "").strip().lower()
    if forced in ("avx", "none"):
        return forced
    if avx2:
        return ""
    try:
        txt = open("/proc/cpuinfo").read()
        flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
    except (OSError, AttributeError):
        flags = set()
    f = "avx" if "avx" in flags else "sse4.2" if {"sse4_2", "popcnt"} <= flags else ""
    return {"avx": "avx", "sse4.2": "none"}.get(f, "unsupported")


def gpus():
    """Every AMD GPU, numbered as the kernel's KFD topology and HIP number them (the order the engine is told to
    use) - setup's list of the cards."""
    return amd_gpus()


SPLIT_MIN_VRAM_GB = 8                                   # a card sharing a model holds the dense weights and its own
                                                        # prompt buffers too (docs/MULTI_GPU.md)
SPLIT_PROMPT_VRAM_GB = 12                               # #448: a split stage lends a prompt chunk's buffers from its
                                                        # own cache; below this one cannot fund a 4096-token chunk
                                                        # (a 10 GB RTX 3080 beside a 32 GB card: 512 tokens, prompts
                                                        # 6.2x slower), while the big card alone could


def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can)."""
    p = amd_problem(g)
    if p:
        return p
    if together and g["vram_gb"] < SPLIT_MIN_VRAM_GB - 0.5:
        return (f"not supported together with other GPUs - {g['vram_gb']:.0f} GB of VRAM (a card sharing the model "
                f"needs {SPLIT_MIN_VRAM_GB} GB or more)")
    return None


def gpu_rank(g):
    """The order cards share a model in: a discrete card before an APU, then the most VRAM (amd_rank)."""
    return amd_rank(g)


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def gpu_table(found) -> None:
    say("  Your AMD GPUs:")
    for g in found:
        p = gpu_problem(g)
        say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + ("can be used" if p is None else p))


def together_ok(found) -> list:
    """The cards that can share one model, in the order they would (empty if fewer than two)."""
    ok_ = sorted([g for g in found if gpu_problem(g, together=True) is None], key=gpu_rank)
    return ok_ if len(ok_) >= 2 else []


def split_short(cards) -> list:
    """#448: the later cards of a split that would cap its prompt chunk below what the first card alone reads (a card
    under SPLIT_PROMPT_VRAM_GB beside one that has it).  Empty: the split is recommended as before."""
    if len(cards) < 2 or cards[0]["vram_gb"] < SPLIT_PROMPT_VRAM_GB - 0.5:
        return []
    return [g for g in cards[1:] if g["vram_gb"] < SPLIT_PROMPT_VRAM_GB - 0.5]


def split_short_note(g) -> str:
    return (f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB) is too small to lend a split its prompt buffers: "
            "it would cap prompt reading at 512-2048-token chunks, several times slower than the first card alone "
            "(#448). It can serve as a helper expert cache instead (docs/SECOND_GPU.md)")


def parse_gpus(text, found) -> list:
    """--gpus / --gpu with several: "0,2" or "all" (every card that can share the model), numbered as setup lists
    the AMD cards."""
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in together_ok(found)]
        if not sel:
            gpu_table(found)
            fail("--gpus all: this PC does not have two GPUs Strata can use together")
        return sel
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        fail(f"--gpus takes GPU numbers as setup lists them, e.g. --gpus 0,2 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 0,2 (one GPU: --gpu 0)")
    return sel


def engine_archs():
    """The AMD architectures the installed HIP engine was compiled for ("gfx1201", ...), or None."""
    try:
        meta = json.loads((ROOT / "engine" / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return [str(x) for x in meta.get("archs", [])] if meta.get("backend") == "hip" else None


def engine_runs_on(g) -> bool:
    ea = engine_archs()
    return ea is None or not ea or g["arch"] in ea


def start_gpus(text):
    """--gpus when starting an installed model: AMD cards as setup lists them ("all": every supported AMD card)."""
    return parse_gpus(text, gpus()) if text else None


def find_tool(name):
    """A tool on PATH, or the one pip installed next to this Python (cmake, ninja)."""
    p = shutil.which(name)
    if p:
        return p
    for d in (Path(sys.executable).parent, Path.home() / ".local" / "bin"):
        c = d / name
        if c.exists():
            return str(c)
    return None


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ downloads
def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    ms = ms_file(url) if model_source() == "modelscope" else None
    if ms is not None:
        if reachable(ms_url(*ms), timeout=30):
            url = ms_url(*ms)
        else:
            warn(f"{what or dst.name}: ModelScope does not answer; downloading it from {hf_endpoint()}")
            ms = None
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404 and hf_unpinned(url) != url:  # #214: the pinned revision is gone from the repository
                warn(f"{what or dst.name}: not at the pinned revision any more; downloading the repository's "
                     "current file")
                url = hf_unpinned(url)
                continue
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + source_hint(url))
            time.sleep(5)
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + source_hint(url))
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        if total and have >= total:                    # stopped after the last byte, before the rename: nothing to
            break                                      # ask for (a range past the end is a 416, retried 30 times)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what or dst.name}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    meta = ms_meta(*ms) if ms is not None else None
    if meta is not None and meta[1]:
        verify_sha256(dst, meta[0], meta[1])           # ModelScope's published hash; kept in the finish mark
    else:
        if ms is not None:
            warn(f"{what or dst.name}: ModelScope publishes no SHA-256 for it: the file is not verified")
        mark(dst)
    ok(f"{what or dst.name} downloaded")


def whole_shard(s: Path) -> bool:
    """A shard as long as its own tensor directory says (check_shards' test, without stopping setup)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(s)
        return s.stat().st_size >= g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors),
                                                     default=0)
    except (OSError, ValueError, struct.error):
        return False


SHARD_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def gguf_dir_shards(folder: Path, fam: dict, model: str) -> list[Path]:
    """--gguf-dir's shards (#305): every -0000i-of-0000N file of the model, N read from the first shard's name as the
    engine and tools/iq_pack.py do.  The published name first; else the one first shard in the folder whose name has
    the size in it (an upload split or named differently: -00001-of-00003, Unsloth's ...-00001-of-00004.gguf).  A
    missing shard is check_shards' error later, as before."""
    first = folder / model_file(fam, model, 1)
    if not first.exists():
        found = sorted(p for p in folder.glob("*-00001-of-*.gguf") if SHARD_NAME.search(p.name))
        mine = [p for p in found if model.lower() in p.name.lower()]
        pick = mine if mine else found
        if len(pick) == 1:
            first = pick[0]
    m = SHARD_NAME.search(first.name)
    total = int(m.group(2)) if m else 1
    if not m or total < 1:
        return [first]
    stem = first.name[:m.start()]
    return [first.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total)) for i in range(1, total + 1)]


# #444: the quantization in a GGUF's name (Unsloth's UD-IQ3_XXS, a K-quant's Q2_K_XL, a GSQ-RCO IQ3_S, ...)
GGUF_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))"
                        r"(?=-\d{5}-of-\d{5}\.gguf$|\.gguf$)", re.I)
SUPPORTED_GGUFS = ("Strata runs ISTA-DASLab's GSQ-RCO files (Qwen3.8-Flash-Next Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Swift "
                   "1.5's; the Coder's IQ1_M) and Unsloth's UD-IQ4_XS only: other GGUFs (Unsloth's "
                   "UD-IQ3_XXS or "
                   "UD-Q2_K_XL, K-quants) cannot be used")


def gguf_unsupported(name: str) -> str | None:
    """#444: the quantization a GGUF's name says, when it is one Strata cannot run (not a setup size); else None."""
    m = GGUF_QUANT.search(name)
    return m.group(1) if m and m.group(1).upper() not in MODELS and not name.lower().startswith("mmproj") else None


def gguf_choice(name: str) -> tuple | None:
    """#444: (--family, --model) whose published first shard this file is, or None (the Coder's IQ1_M is named like
    the original's sizes: the size tells them apart)."""
    for f, d in FAMILIES.items():
        for m in MODELS:
            if f in MODELS[m].get("families", ("qwen", "swift")) and name == model_file(d, m, 1):
                return f, m
    return None


def gguf_dir_problem(folder: Path, first: Path, fam: dict, model: str) -> tuple | None:
    """#444: (message, hint) when --gguf-dir has no file setup can use for this choice: the chosen shard is a GGUF
    Strata cannot run (Unsloth's UD-IQ3_XXS taken for IQ3_XXS by its name), or it is missing and the folder holds
    other GGUFs - then the hint names the --family/--model of the usable ones.  None otherwise (a missing shard in a
    folder without GGUFs stays check_shards' "missing")."""
    bad = gguf_unsupported(first.name) if first.exists() else None
    if bad:
        return f"{first.name} is {bad}, a GGUF Strata cannot run", SUPPORTED_GGUFS
    if first.exists():
        return None
    firsts = sorted(p.name for p in folder.glob("*.gguf")
                    if not p.name.lower().startswith("mmproj") and (not SHARD_NAME.search(p.name)
                                                                     or SHARD_NAME.search(p.name).group(1) == "00001"))
    usable = list(dict.fromkeys(c for c in map(gguf_choice, firsts) if c))
    unusable = [n for n in firsts if gguf_unsupported(n)]
    if not usable and not unusable:
        return None
    hint = SUPPORTED_GGUFS
    if unusable:
        hint += ".\n       Not usable here: " + ", ".join(unusable)
    if usable:
        hint += ".\n       Usable here: " + ", ".join(f"--family {f} --model {m}" for f, m in usable)
    return f"{folder} has no {fam['title']} {model} file", hint


def verify_sha256(s: Path, size: int, sha: str) -> None:
    """A shard's size and SHA-256 against the pinned values (the Unsloth file); the result is kept in its finish mark,
    so the ~5 minutes of hashing 111 GB happen once.  A wrong file is deleted, so the next run downloads it again."""
    m = s.with_name(s.name + ".done")
    if m.exists() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace"):
        return
    have = s.stat().st_size if s.exists() else -1
    if have != size:
        fail(f"{s.name} is {have:,} bytes, not {size:,}", "delete it and run setup again (the download restarts)")
    say(f"  checking {s.name} (SHA-256, {size / 1e9:.1f} GB) ...")
    h = hashlib.sha256()
    with open(s, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != sha:
        s.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
        fail(f"{s.name} has the wrong SHA-256 ({h.hexdigest()}, expected {sha}): deleted",
             "run setup again to download it again")
    mark(s, f"sha256 {sha}")


def resident_budget_gib(model, ram, kv_ram_gb=0.0) -> int:
    """UD-IQ4_XS: the GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM (GiB, ram_gb()) less
    24 for the OS, the engine and the file cache the other experts are read through, less a KV cache streamed to
    RAM; at most all of them, at least 8.  64 GB: 40, the measured setting (docs/UNSLOTH_Q4.md)."""
    gib = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_ram_gb)
    return max(8, min(gib, int(MODELS[model]["arena_gb"] / 1.073741824)))


def budget_choice(model, ram, asked) -> float:
    """S4: the budget model's RAM budget (UD-IQ4_XS): --resident-budget-gib N as given, else the recommendation (resident_budget_gib).
    More than the recommendation is kept, with what it risks (the owner's rule: setup recommends, it never forces)."""
    rec = resident_budget_gib(model, ram)
    if asked is None:
        return rec
    if asked > rec:
        warn(f"a {asked:g} GiB RAM budget is more than setup recommends for this PC ({rec} GiB: the RAM less "
             f"{UNSLOTH_RAM_LEFT_GB} GB for the OS, the engine and the file cache that reads the other experts). Kept "
             "as you chose: the engine clamps it to the RAM it finds free at start (less 4 GB), and the file cache "
             "gets less room - it may be slower, or run the PC out of RAM under load")
    return int(asked) if asked == int(asked) else asked


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")
        if len(g.tensors) == 1 and have - need >= 64:       # (a few bytes may be the last tensor's alignment padding)
            # #657: a shard that holds one tensor (the PLE table) is exactly as long as that tensor; extra bytes are a
            # damaged or wrong file that the engine would refuse at start ("PLE table size mismatch")
            fail(f"{s.name} is longer than its tensor: {have:,} bytes, {need:,} expected ({have - need:,} extra)",
                 f"delete {s.name} and its .done mark and run setup again")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git."""
    llama = ROOT / "third_party" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = ROOT / "third_party" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        # llama.cpp's own web UI (tools/ui) is not used, so it is not unpacked (#206)
        f.extractall(tmp, [m for m in f.namelist() if "/tools/ui/" not in m])
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    # PR #63: a rename can fail with PermissionError while a scanner still holds a file of the fresh unpack;
    # shutil.move falls back to copy-and-delete, and a few retries let the scanner finish.  The target is `llama`
    # itself - moving into its parent would keep the zip's `llama.cpp-<sha>` folder name.
    for attempt in range(5):
        try:
            shutil.move(str(top), str(llama))
            break
        except PermissionError:
            if attempt == 4:
                raise
            shutil.rmtree(llama, ignore_errors=True)   # a partial copy from the failed attempt
            time.sleep(2)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """requirements.txt's requirements, comments and blank lines left out."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements (#214) recorded bare names: those packages are kept as they are (nothing is reinstalled), and the
    pinned dependencies it already has count as installed."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


def model_file(fam: dict, model: str, i: int) -> str:
    """Shard i's file name (the family's pattern; #621: Unsloth's UD-IQ4_XS has three shards)."""
    return fam["file"].format(q=model, i=i)


def model_shards(fam: dict) -> int:
    return fam.get("shards", 2)


def budget_model(cfg: dict) -> str:
    """The Unsloth model a config with a RAM budget runs, from its --native shard's name (UD-IQ4_XS)."""
    a = cfg.get("args", [])
    native = Path(a[a.index("--native") + 1]).name.upper() if "--native" in a and a.index("--native") + 1 < len(a) \
        else ""
    return next((m for m, d in MODELS.items() if d.get("budget") and f"-{m}-" in native), "UD-IQ4_XS")


def unsloth_split_need_gb(model="UD-IQ4_XS") -> float:
    """#498: the RAM a budget model needs on several GPUs, where it has no RAM budget (the engine refuses
    --resident-budget-gib with a layer split): its GGUF files and UNSLOTH_RAM_LEFT_GB more (~118 GB for UD-IQ4_XS).
    The estimate is a worst case, measured conservative on the 111 GB Unsloth file at 165 GiB (2x RTX 3090:
    MemAvailable never under 68 GiB); the 0-free case of #384 was 47 GB with a 70 GB model."""
    return MODELS[model]["download_gb"] + UNSLOTH_RAM_LEFT_GB


REMOTE_EXPERT_OPT = "--remote-expert-opt"
HELPER_CACHE_FLAGS = ("--expert-cache-device1", "--expert-cache-device2", "--expert-cache-device3")


def recommend_remote_expert_opt(cfg: dict, off: bool = False) -> None:
    """0.1.39b (#578): a config on two or more GPUs with a helper cache gets --remote-expert-opt - the helper expert
    caches (--expert-cache-device1..3) then stay complementary to the main GPU's, return their rows already weighted
    and skip the CPU's activation quantization where no expert is left to it (dual RTX 4090: +63% mixed, +132% code
    over the plain helper path).  The engine uses it only with a helper cache; a layer split runs as before.  A
    recommendation: `off` (setup's --no-remote-expert-opt) or "remote_expert_opt": false in the config keeps it out,
    and a single-GPU config is not touched."""
    if not isinstance(cfg.get("gpu"), list) or len(cfg["gpu"]) < 2:
        return
    args = cfg.setdefault("args", [])
    if off or cfg.get("remote_expert_opt") is False:
        if REMOTE_EXPERT_OPT in args:
            args.remove(REMOTE_EXPERT_OPT)
        return
    # #1352: --pipeline-windows (the first card starts the next window) is switched off by --remote-expert-opt (the
    # helper caches), so a config that asks for it keeps its pipeline: one of the two, not both (docs/MULTI_GPU.md)
    pw = args.index("--pipeline-windows") if "--pipeline-windows" in args else -1
    if pw >= 0 and pw + 1 < len(args) and args[pw + 1] != "0":
        if REMOTE_EXPERT_OPT in args:
            warn("--pipeline-windows and --remote-expert-opt are both in this config: the engine turns the pipeline off "
                 "beside the helper caches. Keep one - remove --remote-expert-opt for the pipeline "
                 "(docs/MULTI_GPU.md, #1352)")
        else:
            ok("multi-GPU: --pipeline-windows is set, so --remote-expert-opt is not added (the helper caches turn the "
               "pipeline off; docs/MULTI_GPU.md)")
        return
    # #1447: the flag acts only on the helper caches (--expert-cache-device1..3); without one the engine builds nothing
    # from it, and a plain layer split gained a flag that says "helpers" for no reason. Setup adds it only beside a
    # helper cache; a flag already in the config stays (the user's, or an earlier setup's - harmless).
    if not any(a.split("=", 1)[0] in HELPER_CACHE_FLAGS for a in args):
        return
    if REMOTE_EXPERT_OPT not in args:
        args.append(REMOTE_EXPERT_OPT)
        ok("multi-GPU: --remote-expert-opt (helper expert caches complementary to the main GPU's, #578; "
           "--no-remote-expert-opt leaves it out)")


def split_budget(cfg: dict, yes: bool = False, explicit: bool = False) -> bool:
    """#498: a model with its RAM budget (--resident-budget-gib) started on several GPUs.  The engine refuses the
    budget with a layer split (it exited with code 2), so the split runs without it - all the experts loaded into
    RAM at start - where the RAM holds the GGUFs and 24 GB more; else setup says so and asks (#737: a recommendation,
    not a wall - 128 GB ran it fine): a "no" stops, before the config is saved; `explicit` (--gpus) with --yes goes on.
    True when the config changed."""
    a = cfg.get("args", [])
    if "--resident-budget-gib" not in a:
        return False
    model = budget_model(cfg)
    need, ram = unsloth_split_need_gb(model), ram_gb()
    if ram < need:
        confirm_risk(f"{model} cannot share its RAM budget across GPUs (the engine has no layer split with it), and "
                     f"without the budget it needs ~{need:.0f} GB of RAM (its GGUF files and {UNSLOTH_RAM_LEFT_GB} GB "
                     f"more); this PC has {ram:.0f} GB. The estimate is a worst case: a 128 GB PC ran it (#737), but "
                     "it may page or stall here.", explicit, yes,
                     f"{model} on several GPUs needs ~{need:.0f} GB of RAM; this PC has {ram:.0f} GB",
                     "start it on one GPU: ./setup.sh --gpu N")
    i = a.index("--resident-budget-gib")
    del a[i:i + 2]
    ok(f"{model} on several GPUs: no RAM budget (the engine has none with a layer split) - all its experts are "
       "loaded into RAM from the model files at start, and the files pass through the OS file cache (#498)")
    return True


def offer_together(cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """Starting a model set up for one card on a PC with two or more that can share it: asked once (the answer is
    saved in its config)."""
    if isinstance(cfg.get("gpu"), list) or cfg.get("gpus_asked"):
        return cfg
    found = gpus()
    can = together_ok(found)
    if not can:
        return cfg
    # #498: a model with its RAM budget has no layer split; without it the RAM must hold the GGUFs and 24 GB more
    budget = "--resident-budget-gib" in cfg.get("args", [])
    if budget and ram_gb() < unsloth_split_need_gb(budget_model(cfg)):
        return cfg
    pair = can[:2]
    cfg["gpus_asked"] = True
    # #642: the resident low-RAM variant runs on a layer split, so a config with it is offered on both cards as for
    # any other config
    say()
    say("  This PC has " + " and ".join(gpu_name(g) for g in pair) + ": Strata can share the model across both.")
    say("  Together they hold about twice the model's experts and read prompts about 20% faster (docs/MULTI_GPU.md).")
    if budget:
        say(f"  This model ({budget_model(cfg)}) runs on one GPU with a RAM budget of its experts (recommended: the tested")
        say("  setup). On both it has no budget: all its experts are loaded into RAM at start, which this PC's RAM")
        say("  holds - about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s).")
    short = split_short(pair)             # #448: one card recommended (asked "n" by default)
    for g in short:
        say(f"  {split_short_note(g)}.")
    missing = [g for g in pair if not engine_runs_on(g)]
    if missing:
        say("  The installed engine has no code for " + ", ".join(g["name"] for g in missing) + ": to use them "
            "together, run ./setup.sh --setup --gpus " + ",".join(str(g["index"]) for g in pair))
    elif ask("  Use both from now on? (you can change it later: ./setup.sh --gpu N for one card)",
             ["y", "n"], "n" if short or budget else "y", yes) == "y":
        cfg["gpu"] = [g["index"] for g in pair]
        cfg["layer_split"] = cfg.get("layer_split") or "auto"
        split_budget(cfg)
        recommend_remote_expert_opt(cfg)
        ok("from now on this model runs on " + " + ".join(gpu_name(g) for g in pair))
    else:
        ok("staying on one GPU (./setup.sh --gpus " + ",".join(str(g["index"]) for g in pair) + " switches)")
    write_config(cfg_path, cfg)
    return cfg


# ------------------------------------------------------------------------------------------------ AMD
# The RX 7900 XT / XTX (gfx1100) and the RX 9070 series / Radeon AI PRO R9700 (gfx1201) on Linux, through the HIP
# backend (docs/AMD_HIP.md); the RX 7800 XT / 7700 XT (gfx1101, #254) and the RX 9060 XT (gfx1200, #256) were run by
# their owners; the RX 6800 / 6900 series (gfx1030, #311) and the RX 6700 XT (gfx1031, #524) run but are unvalidated.  There is no ready-made AMD engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo;
# a system ROCm 7 in /opt/rocm is used when it has hipcc and hipBLAS) and the engine is compiled here for the cards.
# Images: the CPU encoder only (docs/AMD_HIP.md).
ROCM_INDEXES = {"gfx1100": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",   # TheRock's wheels per GPU family
                "gfx1101": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1102": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1200": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1201": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1030": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1031": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1151": "https://rocm.nightlies.amd.com/v2/gfx1151/",       # Strix Halo (docs/STRIX_HALO.md)
                "gfx1103": "https://rocm.nightlies.amd.com/v2/gfx110X-all/"}   # Radeon 780M: only with STRATA_EXPERIMENTAL_GFX1103=1
# The TheRock nightly indexes are pruned and move on, and each GPU family's index holds its own range (#1103: gfx103X-all
# starts at 7.13.0a20260422 and has no 7.10; #1267: the 7.10 wheel segfaults on a Strix Halo with kernel 7.2.8), so one
# version for every card cannot work.  The wheel is chosen per index (the family): a preferred version, tried first; when
# the index no longer offers it, the newest one of the same 7.x line, else of the same major, with a warning (rocm_pick).
# STRATA_ROCM_VERSION still forces one exact version (no lookup).
ROCM_VERSION_DEFAULT = "7.10.0a20251120"                 # what Strata's HIP build was tested with (gfx120X, gfx110X)
ROCM_FAMILY_PINS = {"gfx103X-all": "7.13.0a20260515",    # #1103: 7.13.0a20260515 runs; the 7.14 nightlies time out
                    "gfx110X-all": "7.10.0a20251121",    # the experimental gfx1103: this index has no ...20251120
                    "gfx1151": "7.14.0a20260608"}        # #1267: 7.14.0a20260529 to 20260608 run on kernel 7.2.8; the 7.10 wheel segfaults
ROCM_VERSION_OVERRIDE = os.environ.get("STRATA_ROCM_VERSION") or None
ROCM_VERSION = ROCM_VERSION_OVERRIDE or ROCM_VERSION_DEFAULT   # kept for callers that read one version
ROCM_SYSTEM_MIN = (7, 0)       # an older system ROCm is passed over for the wheels (gfx1201 needs ROCm 6.4 or newer)
# STRATA_EXPERIMENTAL_GFX1103=1 (opt-in, unsupported): the Radeon 780M / 760M / 740M iGPU (Ryzen 7040 / 8040, gfx1103) is taken
# as a unified-memory AMD card like Strix Halo, with the portable kernels (no WMMA) (measured on one machine: Ryzen 7 255).  Unset: unchanged.
GFX1103_OPT_IN = os.environ.get("STRATA_EXPERIMENTAL_GFX1103") == "1"
AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030", "gfx1031", "gfx1151") + (("gfx1103",) if GFX1103_OPT_IN else ())
AMD_NAMES = {"gfx1100": "AMD Radeon RX 7900 series (gfx1100)",   # when sysfs has no product name
             "gfx1101": "AMD Radeon RX 7800 XT / 7700 XT (gfx1101)",
             "gfx1102": "AMD Radeon RX 7600 / 7600 XT (gfx1102)",
             "gfx1200": "AMD Radeon RX 9060 series (gfx1200)",
             "gfx1201": "AMD Radeon RX 9070 series / AI PRO R9700 (gfx1201)",
             "gfx1030": "AMD Radeon RX 6800 / 6900 series (gfx1030)",
             "gfx1031": "AMD Radeon RX 6700 XT series (gfx1031)",
             "gfx1151": "AMD Radeon 8060S / 8050S / 8040S (Ryzen AI Max, Strix Halo, gfx1151)",
             # integrated Radeons of other Ryzen families: named so they are not taken for Strix Halo (not supported)
             "gfx1150": "AMD Radeon 890M / 880M (Ryzen AI 300, Strix Point, gfx1150)",
             "gfx1152": "AMD Radeon 860M / 840M (Ryzen AI 300, Krackan Point, gfx1152)",
             "gfx1103": "AMD Radeon 780M / 760M / 740M (Ryzen 7040 / 8040, Phoenix / Hawk Point, gfx1103)"}
AMD_CARDS = ("the RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 9060 XT (gfx1200) and "
             "RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030) and RX 6700 XT "
             "(gfx1031, #524), and the RX 7600 / 7600 XT (gfx1102, one run reported, #942), all unvalidated, and the Ryzen AI Max \"Strix Halo\" APU (Radeon 8060S / 8050S / 8040S, "
             "gfx1151: experimental, docs/STRIX_HALO.md)")


def rocm_index(arch):
    return os.environ.get("STRATA_ROCM_INDEX") or ROCM_INDEXES[arch]


def rocm_family(index):
    """The family name of a TheRock index url (.../v2/gfx103X-all/ -> gfx103X-all)."""
    return index.rstrip("/").rsplit("/", 1)[-1]


def rocm_vkey(version):
    """Sort key of a TheRock version: 7.13.0a20260515 -> (7, 13, 0, 20260515); None when it is not one."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[a-z]+(\d+))?", version)
    return tuple(int(g or 0) for g in m.groups()) if m else None


def rocm_index_versions(html):
    """The versions of the `rocm` package that a TheRock index page (<index>/rocm/) lists, oldest first."""
    return sorted({m for m in re.findall(r"rocm-(\d+\.\d+\.\d+(?:[a-z]+\d+)?)\.(?:tar\.gz|whl)", html or "")
                   if rocm_vkey(m)}, key=rocm_vkey)


def rocm_pick(available, preferred):
    """(version, note) for an index that lists `available`: `preferred` when it is there, else the newest of the same
    7.x line, else the newest of the same major (the note says so); `preferred` unchanged when the list is empty (the
    index could not be read: pip then reports its own error) or nothing of that major exists (the note says so)."""
    if not available or preferred in available:
        return preferred, None
    want = rocm_vkey(preferred)
    for same, what in ((lambda k: k[:2] == want[:2], "line"), (lambda k: k[0] == want[0], "major")):
        hits = [v for v in available if same(rocm_vkey(v))]
        if hits:
            got = max(hits, key=rocm_vkey)
            return got, (f"ROCm {preferred} is not on this index any more: using {got}, the newest of the same {what} "
                         f"(set STRATA_ROCM_VERSION to choose another)")
    return preferred, (f"ROCm {preferred} is not on this index and nothing of its major version is: "
                       f"{available[0]} to {available[-1]} are (set STRATA_ROCM_VERSION to one of them)")


def rocm_wanted(index, listing=None):
    """(ROCm wheel version, warning or None) for a TheRock index (see ROCM_FAMILY_PINS).  `listing` is the index
    page's html (None: fetched from <index>/rocm/; a failed fetch keeps the preferred version)."""
    if ROCM_VERSION_OVERRIDE:
        version, note = ROCM_VERSION_OVERRIDE, None
    else:
        preferred = ROCM_FAMILY_PINS.get(rocm_family(index), ROCM_VERSION_DEFAULT)
        if listing is None:
            try:
                req = urllib.request.Request(index.rstrip("/") + "/rocm/", headers={"User-Agent": "strata-setup"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    listing = r.read().decode("utf-8", "replace")
            except (OSError, ValueError):
                listing = ""
        version, note = rocm_pick(rocm_index_versions(listing), preferred)
    if rocm_family(index) == "gfx1151" and (rocm_vkey(version) or (9, 0))[:2] < (7, 11):
        note = ((note + "; ") if note else "") + (
            "the ROCm 7.10 wheels segfault in the HSA runtime on a Strix Halo with a new kernel (7.2.8, #1267): "
            "the 7.14 line (docs/STRIX_HALO.md) is the one that works")
    return version, note


# Strix Halo (Ryzen AI Max 380 / 385 / 390 / 395 + PRO: Radeon 8040S / 8050S / 8060S, RDNA 3.5, gfx1151) and the other
# integrated Radeons it must never be taken for.  pci.ids lists ONE device id for the whole Strix Halo family
# ("1586  Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics]"; the 8040S shares it: the
# three differ by their CU count, 40 / 32 / 16, not by id).  Aurora's real answers (tools/test_setup_strix_halo.py):
# KFD gfx_target_version 110501, vendor_id 4098, device_id 5510 (0x1586), simd_count 80.
STRIX_HALO_ARCH = "gfx1151"
STRIX_HALO_PCI_IDS = frozenset({0x1586})
AMD_IGPU_PCI = {0x1586: "gfx1151",                                    # Strix Halo
                0x150E: "gfx1150",                                    # Strix Point: Radeon 880M / 890M
                0x1114: "gfx1152", 0x1902: "gfx1152",                 # Krackan Point (Krackan2): Radeon 840M / 860M
                0x15BF: "gfx1103", 0x15C8: "gfx1103", 0x164F: "gfx1103",   # Phoenix: Radeon 740M / 760M / 780M
                0x1900: "gfx1103", 0x1901: "gfx1103"}                 # Hawk Point
STRIX_HALO_BY_SIMDS = {80: "8060S", 64: "8050S", 32: "8040S"}   # KFD simd_count (2 per CU): 40 / 32 / 16 CUs
UMA_OS_LEFT_GB = 6             # the share of the unified memory kept for the OS: what the engine's device_free_bytes() leaves


def gfx_arch_is(arch, name) -> bool:
    """include/strata/kernels/gfx_arch.hpp's gfx_arch_is: `arch` (gcnArchName, "gfx1151" or "gfx1151:sramecc-:xnack-")
    is exactly `name`, never a prefix: gfx1150 and gfx1152 are not gfx1151 (and gfx1151 is not gfx115)."""
    arch = str(arch or "")
    return arch == name or arch.startswith(name + ":")


def is_strix_halo(g) -> bool:
    return isinstance(g, dict) and gfx_arch_is(g.get("arch"), STRIX_HALO_ARCH)


def amd_apply_uma(g: dict, gtt_gb: float = 0.0, ram: float | None = None) -> dict:
    """Strix Halo is an APU with unified memory.  Its "dedicated VRAM" is a BIOS carve-out (512 MB to 96 GB; 2 GB on the
    maintainers' box) and the GPU also reaches shared system memory (Linux: the GTT pool, `mem_info_gtt_total`, half the
    RAM unless the kernel was told otherwise).  The shared part IS the RAM
    the model's experts live in, so it is not extra room beside the RAM.  g gets: uma, dedicated_gb (the carve-out: the
    only part that adds to the RAM, which the low-RAM mode counts) and vram_gb = the memory the GPU can use in all
    (the carve-out plus the shared pool less what the OS keeps), which sizes the context and the parallel slots."""
    if g.get("uma") or not (is_strix_halo(g) or (GFX1103_OPT_IN and gfx_arch_is(g.get("arch"), "gfx1103"))):
        return g
    ram = ram_gb() if ram is None else ram
    carve = max(0.0, float(g.get("vram_gb") or 0.0))
    shared = gtt_gb if gtt_gb > 0 else ram / 2
    shared = max(0.0, min(shared, ram - UMA_OS_LEFT_GB))
    g.update({"uma": True, "dedicated_gb": carve, "shared_gb": shared, "vram_gb": carve + shared})
    return g


def low_ram_vram(g) -> float:
    """The GPU memory the low-RAM mode may count as room beside the RAM: a discrete card's VRAM; an APU's BIOS
    carve-out only (its shared memory is the RAM itself)."""
    return g.get("dedicated_gb", 0.0) if g.get("uma") else g["vram_gb"]


def amd_mem_text(g) -> str:
    """How setup says a card's memory: "24 GB VRAM", or for an APU its unified memory."""
    if g.get("uma"):
        return (f"unified memory: {g['dedicated_gb']:.1f} GB BIOS carve-out + {g['shared_gb']:.0f} GB shared with the "
                f"RAM, {g['vram_gb']:.0f} GB usable by the GPU")
    return f"{g['vram_gb']:.0f} GB VRAM"


STRIX_HALO_FAMILY, STRIX_HALO_MODEL = "unsloth", "UD-IQ4_XS"    # the model docs/STRIX_HALO.md measures
STRIX_HALO_MIN_GB = 80         # UD-IQ4_XS keeps all of its experts in memory from ~80 GB (its menu line says so)


def strix_halo_recommends(gpu, ram) -> bool:
    """Is UD-IQ4_XS the recommended model here: a Strix Halo whose unified memory (the OS's RAM plus the BIOS carve-out)
    holds it - the model docs/STRIX_HALO.md measures.  A recommendation only: the menus still list every size."""
    return is_strix_halo(gpu) and bool(gpu.get("uma")) and ram + gpu.get("dedicated_gb", 0.0) >= STRIX_HALO_MIN_GB


def amd_device_access_problem(dev="/dev", access=os.access, listing=glob.glob) -> str | None:
    """Linux AMD: /dev/kfd and the render nodes must be readable and writable by this user (the render / video groups).
    None when they are, else a sentence for the user.  A missing /dev/kfd is not this problem (no driver / no ROCm)."""
    kfd = os.path.join(dev, "kfd")
    if not os.path.exists(kfd):
        return None
    bad = [kfd] if not access(kfd, os.R_OK | os.W_OK) else []
    bad += [n for n in sorted(listing(os.path.join(dev, "dri", "renderD*"))) if not access(n, os.R_OK | os.W_OK)]
    if not bad:
        return None
    return (f"this user cannot open {', '.join(bad)}: ROCm will find no GPU and the engine will fail with 'no ROCm-capable "
            "device' although nothing else holds the GPU. Add your user to the render and video groups: "
            "sudo usermod -aG render,video $USER, then log out and in again")


def gfx1103_notes(gpu, ram) -> list[str]:
    """What setup tells the owner of a Radeon 780M / 760M / 740M (Phoenix / Hawk Point, gfx1103, unified memory): never
    the Strix Halo text, which is about another chip (gfx1151).  Opt-in only (STRATA_EXPERIMENTAL_GFX1103=1)."""
    notes = [f"  Radeon 780M / 760M class (gfx1103, Ryzen 7040 / 8040): the CPU and the GPU share one memory pool ({ram:.0f} GB "
             f"seen by the OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in "
             "that pool and the expert cache is sized from the memory the OS can give back."]
    notes.append("  The engine is compiled here for gfx1103 (experimental opt-in, STRATA_EXPERIMENTAL_GFX1103=1; "
                 "measured on one machine, not validated on a real card): see docs/AMD_HIP.md. This is not a Strix Halo.")
    if gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     "default is about half of the RAM). The kernel option ttm.pages_limit raises it; setup changes no "
                     "host setting - a bigger model needs the room, a smaller one runs as it is")
    return notes


def igpu_notes(gpu, ram) -> list[str]:
    """The unified-memory notes for exactly this chip: Strix Halo (gfx1151) or the gfx1103 opt-in, nothing else."""
    if is_strix_halo(gpu):
        return strix_halo_notes(gpu, ram)
    if gfx_arch_is(gpu.get("arch"), "gfx1103"):
        return gfx1103_notes(gpu, ram)
    return []


def strix_halo_notes(gpu, ram) -> list[str]:
    """What setup tells a Strix Halo owner (a leading "!" = a warning): the unified memory, the build, the guide, and the
    memory settings that give the GPU room.  Nothing is changed or refused: recommend, never force."""
    notes = [f"  Strix Halo (gfx1151, Ryzen AI Max): the CPU and the GPU share one memory pool ({ram:.0f} GB seen by the "
             f"OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in that pool "
             "and the expert cache is sized from the memory the OS can give back, not from the carve-out."]
    notes.append("  The engine is compiled here for gfx1151 (experimental, measured on one machine): "
                 "see docs/STRIX_HALO.md.")
    if gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     f"default is about half of the RAM). docs/STRIX_HALO.md boots with ttm.pages_limit and "
                     "ttm.page_pool_size to give it most of the RAM; setup changes no host setting - a bigger model "
                     "needs the room, a smaller one runs as it is")
    if gpu.get("dedicated_gb", 0.0) > 16:
        notes.append(f"!the BIOS carve-out is {gpu['dedicated_gb']:.0f} GB: that memory is taken from the OS, while the GPU "
                     "reaches the shared memory anyway. A small carve-out (the BIOS's default or 512 MB - 2 GB) leaves "
                     "the RAM to the model")
    if strix_halo_recommends(gpu, ram):
        notes.append(f"  Recommended model: {STRIX_HALO_MODEL} (the one docs/STRIX_HALO.md measures); the menus list the "
                     "others.")
    return notes


def amd_rank(g):
    """The order setup picks a card in, best first: a discrete card before an APU (its own GDDR is faster than the
    shared LPDDR5X), then the most memory.  A Strix Halo beside a Radeon RX / PRO card is therefore left to --gpu N."""
    return (bool(g.get("uma")), -round(g["vram_gb"]), g["index"])


def amd_pci_devices(sysfs="/sys") -> list[dict]:
    """The AMD (vendor 0x1002) display devices the kernel lists in /sys/class/drm/card*/device, as [{"pci_id",
    "vram_gb", "gtt_gb", "arch"}] with the arch AMD_IGPU_PCI knows ("" when it does not).  The KFD topology answers the
    same without ROCm; this one only names a card the topology does not list (a kernel without /dev/kfd)."""
    found = []
    base = Path(sysfs) / "class/drm"
    if not base.is_dir():
        return found
    for card in sorted(base.glob("card[0-9]*")):
        if not card.name[4:].isdigit():
            continue
        try:
            dev = card / "device"
            if int((dev / "vendor").read_text().strip(), 16) != 0x1002:
                continue
            did = int((dev / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        row = {"pci_id": did, "arch": AMD_IGPU_PCI.get(did, ""), "vram_gb": 0.0, "gtt_gb": 0.0}
        for key, f in (("vram_gb", "mem_info_vram_total"), ("gtt_gb", "mem_info_gtt_total")):
            try:
                row[key] = int((dev / f).read_text()) / 2 ** 30
            except (OSError, ValueError):
                pass
        found.append(row)
    return found


def amd_gpus(sysfs="/sys"):
    """AMD GPUs from the kernel's KFD topology (the amdgpu driver; no ROCm needed), numbered as HIP numbers them:
    the GPU nodes in order, the CPU nodes skipped.  Integrated GPUs are listed too (not supported).
    sysfs: the tree to read (tools/test_setup_amd.py passes a mocked one)."""
    base = Path(sysfs) / "class/kfd/kfd/topology/nodes"
    found = []
    if not base.is_dir():
        return found
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = v.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(sysfs) / f"class/drm/renderD{props.get('drm_render_minor', '')}/device"
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        if name == f"AMD Radeon ({arch})" and arch in AMD_NAMES:
            name = AMD_NAMES[arch]
            if gfx_arch_is(arch, STRIX_HALO_ARCH) and int(props.get("simd_count") or 0) in STRIX_HALO_BY_SIMDS:
                name = (f"AMD Radeon {STRIX_HALO_BY_SIMDS[int(props['simd_count'])]} "
                        "(Ryzen AI Max, Strix Halo, gfx1151)")   # the CU count tells the 8060S / 8050S / 8040S apart
        try:
            gtt = int((dev / "mem_info_gtt_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            gtt = 0.0
        g = {"index": len(found), "name": name, "vram_gb": vram, "arch": arch, "driver": "amdgpu", "vendor": "amd"}
        try:
            g["pci_id"] = int((dev / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            pass
        found.append(amd_apply_uma(g, gtt))            # Strix Halo: unified memory (carve-out + shared)
    return found


def amd_problem(g):
    if g["arch"] not in AMD_ARCHS:
        return (f"not supported - Strata's AMD backend runs on {AMD_CARDS} only, this is {g['arch']}"
                + (" (an integrated Radeon, not Strix Halo)" if g["arch"] in ("gfx1150", "gfx1152", "gfx1103") else ""))
    return None


def amd_parse_gpus(text, amd) -> list:
    """--gpus with AMD cards, numbered as HIP numbers them (setup's list): "1,0", or "all" (every supported card, the
    most VRAM first).  Every chosen card must be one Strata supports (AMD_ARCHS; they may be of different
    architectures: the engine is compiled for each).  Returns the cards, the main one first."""
    usable = [g for g in amd if amd_problem(g) is None]
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in sorted(usable, key=amd_rank)]
    else:
        try:
            sel = [int(x) for x in str(text).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes AMD GPU numbers as setup lists them, e.g. --gpus 1,0 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 1,0 (one GPU: --gpu 1)",
             "this PC has " + (f"{len(usable)} AMD card{'s' if len(usable) != 1 else ''} Strata can use"
                               + (": " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in usable) if usable else "")))
    byid = {g["index"]: g for g in amd}
    for i in sel:
        g = byid.get(i)
        p = "not found on this PC" if g is None else amd_problem(g)
        if p is not None:
            fail(f"AMD GPU {i}{'' if g is None else ' (' + g['name'] + ')'} cannot be used: {p}",
                 ("use these together: --gpus " + ",".join(str(x["index"]) for x in usable)) if len(usable) >= 2 else
                 ("use one card: --gpu " + str(usable[0]["index"])) if usable else f"the AMD backend runs on {AMD_CARDS}")
    return [byid[i] for i in sel]


# ------------------------------------------------------------------------------------------------ AMD (ROCm / HIP)
def rocm_version(root):
    """(major, minor) of a ROCm install, from rocm-core's header; None when it has none."""
    try:
        text = (Path(root) / "include" / "rocm-core" / "rocm_version.h").read_text(encoding="utf-8")
        return tuple(int(re.search(rf"#define\s+ROCM_VERSION_{k}\s+(\d+)", text).group(1)) for k in ("MAJOR", "MINOR"))
    except (OSError, AttributeError, ValueError):
        return None


def rocm_dev_missing(sysroot: Path) -> list:
    """#446: the HIP development files the engine's build needs that a system ROCm lacks (a runtime-only install has
    hipcc and libhipblas but not these, and cmake's enable_language(HIP) then fails on the hip-lang package)."""
    lang = "cmake/hip-lang/hip-lang-config.cmake"     # where CMake's HIP support looks for it
    need = {"lib/" + lang: [sysroot / d / lang for d in ("lib", "lib64", "lib/x86_64-unknown-linux-gnu")],
            "include/hip/hip_runtime.h": [sysroot / "include" / "hip" / "hip_runtime.h"]}
    return [name for name, paths in need.items() if not any(p.is_file() for p in paths)]


def rocm_root(archs):
    """ROCm for compiling and running the HIP engine for `archs` (one arch or a list: the cards of a layer split):
    (root, library folders).  A system ROCm 7 with hipcc, hipBLAS and the HIP development files (#446), else AMD's
    TheRock wheels (rocm_wanted: a version per card family, from the family's index) installed into .venv."""
    archs = [archs] if isinstance(archs, str) else list(archs)
    sysroot = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    if (sysroot / "bin" / "hipcc").exists() and list((sysroot / "lib").glob("libhipblas.so*")):
        ver = rocm_version(sysroot)
        missing = rocm_dev_missing(sysroot)
        if (ver is None or ver >= ROCM_SYSTEM_MIN) and not missing:
            return sysroot, [str(sysroot / "lib")]
        if ver is not None and ver < ROCM_SYSTEM_MIN:
            warn(f"the ROCm in {sysroot} is {ver[0]}.{ver[1]}; Strata needs {ROCM_SYSTEM_MIN[0]}.{ROCM_SYSTEM_MIN[1]} "
                 "or newer: using AMD's wheels in .venv instead")
        else:                                          # #446: a runtime-only ROCm (no -dev packages): cmake would fail
            warn(f"the ROCm in {sysroot} has no HIP development files ({', '.join(missing)}): using AMD's wheels in "
                 ".venv instead (or install them, e.g. AMD's amdrocm-core-dev package for your ROCm and card)")
    indexes = list(dict.fromkeys(rocm_index(a) for a in archs))
    if len(indexes) > 1:                               # TheRock's wheels hold one GPU family's libraries
        fail(f"cards of two GPU families ({', '.join(archs)}) need a system ROCm 7 (in /opt/rocm): AMD's Python "
             "wheels come per family", "install ROCm 7 system-wide, or use cards of one family (--gpu N for one card)")
    index = indexes[0]
    stamp = Path(sys.prefix) / ".strata-rocm.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    wheel = None
    if have.get("index") == index and have.get("version") and not ROCM_VERSION_OVERRIDE \
            and (Path(sys.executable).parent / "rocm-sdk").exists() and have["version"] == ROCM_FAMILY_PINS.get(rocm_family(index), ROCM_VERSION_DEFAULT):
        wheel = have["version"]                         # the family's pin is installed: no lookup
    if wheel is None:
        wheel, note = rocm_wanted(index)
        if note:
            warn(note)
    if have.get("version") != wheel or have.get("index") != index:
        say(f"  Installing ROCm {wheel} for AMD GPUs into .venv (AMD's TheRock wheels, ~10 GB, no sudo) ...")
        pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--index-url", index]
        if have.get("version") == wheel:               # the same version for another GPU family: its own libraries
            run(pip + ["--force-reinstall", "--no-deps", f"rocm=={wheel}"])
        run(pip + [f"rocm[libraries,devel]=={wheel}"])
        stamp.write_text(json.dumps({"version": wheel, "index": index}))
    sdk = Path(sys.executable).parent / "rocm-sdk"
    root = Path(out([str(sdk), "path", "--root"]).strip())
    if not (root / "llvm" / "bin" / "clang++").exists():
        fail(f"ROCm was installed but its compiler is missing ({root})",
             f"remove {stamp} and run this again; or install ROCm 7 system-wide")
    # the card family's libraries only (gfx120X-all -> _rocm_sdk_libraries_gfx120X_all): another family's
    # libhipblaslt.so first on the path would have no kernels for this card
    family = "_rocm_sdk_libraries_" + index.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    dirs = [str(root / "lib")]
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        libs = sorted(sp.glob("_rocm_sdk_libraries_*"))
        libs = [d for d in libs if d.name.lower() == family.lower()] or libs
        dirs += [str(d / "lib") for d in libs if (d / "lib").is_dir()]
    ok(f"ROCm: {root}")
    return root, list(dict.fromkeys(dirs))


def hipblaslt_version(lib_dirs):
    """The installed hipBLASLt's version as the engine reads it (hipblasLtGetVersion: 1.4.1 -> 100401), from the
    header of the ROCm whose libraries the engine loads; None when not found."""
    for d in lib_dirs:
        try:
            text = (Path(d).parent / "include" / "hipblaslt" / "hipblaslt-version.h").read_text(encoding="utf-8")
            v = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", text).group(1))
                 for k in ("MAJOR", "MINOR", "PATCH")]
        except (OSError, AttributeError, ValueError):
            continue
        return v[0] * 100000 + v[1] * 100 + v[2]
    return None


def hipblaslt_table(arch, lib_dirs, ver=None):
    """tools/hip/<arch>-hipblaslt-<version>.txt for this card AND the installed hipBLASLt, else None: its solution
    ids are valid only for that pair (the engine refuses any other table and uses plain hipBLAS).  ver: the
    hipBLASLt version read from the installed headers."""
    ver = ver or hipblaslt_version(lib_dirs)
    table = ROOT / "tools" / "hip" / f"{arch}-hipblaslt-{ver}.txt"
    if ver is not None and table.exists():
        head = table.read_text(encoding="utf-8").split("\n", 2)[:2]
        if f"STRATA_HIPBLASLT_TUNING_V1 {arch} {ver}" in (h.strip() for h in head):
            ok(f"hipBLASLt tuning table: {table.name} (faster prompts)")
            return table
    have = sorted(p.name for p in (ROOT / "tools" / "hip").glob(f"{arch}-hipblaslt-*.txt"))
    warn(f"no hipBLASLt tuning table for {arch} with hipBLASLt {ver or '(version unknown)'}"
       + (f" (have: {', '.join(have)})" if have else "")
       + ": the prompt's dense matrix products use plain hipBLAS (tools/hip/tune_hipblaslt makes a table: "
         "docs/AMD_HIP.md, Tuning table)")
    return None


def build_engine_hip(gpu, llama, vision="none") -> Path:
    """Compile the HIP engine for this AMD GPU into engine/ (again only when its source changed: a `git pull`).
    gpu["archs"]: every architecture it needs code for (the cards of a layer split), else gpu["arch"].  vision "cpu"
    (#304): the image encoder too, for the CPU (there is no HIP encoder build yet)."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    archs = sorted(set(gpu.get("archs") or [gpu["arch"]]))
    has_archs = set(archs) <= set(meta.get("archs", []))
    floor = cpu_floor(cpu_info()[1])                     # "" on an AVX2 CPU: the normal engine
    engine_ok = meta.get("backend") == "hip" and (eng / EXE).exists() and meta.get("src") == src and has_archs and \
        (meta.get("isa_floor") or "") == floor
    vision_ok = vision == "none" or ((eng / VEXE).exists() and meta.get("vision_src") == vsrc)
    if engine_ok and vision_ok:
        ok("engine already built for this PC")
        return eng
    if engine_ok:
        return build_vision_cpu(eng, stamp, meta, llama, vsrc)
    if not (shutil.which("c++") or shutil.which("g++")) or not shutil.which("git"):
        fail("a C++ compiler and git are needed to compile the AMD engine",
             "Ubuntu/Debian: sudo apt install build-essential git   Fedora: sudo dnf install gcc-c++ git")
    root, dirs = rocm_root(archs)
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                       "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    say("  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
        if meta.get("backend") == "hip" and (eng / EXE).exists() and has_archs
        else f"  Compiling the Strata engine for your AMD GPU{'s' if len(archs) > 1 else ''} ({', '.join(archs)}; "
             "10-20 minutes, once) ...")
    cmake_build(ROOT, ROOT / "build-hip", "strata",
                ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                 "-DSTRATA_PREFILL_MMQ=ON", "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(archs),
                 f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                 "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                 f"-DSTRATA_GGML_DIR={llama}", *isa_floor_defs(floor, ROOT / "build-hip", meta)])
    shutil.copy2(ROOT / "build-hip" / EXE, eng / EXE)
    meta = {"source": "local-hip", "backend": "hip", "version": source_version(), "archs": archs, "vision": "none",
            "lib_dirs": dirs, "src": src, **({"isa_floor": floor} if floor else {})}
    if vision != "none":
        return build_vision_cpu(eng, stamp, meta, llama, vsrc)
    stamp.write_text(json.dumps(meta, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


def hip_vision(asked) -> str:
    """The image encoder with the AMD backend (--vision): the CPU one when asked for (#304); a HIP (GPU) encoder build
    is a later step, so `yes`/`gpu` leave images off, as before, and say how to get them."""
    if asked in ("yes", "gpu"):
        warn("the AMD backend has no GPU image encoder yet: images off (--vision cpu reads them on the CPU)")
    return "cpu" if asked == "cpu" else "none"


def build_vision_cpu(eng: Path, stamp: Path, meta: dict, llama, vsrc) -> Path:
    """#304: the CPU image encoder beside the HIP engine (tools/vision, built without the GPU encoder), recorded in its
    BUILD.json."""
    if not ((eng / VEXE).exists() and meta.get("vision_src") == vsrc):
        say("  Compiling the image encoder (for the CPU) ...")
        cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision", "strata-vision",
                    [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_PORTABLE=OFF"])
        shutil.copy2(ROOT / "build-vision" / "bin" / VEXE, eng / VEXE)
    stamp.write_text(json.dumps({**meta, "vision": "cpu", "vision_src": vsrc}, indent=1))
    ok(f"engine: {eng / EXE}, image encoder (CPU): {eng / VEXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the engine
def cmake_build(src, bdir, target, defs):
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv python -m pip install cmake ninja")
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *defs]
    build = [cmake, "--build", str(bdir), "--target", target, "-j", str(max(2, (os.cpu_count() or 4) // 2))]
    # A failed build is tried once more: the compiler now and then fails to parse a file it just wrote, and
    # the same command then gets past it (issue #45); a second attempt only compiles what is still missing.
    run(conf)
    if run(build, check=False).returncode != 0:
        say("  (the build stopped - trying it once more)")
        run(build)


ENGINE_SOURCES = ("CMakeLists.txt", "cmake", "src", "include", "third_party/ggml")
VISION_SOURCES = ("tools/vision",)


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def isa_floor_defs(floor: str, bdir: Path, meta: dict) -> list:
    """The experimental older-CPU build's CMake definition (STRATA_ISA_FLOOR, CMakeLists.txt), none for the normal
    build.  A build folder configured for another floor is configured afresh: ggml's CPU options are cached there."""
    if (meta.get("isa_floor") or "") != floor and (bdir / "CMakeCache.txt").exists():
        (bdir / "CMakeCache.txt").unlink()
    return [f"-DSTRATA_ISA_FLOOR={floor}"] if floor else []


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Strata folder (`Strata-data`), not inside it: updating Strata by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Strata folders this user ran, is kept in a small
# per-user file, so every Strata folder on the PC finds the same files.
DATA_ITEMS = ("models", "packs", "mtp")


LOW_RAM_HEADROOM_GB = 10   # RAM beside the experts: the OS, the engine's other buffers, the server
RESIDENT_ENGINE = (0, 1, 30)   # the first engine with --resident-experts (the low-RAM mode's resident variant)


def low_ram_needed(model, ram) -> bool:
    """The model's experts do not fit this PC's RAM with room left for the rest: they are then mapped from the pack's
    experts.bin instead of copied into RAM (the low-RAM mode)."""
    return ram < MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB


def low_ram_gpu_gb(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how many GB of the model's experts the GPU's cache holds: its VRAM minus ~5 GB for the dense weights,
    buffers and a 32K context's KV cache, minus the KV cache of a longer context (in VRAM in the low-RAM mode: its RAM
    has no room for KV streaming)."""
    kv_tok = 13 * KV_CELL_BYTES.get(kv, 1056)           # bytes per context token: 12 QSA layers + the draft layer
    longer = max(0, ctx - 32768) * kv_tok / 1e9
    return max(0.0, min(MODELS[model]["arena_gb"], vram_gb - 5 - longer))


def low_ram_gpu_share(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how much of the model's experts the GPU holds."""
    return low_ram_gpu_gb(model, vram_gb, ctx, kv) / MODELS[model]["arena_gb"]


def low_ram_resident(model, ram, vram_gb, ctx=32768, kv="int8") -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM with the usual room beside them, so they are
    copied into RAM once (the resident variant, `--resident-experts`) instead of being read through the OS file cache
    (plain `--mmap-experts`, which a PC this short of RAM keeps re-reading from the SSD)."""
    rest = MODELS[model]["arena_gb"] - low_ram_gpu_gb(model, vram_gb, ctx, kv)
    return ram >= rest + LOW_RAM_HEADROOM_GB


def low_ram_fits(model, ram, vram_gb) -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM left beside the rest (as file cache)."""
    arena = MODELS[model]["arena_gb"]
    return ram - 6 + max(0.0, vram_gb - 5) >= arena


def low_ram_wanted(model, ram, choice="auto") -> bool:
    """Does setup put this model in the low-RAM mode on this PC: its experts do not fit the RAM with the usual room
    beside them (`low_ram_needed`), or the user asked for it.  Unsloth's 4-bit file is not the low-RAM mode at all:
    it always reads part of its experts from the files, through its RAM budget.  Step 5 applies exactly this answer,
    and the launcher's preset diff asks it too - so a preset that says "auto" is compared with what auto decides,
    not with the word."""
    if MODELS[model].get("budget"):
        return False
    return choice in ("on", "resident", "mmap") or (choice == "auto" and low_ram_needed(model, ram))


def kv_streaming_ram_gb(ctx, kv) -> float:
    """The RAM a streamed KV cache takes: ~13.7 KB per context token with 8-bit KV (1.7 GB at 128K), 10.6 KB with
    K8V4, 7.5 KB with 4-bit - 12 QSA layers + the draft layer."""
    return ctx * (13 * KV_CELL_BYTES.get(kv, 1056)) / 1e9


def kv_streaming_wanted(model, ctx, kv, ram, choice="auto") -> bool:
    """Will setup stream this model's KV cache on this PC: from 64K up, when the RAM holds the cache beside the
    model's experts (+1 GB); `--kv-streaming on|off` overrides the RAM test (the owner's rule).  Step 7 writes
    `--kv-resident` on exactly this answer, and the launcher's preset diff asks it too: a config's `--kv-resident`
    and a preset's "auto" are the same setting when this says yes."""
    if choice == "off":
        return False
    return ctx >= 65536 and (ram >= MODELS[model]["ram_gb"] + kv_streaming_ram_gb(ctx, kv) + 1 or choice == "on")


def unsloth_together(a, model, ram, gpu, chosen) -> bool:
    """#498: the budget model (UD-IQ4_XS) with several GPUs chosen.  Its RAM budget (--resident-budget-gib) has no
    layer split, so a split runs without it: all its experts loaded into RAM from the GGUFs at start, as with the
    2-3-bit models - only where the RAM holds the GGUF files and 24 GB more (unsloth_split_need_gb; the estimate was
    measured conservative on the 111 GB Unsloth file: 165 GiB, 2x RTX 3090, 31 -> 64-78 tok/s).  An explicit --gpus is
    honoured there; otherwise asked, one GPU by default (--yes: one GPU, as before); an explicit
    --resident-budget-gib keeps one GPU.  True: all of them."""
    names = " + ".join(gpu_name(g) for g in chosen)
    need = unsloth_split_need_gb(model)
    if ram < need:
        note = (f"{model} on several GPUs has no RAM budget and needs ~{need:.0f} GB of RAM (its GGUF files and "
                f"{UNSLOTH_RAM_LEFT_GB} GB more), this PC has {ram:.0f}; the estimate is a worst case (#737: a 128 GB PC "
                "ran it)")
        if a.gpus and a.resident_budget_gib is None:    # #737: asked for by name: a question, not a refusal
            confirm_risk(note + ", but it may page or stall here", True, a.yes,
                         f"{model} on several GPUs needs ~{need:.0f} GB of RAM; this PC has {ram:.0f} GB",
                         f"start it on one GPU: --gpus {gpu['index']}", "  Use them together anyway?")
            ok(f"{model} on {names}, as you chose (--gpus), with {ram:.0f} GB of RAM")
            return True
        warn(f"{note} - using {gpu_name(gpu)} only (--gpus " + ",".join(str(g["index"]) for g in chosen) +
             " uses them together anyway)")
        return False
    if a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib has no layer split: {model} runs on one GPU with it - using {gpu_name(gpu)} only "
             f"(leave the budget out to use {names} together)")
        return False
    note = (f"no RAM budget - all of its experts (~{MODELS[model]['arena_gb']:.0f} GB) are loaded into RAM from the "
            f"model files at start, and the files pass through the OS file cache (needs ~{need:.0f} GB of RAM, this "
            f"PC has {ram:.0f})")
    if a.gpus:
        ok(f"{model} on {names}, as you chose (--gpus): {note}")
        return True
    if not a.yes:
        say()
        say(f"  {model} can run on one GPU with a RAM budget of its experts, or on {names} together without one:")
        say(f"  1) {gpu_name(gpu)} only: the most-used experts kept in RAM, the rest read from the SSD   (recommended: "
            "the tested setup)")
        say(f"  2) {names} together: {note};")
        say("     about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s)")
        if ask(f"{model}: which GPUs?", ["1", "2"], "1", a.yes) == "2":
            ok(f"{model} on {names}: {note}")
            return True
    warn(f"{model} runs on one GPU: using {gpu_name(gpu)} only (--gpus " + ",".join(str(g["index"]) for g in chosen) +
         f" shares it across {names} without the RAM budget: this PC's RAM holds it)")
    return False


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """The owner's rule: setup recommends, it never forces.  A choice setup expects to fail or run badly is said
    plainly (msg), then asked (default n; `default` keeps an older question's own default), or with --yes taken as
    consent when it was asked for explicitly (a flag such as --model or --gpus): --yes alone keeps the stop
    (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def confirm_paging(model, ram, choice, yes, explicit_model=False):
    """The model's experts do not fit this PC's RAM and the low-RAM mode is off.  #125: a warning and a question, not
    a stop - the user may accept paging.  Asked "no" by default, so an unattended --yes install stops here, unless
    the low-RAM mode was turned off explicitly (--low-ram off, #250) or the size was (--model): that is the choice
    already made."""
    need_gb, arena = MODELS[model]["ram_gb"], MODELS[model]["arena_gb"]
    off = choice == "off"
    confirm_risk(f"{model} needs about {need_gb} GB of RAM and this PC has {ram:.0f} GB: its experts alone are "
                 f"{arena:.0f} GB and must stay in RAM, so the system will page part of them from disk. Expect it "
                 "to be much slower, and it may not start at all.\n       A smaller size (Q2_0 or IQ2_XS) fits; more "
                 "RAM fixes it.", off or explicit_model, yes,
                 f"{model} needs about {need_gb} GB of RAM; this PC has {ram:.0f} GB",
                 "choose Q2_0 or IQ2_XS, or add RAM" + ("" if off else f"; or --model {model} --yes (or --low-ram off "
                                                                        "--yes) to install it anyway"),
                 "  Install it anyway?", "y" if off else "n")
    warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose" + (" (--low-ram off)" if off else
                                                                          " (--model)" if explicit_model else ""))


def ctx_ram_need(model, ctx, low_ram=False):
    """#406: the RAM (GB) setup estimates for a long context with IQ3_XXS / IQ3_S: their experts + the context's
    8-bit KV cache + 24 GB of room for everything else (the 0.1.29 arithmetic, counted).  None where the context does
    not count against RAM by this rule: the other sizes, and the low-RAM mode (its KV cache stays in VRAM)."""
    if model not in ("IQ3_XXS", "IQ3_S") or low_ram:
        return None
    return MODELS[model]["arena_gb"] + ctx * 13 * 1056 / 1e9 + 24


def ram_ctx(model, ram, low_ram=False) -> int:
    """#406: the longest context the RAM rule recommends: 128K, or longer where the estimate fits this PC's RAM.  It
    is part of the recommended default (the smaller of it and the GPU's rule); a longer choice is kept, with a note.
    The 200K menu step is never the recommendation (#608)."""
    # 204800 (#608) is a menu step between 128K and 256K, never the recommendation: where 256K does not fit, 128K stays
    return max(c for c in CONTEXTS if c <= 131072 or (c != 204800 and (ctx_ram_need(model, c, low_ram) or 0) <= ram))


def settings_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def other_installs(settings: dict) -> list:
    """Strata folders besides this one that may hold model files: the ones this user ran before, and Strata* folders
    next to this one (a zip unpacked again lands in e.g. `Strata-main (1)\\Strata-main`)."""
    cands = [Path(p) for p in settings.get("installs", [])]
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith("strata"):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith("strata")]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                if v == o or v.startswith(o + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        write_config(cfg_file, new_cfg)


def data_folder(requested: str | None) -> tuple:
    """(the data folder, folders on other drives that still hold model files).  Moves the model files of this folder
    and of earlier Strata folders on the same drive into the data folder, and points their configs there."""
    settings = load_settings()
    dest = Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "Strata-data"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    elsewhere = []
    # #198: the data folder remembered before (a --data-dir to a new place) is a source too, and so is a Strata-data
    # folder nested in any of them (an install that kept its models one level down)
    sources = [ROOT, *other_installs(settings)]
    if settings.get("data_dir") and Path(settings["data_dir"]) != dest:
        sources.append(Path(settings["data_dir"]))
    sources += [f / "Strata-data" for f in list(sources) if (f / "Strata-data") != dest]
    seen = set()
    for folder in sources:
        key = os.path.normcase(str(folder))
        if key in seen:
            continue
        seen.add(key)
        if folder == dest or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in model_config_files(folder):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Strata finds them there)")
    installs = [str(ROOT)] + [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir()]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def write_config(path: Path, cfg: dict):
    """A run config, written whole or not at all (#459): to a temporary file first, then moved over the old one, so
    a setup stopped half-way (a closed window, a full disk) never leaves an empty strata-*.json behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the user's - a
# "sampling" or "mcp_servers" block, "allowed_hosts", "cors_origins", "open_browser" - and is kept when setup runs again
SETUP_KEYS = frozenset({"exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port", "backend", "env",
                        "gpu", "gpus_asked", "layer_split", "host", "api_key", "draft_vocab", "vision"})
SETUP_ENV = frozenset({"STRATA_HIPBLASLT_TUNING", "STRATA_RESIDENT_PIN", "STRATA_NO_ARENA_THP"})   # the "env" entries setup writes
SETUP_VISION = frozenset({"exe", "mmproj", "model", "gpu", "max_tokens", "threads"})


def carry_over(old: dict, cfg: dict) -> list[str]:
    """#629: setup run again for an installed model keeps what the user added to its run config: every key setup does
    not write (`SETUP_KEYS`), the "env" entries setup does not write, and in "vision" the keys setup does not write
    plus an mmproj of their own (a Q8_0 one, #625) that still exists.  `cfg` (the new config) is updated in place;
    the names of what was kept are returned.  Engine options added by hand to "args" are not merged (setup chooses
    those): `args_dropped` names them."""
    kept = []
    for k, v in old.items():
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    env = {k: v for k, v in (old.get("env") or {}).items() if k not in SETUP_ENV and k not in (cfg.get("env") or {})} \
        if isinstance(old.get("env"), dict) else {}
    if env:
        cfg["env"] = {**(cfg.get("env") or {}), **env}
        kept += [f"env {k}" for k in env]
    ov, nv = old.get("vision"), cfg.get("vision")
    if isinstance(ov, dict) and isinstance(nv, dict):
        for k, v in ov.items():
            if k not in SETUP_VISION and k not in nv:
                nv[k] = v
                kept.append(f"vision {k}")
        mm = ov.get("mmproj")                          # a file of the user's own: not the one setup downloads
        if isinstance(mm, str) and Path(mm).name != Path(str(nv.get("mmproj"))).name and Path(mm).is_file():
            nv["mmproj"] = mm
            kept.append("vision mmproj")
    kept += carry_profile_args(old, cfg)
    return kept


def flag_value(args: list, flag: str):
    """The value after `flag` in an argument list, or None."""
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return str(args[i + 1])
    return None


def carry_profile_args(old: dict, cfg: dict) -> list[str]:
    """#775: a learned expert profile the user wired in by hand (--expert-profile <their file>, --expert-profile-save
    <file>) survives a setup run: setup writes the shipped profile, which a measured one of their own beats (2.5-3.1x at
    256K on a 16 GB card in the report).  Their --expert-profile replaces the shipped one only when it is another file
    that still exists; --expert-profile-save is kept when the new config has none.  The names kept are returned."""
    oargs, nargs = old.get("args"), cfg.get("args")
    if not isinstance(oargs, list) or not isinstance(nargs, list):
        return []
    kept = []
    mine, shipped = flag_value(oargs, "--expert-profile"), flag_value(nargs, "--expert-profile")
    if mine and shipped and Path(mine).name != Path(shipped).name and Path(mine).is_file():
        nargs[nargs.index("--expert-profile") + 1] = mine
        kept.append("args --expert-profile")
    save = flag_value(oargs, "--expert-profile-save")
    if save and flag_value(nargs, "--expert-profile-save") is None:
        nargs += ["--expert-profile-save", save]
        kept.append("args --expert-profile-save")
    return kept


def args_dropped(old: dict, cfg: dict) -> list[str]:
    """#629: the engine options of the earlier run config that the new one has no more (by flag name): options added
    by hand, which a setup run does not carry over - the start of the line that names them."""
    def flags(c):
        a = c.get("args") if isinstance(c.get("args"), list) else []
        return [str(x) for x in a if str(x).startswith("--")]
    new = set(flags(cfg))
    return list(dict.fromkeys(f for f in flags(old) if f not in new))


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None) -> None:
    """#629: setup's run config, written over an earlier one for the same model without losing what the user added
    to it: the keys setup does not write are carried over (carry_over), and the earlier file is kept as
    strata-<model>.json.bak when it changes.  `source`: an earlier install's config to carry the keys over from when
    this folder has none yet (a copy set up like the last one).  A line says what was kept, one what was not."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg) if old is not None else []
    bak = None
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            shutil.copyfile(cfg_path, bak)
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
            bak = None
    write_config(cfg_path, cfg)
    if kept:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))
    if bak is not None:
        dropped = args_dropped(old, cfg) if old is not None else []
        say(f"  the earlier run config is kept as {bak.name}" + (
            f"; engine options it had that this one has not (setup chooses those): {' '.join(dropped)}"
            if dropped else ""))


def readable_config(path: Path) -> bool:
    """#459: a config that parses as a JSON object; any other gets a one-line warning naming it."""
    text = None
    try:
        text = path.read_text(encoding="utf-8-sig")
        if isinstance(json.loads(text), dict):
            return True
        why = "not a JSON object"
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:                                 # JSONDecodeError, or bytes that are not UTF-8
        why = "the file is empty" if text is not None and not text.strip() else "not valid JSON"
    warn(f"skipped the earlier config {path} ({why}): setting this copy up without it")
    return False


def model_config_files(folder: Path) -> list:
    """#346: the strata-*.json files of a folder that can be model configs. The server keeps a config's Chat settings
    next to it as strata-<model>.shared-settings.json: it matches the pattern but is no config (no exe, no args)."""
    return [p for p in folder.glob("strata-*.json") if not p.name.endswith(".shared-settings.json")]


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Strata folder on this PC, for a folder that has none yet.  One
    that does not parse (an empty or cut-off file, #459) is skipped with a warning: the newest readable one is used,
    and with none this copy is set up as a fresh install."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += model_config_files(folder)
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return next((c for c in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True) if readable_config(c)), None)


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, KV, images, projection, network)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("strata-"):]
    family = next((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])), "qwen")
    model = (tag[len(FAMILIES[family]["tag"]):] if tag.startswith(FAMILIES[family]["tag"]) else tag).upper()
    if model not in MODELS:                            # (sizes have no dash except UD-IQ4_XS: the old rule)
        model = tag.split("-")[-1].upper()
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    esp = val("--control-vector-scaled")
    esp_path = esp.rsplit(":", 1)[0] if esp else None
    return {"family": family, "model": model if model in MODELS else None,
            "context": int(val("--max-context")) if val("--max-context") else None,
            "kv": val("--kv") if val("--kv") in ("int8", "q4_0") else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "esp": ("on" if Path(esp_path).name == ESP_VECTOR.name else esp_path) if esp_path else "off",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split"),
            # #493: --vram-reserve-mib given at setup (images write the default 700 themselves)
            "vram_reserve_mib": int(val("--vram-reserve-mib")) if (val("--vram-reserve-mib") or "").isdigit() and (
                vis is None or int(val("--vram-reserve-mib")) != VISION["cpu"]["reserve_mib"]) else None}


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def model_config(path: Path) -> bool:
    """#549: a model's run config (a JSON object with "exe" and "args"). Any other strata-*.json in the folder (a
    file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not a Strata model config")
    return False


def installed_configs():
    return [p for p in sorted(model_config_files(ROOT), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version)."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


def source_ver() -> tuple:
    """The same version as a tuple: the engine this source tree builds is always this version (a failed build keeps
    the previous engine, issue #49)."""
    return tuple(int(x) for x in source_version().split(".")[:4] if x.isdigit())


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            m = re.search(rb"engine=(\d+\.\d+\.\d+(?:\.\d+)?)\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:4] if x.isdigit())


def rotational_disk(path) -> str | None:
    """#605: the disk's name when `path` is on a rotational disk (sysfs queue/rotational), else None."""
    try:
        st = os.stat(path)
        p = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        for q in (p, p.parent):                        # a partition has its disk's queue
            f = q / "queue" / "rotational"
            if f.exists():
                return q.name if f.read_text().strip() == "1" else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


def hip_config_cards(sel) -> list[dict]:
    """#566: the AMD cards a HIP config runs on - its "gpu" (one index or a list) in HIP's numbering, as amd_gpus
    lists them; no "gpu": the supported card with the most VRAM, as setup picks it.  Each card's name carries its
    architecture (an RX 7900 XTX and an RX 7900 XT are both gfx1100; a card without a product name in sysfs is known
    by its arch only).  A card that is not found is {} (the key then says "?")."""
    amd = amd_gpus()
    if sel is None:
        usable = [g for g in amd if amd_problem(g) is None]
        sel = min(usable, key=amd_rank)["index"] if usable else None
    byid = {g["index"]: g for g in amd}
    cards = []
    for i in (sel if isinstance(sel, list) else [sel]):
        g = byid.get(i)
        if g is None:
            cards.append({})
            continue
        arch = g.get("arch") or ""
        name = g.get("name") or "?"
        cards.append({**g, "name": name if not arch or arch in name else f"{name} ({arch})"})
    return cards


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache).  The cards are AMD's
    (hip_config_cards)."""
    gl = hip_config_cards(cfg.get("gpu"))
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning Strata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  draft depth, the CPU threads, the expert cache). It takes roughly 15-30 minutes (3 rounds of each value); the PC is busy meanwhile.")
    try:
        since = os.path.getsize(cfg["log"]) if cfg.get("log") and os.path.isfile(cfg["log"]) else 0
    except OSError:
        since = 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        why = CAL.engine_error(cfg.get("log"), since)  # #447: the engine's own reason, not only "see the log"
        if why:
            say(f"       the engine said: {why}")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    write_config(cfg_path, cfg)
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" for k, v in res["settings"].items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return (load_settings().get("calibration") or {}).get(hardware_key(cfg))


def setup_calibration(cfg: dict) -> dict | None:
    """The calibration a (re-)install applies to its new config: the one saved for this PC and model (#566:
    hardware_key names the AMD cards, so a `./setup.sh --calibrate` run is matched to its card and model).
    Setup does not offer the tuning itself on AMD: each control is verified on HIP first."""
    return saved_calibration(cfg)


def upgrade_config(cfg_path: Path, cfg: dict) -> dict:
    """Configs written before v0.1.13 read prompts in fixed 2048-token chunks; the engine now picks the chunk
    itself (`--prefill auto`: up to 8192, as the free VRAM allows - about 2x faster on long prompts)."""
    a = cfg.get("args", [])
    changed = False
    ver = engine_version(cfg["exe"]) if "--prefill" in a else (0, 0, 0)
    if "--prefill" in a and a[a.index("--prefill") + 1] == "2048" and ver >= (0, 1, 13):
        a[a.index("--prefill") + 1] = "auto"
        changed = True
        ok("prompt reading: the engine now picks its chunk size (--prefill auto)")
    elif "--prefill" in a and a[a.index("--prefill") + 1] == "auto" and (0, 0, 0) < ver < (0, 1, 13):
        a[a.index("--prefill") + 1] = "2048"           # an older engine kept after a failed update (issue #49)
        changed = True
        warn(f"the installed engine is {'.'.join(map(str, ver))}: prompts are read in 2048-token chunks until it is updated")
    if changed:
        write_config(cfg_path, cfg)
    return cfg


def update_install(have: list, a) -> int:
    """#475: `setup.py --update` (update.sh, after its git pull): what a plain start does to an install before it
    starts the model, without starting it - the Python packages, the engine when its source changed (compiled again
    for this PC), each installed model's config upgrades and its draft subset.  No question is asked and the model
    files are not touched; a model still running keeps its engine (close it and run this again)."""
    have = [p for p in have if model_config(p)]        # #549: a strata-*.json that is no model config is skipped
    if not have:
        say("  No model is installed in this Strata folder yet: run ./setup.sh to set it up -")
        say("  it finds an earlier install's model files next to it and reuses them.")
        return 0
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")
    for cfg_path in have:
        cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
        cards = hip_config_cards(cfg.get("gpu"))
        if cards:
            build_engine_hip(cards[0], get_llama_cpp(), cfg.get("vision") or "none")
        if "--mtp" in cfg["args"][:-1]:
            refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
        ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
    ver = engine_version(Path(json.loads(have[0].read_text(encoding="utf-8-sig"))["exe"]))
    say()
    ok("Strata is updated" + (f" (engine {'.'.join(map(str, ver))})" if any(ver) else "") +
       ". Start the model with ./setup.sh when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564: the settings a start uses, in one line: the config's engine options (the model's file paths left out)
    and the server's own fields, so a change made by hand to strata-<model>.json can be checked without the log."""
    a, out, i = [str(x) for x in cfg.get("args") or []], [], 0
    while i < len(a):
        flag = a[i]
        val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
        i += 1 if val is None else 2
        if not flag.startswith("--"):
            continue                                   # a positional: the model file
        if val is not None and ("/" in val or "\\" in val or val.lower().endswith((".gguf", ".bin"))):
            continue                                   # a path: --native, --mtp, --profile ...
        out.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', 8080)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "layer_split", "draft_vocab", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            srv.append(f"{k} {','.join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v}")
    return " ".join(out) + ("; " if out else "") + "server " + ", ".join(srv)


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None, keep=None) -> int:
    """keep: settings given on this start that the model keeps from now on (--host, --api-key, --draft-vocab,
    --vram-reserve-mib)."""
    cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    reserve = keep.pop("vram_reserve_mib", None)       # #493: an engine argument, kept in the config's args
    if reserve is not None:
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        write_config(cfg_path, cfg)
        ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    if keep and any(cfg.get(k) != v for k, v in keep.items()):   # #179: a --host/--api-key on a start was ignored
        cfg.update(keep)
        write_config(cfg_path, cfg)
        ok("saved for this model: " + ", ".join(
            "api key" if k == "api_key" else ("the browser opens" if v else "no browser") if k == "open_browser"
            else f"{k.replace('_', ' ')} {v}" for k, v in keep.items()))
    cfg_path.touch()                                     # the most recently used model
    if "--mtp" in cfg["args"][:-1]:
        refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    # AMD, numbered as HIP numbers them (setup's KFD order)
    amd = amd_gpus()
    if isinstance(gpu, list):                          # --gpus: saved, this model runs on these cards from now on
        cards = amd_parse_gpus(",".join(str(i) for i in gpu), amd)
        built = (engine_archs() or [])
        miss = [x for x in cards if built and x["arch"] not in built]
        if miss:
            fail("the installed engine has no code for " + ", ".join(f"{x['name']} ({x['arch']})" for x in miss),
                 "set it up for these cards: ./setup.sh --setup --gpus " + ",".join(map(str, gpu)))
        cfg["gpu"], cfg["gpus_asked"] = gpu, True
        cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
        split_budget(cfg, yes, True)                   # #498: before it is saved (asks when the RAM is short, #737)
        write_config(cfg_path, cfg)
        gpu = None
    elif gpu is not None:                              # --gpu N: this start only, on that card
        cmd += ["--gpu", str(gpu)]
    else:
        cfg = offer_together(cfg_path, cfg, yes)
    use = gpu if gpu is not None else cfg.get("gpu")
    # #364 #384: a resident low-RAM config on several GPUs; #498: a budget model with its RAM budget (by hand)
    if isinstance(use, list) and split_budget(cfg):
        write_config(cfg_path, cfg)
    if isinstance(use, list):
        byid = {x["index"]: x for x in amd}
        ok("GPUs: " + " + ".join(gpu_name(byid[i]) if i in byid else f"GPU {i} (not found)" for i in use)
           + f" together, AMD (layers split {cfg.get('layer_split') or 'auto'})")
    else:
        g = next((x for x in amd if x["index"] == (use if use is not None else x["index"])
                  and amd_problem(x) is None), None)
        if g is not None:
            ok(f"GPU: {g['name']} ({g['vram_gb']:.0f} GB, AMD)")
    browser = open_browser and cfg.get("open_browser") is not False   # #609: "open_browser": false, --no-browser
    if browser:
        cmd.append("--open")
    gb = 0.0
    if "--native" in cfg["args"]:
        try:
            gb = Path(cfg["args"][cfg["args"].index("--native") + 1]).stat().st_size / 1e9
        except (OSError, IndexError):
            pass
    say()
    say("  " + "-" * 100)
    size = f'about {gb:.0f} GB' if gb >= 1 else '34-55 GB'
    a_ = cfg["args"]
    if "--mmap-experts" in a_ and "--resident-budget-gib" not in a_ and "--resident-experts" not in a_:
        # #505: the mapped low-RAM mode loads nothing into RAM up front (the server's narrator says the same)
        say(f"  Starting {cfg.get('model_name', 'the model')}: it maps {size} of experts from the model files (the OS "
            "file cache reads them).")
    else:
        say(f"  Starting {cfg.get('model_name', 'the model')}: it loads {size} into RAM and locks part of it for the "
            "GPU.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - " + (
        "the browser opens when it is ready." if browser else "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    if os.environ.get("STRATA_EXECV"):
        # Replace this process instead of spawning a child. The Docker image sets STRATA_EXECV=1,
        # so there the server is PID 1 and docker stop's SIGTERM reaches the process that can
        # answer the engine with QUIT. Normal starts keep spawning the server as a child.
        os.execv(cmd[0], cmd)
    return subprocess.call(cmd)


# the draft subsets setup copied before (sha256): replaced by the current one, a subset made by hand is kept
OLD_DRAFT_VOCABS = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}   # to 0.1.26: 27 Han tokens


DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin",
                "fr": "draft_vocab_fr.bin"}


def saved_draft_vocab(cfg_path: Path) -> str | None:
    """The draft subset a model's config chose earlier (--draft-vocab), or None: a setup run again without the flag
    rewrites the config, and would otherwise put the default subset back."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in DRAFT_VOCABS else None


def vision_tokens(asked: int | None, earlier: Path | None) -> int:
    """#625: the most image tokens a picture becomes (the config's vision.max_tokens): --vision-tokens N, else what an
    earlier config of this model chose for the encoder (a setup run again keeps it), else the default (VISION).
    More is allowed with a note on the time it takes: setup recommends, it does not cap."""
    default = VISION["cpu"]["max_tokens"]
    if asked is None and earlier is not None:
        try:
            v = json.loads(earlier.read_text(encoding="utf-8-sig")).get("vision")
        except (OSError, ValueError, AttributeError):
            v = None
        mt = v.get("max_tokens") if isinstance(v, dict) else None
        if isinstance(mt, int) and mt > 0 and mt != default:
            asked = mt
    if asked is None:
        return default
    note = ""
    if asked > default:
        note = (" - on the CPU a picture takes longer to encode the more tokens it gets (several seconds more at "
                "1,024 than at 300)")
    ok(f"images: up to {asked} image tokens per picture (--vision-tokens; default {default}){note}")
    return asked


DRAFT_VOCAB_MIB = {"cjk": 348, "cyrillic": 193, "fr": 151, "en": 133}   # the draft head's VRAM per subset (IQ3_S: the largest)
SMALL_DRAFT_VRAM_GB = 14   # #474: below this the default subset's head can be what does not fit


def draft_vocab_note(vram_gb: float, chosen: str | None) -> list[str]:
    """#474: on a card under 14 GB, the default draft subset (cjk, ~348 MiB of VRAM) can be what does not fit at the
    start ("the draft head does not fit"), and the engine's expert cache gets what a smaller one leaves.  Setup
    RECOMMENDS a smaller one here and changes nothing (the owner's rule, #403 #406): a subset chosen with
    --draft-vocab, or kept from an earlier install, gets no note.  [] for every other case."""
    if chosen or not 0 < vram_gb < SMALL_DRAFT_VRAM_GB:
        return []
    return [f"Tip for a {vram_gb:.0f} GB card: the draft layer's default token subset (with Chinese, Japanese and "
            f"Korean) needs up to ~{DRAFT_VOCAB_MIB['cjk']} MiB of VRAM.",
            f"  For English and code answers, ./setup.sh --draft-vocab en needs up to ~{DRAFT_VOCAB_MIB['en']} MiB "
            f"(cyrillic: ~{DRAFT_VOCAB_MIB['cyrillic']}) and leaves the rest to the expert cache - and it is the",
            "  fix when the start stops with \"the draft head does not fit\". The model keeps the choice."]


SMALL_CARD_GB = 7.5            # #496: a card under 8 GB gets a tip (an 8 GB card lists 7.99)


def small_card_note(ctx: int, draft_vocab: str | None) -> list[str]:
    """#496: what frees VRAM on a card under 8 GB when the start stops with "no VRAM is left for the expert cache"
    (the engine already lowers its own reserve on such a card) - a recommendation, setup changes none of it.  (The
    draft layer stays: the server needs it.)"""
    start = "./setup.sh"
    tips = []
    if ctx > 8192:
        tips.append("an 8K context (a smaller KV cache)")
    if draft_vocab != "en":
        tips.append(f"--draft-vocab en (a draft head of ~{DRAFT_VOCAB_MIB['en']} MiB instead of "
                    f"~{DRAFT_VOCAB_MIB[draft_vocab or 'cjk']})")
    lines = ["If the start stops with \"no VRAM is left for the expert cache\" (the engine's log says how much is "
             "short):"]
    if tips:
        lines.append(f"  run {start} again with " + " and ".join(tips) + ", or close other programs that use the GPU.")
    else:
        lines.append("  close other programs that use the GPU.")
    return lines


PARALLEL_MAX = 8               # #465: the engine's batch window holds at most 8 requests
PARALLEL_SHARE = 0.2           # #465: the slots' sessions may take this share of the VRAM the expert cache would hold
PARALLEL_HELD = 0.5            # #465: ... and only where the cache still holds this share of the experts beside them
PARALLEL_COST_NOTE = ("parallel N reduces waiting for several users but costs about 10-25% speed per request on this "
                      "card")


def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    """#465: the VRAM one batch slot's session takes: its KV cache (12 QSA layers; with KV streaming only the 32K
    positions the attention reads stay in VRAM) and the DeltaNet state (~0.17 GB).  Measured: 0.56 GiB at 32K int8."""
    kv_tok = 12 * (576 if kv == "q4_0" else 1056)
    return (min(ctx, 32768) if streaming else ctx) * kv_tok / 1e9 + 0.17


def parallel_recommend(vram_gbs, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """#465: how many requests at once ("parallel") to recommend: 0 = none (one at a time).  Only where the experts
    mostly fit in VRAM - the expert cache (each card's VRAM less ~5 GB, every card of a layer split) still holds
    PARALLEL_HELD of the model's experts beside the slots' sessions, which take at most PARALLEL_SHARE of it, up to 4.
    Where the experts mostly run on the CPU a batch reads about as many experts as the requests one by one and every
    slot's VRAM is expert cache lost: measured on a 12 GB RTX 5070 (Q2_0, 32K), a request alone 11-24% slower with 2-4
    slots, 4 requests together 63 tok/s against 71 one after the other (docs/BATCHING.md)."""
    if isinstance(vram_gbs, (int, float)):
        vram_gbs = [vram_gbs]
    cache_gb = sum(max(0.0, v - 5) for v in vram_gbs)
    slot = parallel_slot_gb(ctx, kv, streaming)
    best = 0
    for n in (2, 3, 4):
        if n * slot <= PARALLEL_SHARE * cache_gb and (cache_gb - n * slot) >= PARALLEL_HELD * arena_gb:
            best = n
    return best


def parallel_note(asked: int | None, vram_gbs, arena_gb: float, ctx: int, kv: str, streaming: bool) -> list[str]:
    """#465: what setup says about "parallel": the recommendation (or, where it would cost speed, why it is left at
    one), or how the asked count compares with it (kept as asked: recommend, never force)."""
    rec = parallel_recommend(vram_gbs, arena_gb, ctx, kv, streaming)
    slot = parallel_slot_gb(ctx, kv, streaming)
    if asked is None or asked <= 1:
        if not rec:
            return [f"Several requests at once: left at one at a time - {PARALLEL_COST_NOTE} (docs/BATCHING.md)."]
        return [f"Several requests at once (opt-in): --parallel {rec} decodes up to {rec} together instead of one "
                f"after the other (each takes ~{slot:.1f} GB of VRAM from the expert cache; docs/BATCHING.md)."]
    lines = [f"parallel requests: {asked} at once (each takes ~{slot:.1f} GB of VRAM from the expert cache, "
             f"{asked * slot:.1f} GB in all)"]
    if asked > PARALLEL_MAX:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not rec:
        lines.append(f"recommended for this card: one at a time - {PARALLEL_COST_NOTE}; kept as you chose")
    elif asked > rec:
        lines.append(f"recommended for this card: {rec} - more slots leave fewer experts in VRAM, which can make every "
                     "request slower; kept as you chose")
    return lines


PREFILL_BIG_RAM_GB = 96        # bench #433 #440 #834 #669: --prefill auto:32768 +21-35% at 96 GB, ~3x slower at 32 GB
PREFILL_RISK_RAM_GB = 64       # below this an explicit auto:32768 is warned about
HEADROOM_RAM_GB = 48           # bench #834: STRATA_RESIDENT_HEADROOM_GIB=6 on a PC with this much RAM or less
AGENT_CACHE_FREE_GB = 24       # bench #882 #440: RAM left beside the model for the conversation cache
AGENT_CACHE_MIB = 8192
SMALL_VISION_VRAM_GB = 12.5    # bench #469: the GPU image encoder shrinks the prompt chunks on cards up to 12 GB


def arg_after(args, flag):
    """The value after `flag` in an argument list, or None."""
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def bench_tips(args, env, ram: float, model_ram_gb: float) -> list[str]:
    """Recommendations from the community bench data (plan 0.1.40 item 14).  Text only: nothing here changes a
    default, the config or the engine's arguments (recommend, never force).  `ram` is this PC's RAM, `model_ram_gb`
    what the chosen model keeps in it."""
    tips = []
    prefill = arg_after(args, "--prefill")
    if prefill == "auto:32768" and ram < PREFILL_RISK_RAM_GB:
        tips.append(f"warning: --prefill auto:32768 on {ram:.0f} GB of RAM: it ran ~3x slower than --prefill auto with "
                    "32 GB (#834 #669); it paid off (+21-35%) with 96 GB")
    elif prefill == "auto" and ram >= PREFILL_BIG_RAM_GB:
        tips.append(f"tip: with {ram:.0f} GB of RAM, --prefill auto:32768 in the config's args read prompts 21-35% "
                    "faster in community benchmarks (#433 #440 #834); not set, nothing changes")
    resident = any(a in args for a in ("--resident-experts", "--resident-budget-gib"))
    if resident and ram <= HEADROOM_RAM_GB and "STRATA_RESIDENT_HEADROOM_GIB" not in (env or {}):
        tips.append(f"tip: on a PC with {ram:.0f} GB of RAM, \"env\": {{\"STRATA_RESIDENT_HEADROOM_GIB\": \"6\"}} in the "
                    "config kept decode speed in a community benchmark and left the system 2 GiB more (#834); the "
                    "default is 4")
    if "--conversation-cache-mib" not in args and ram - model_ram_gb >= AGENT_CACHE_FREE_GB:
        tips.append(f"tip: for several agents or clients at once, --conversation-cache-mib {AGENT_CACHE_MIB} in the "
                    "config's args keeps each one's conversation; without it they were measured re-reading ~90% of "
                    "their prompts (#882 #440; docs/DETAILS.md, Multiple conversations)")
    return tips


DESKTOP_RESERVE_MIB = 3072     # #560 #516: what kept a KDE/Wayland desktop alive beside a full expert cache


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X)."""
    env = os.environ if env is None else env
    return sys.platform.startswith("linux") and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def desktop_reserve_note() -> list[str]:
    """#560 #516: an AMD card that also drives a Linux desktop - with the default 700 MiB reserve the expert cache
    fills it, and when the desktop needs more VRAM amdgpu moves the cache to system RAM, where the OOM killer then ends
    the compositor.  A recommendation, setup changes nothing."""
    return [f"If this AMD card also drives your desktop and the desktop or apps crash once the model is loaded, keep "
            f"more VRAM free: ./setup.sh --vram-reserve-mib {DESKTOP_RESERVE_MIB}",
            "  (remembered for this model; the expert cache gets ~2.3 GB less, a few % of speed)"]


def mtp_corrupt(mtp: Path, env=None) -> bool:
    """#327: True when the MTP tensors an install fetched are not the pinned checkpoint's (tools/mtp_fetch.py verify,
    which hashes only files that changed since they last checked out).  A mirror that ignored range requests left the
    shards' starts there instead, and the draft layer built from them accepted nothing - with no error anywhere."""
    if not (mtp / "tensors").is_dir():
        return False
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(mtp)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 3


def refresh_draft_vocab(rt: Path, choice: str = "cjk") -> None:
    """The draft layer's token subset in the MTP folder: `cjk` (data/draft_vocab.bin, since 0.1.27, #137), `en`
    (data/draft_vocab_en.bin, the English/code subset before it: ~110 MiB less VRAM, English answers 1-2% faster) or
    `cyrillic` (data/draft_vocab_cyrillic.bin: English/code and the whole Cyrillic script, for Ukrainian, Russian,
    Bulgarian, Serbian... answers) or `fr` (data/draft_vocab_fr.bin: English/code and the tokens of French text, #597).
    Copied when missing or when a shipped subset other than the chosen one is there; a subset made by hand is kept."""
    new, dst = ROOT / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
    if not new.exists() or not rt.is_dir():
        return
    if dst.exists():
        old = hashlib.sha256(dst.read_bytes()).hexdigest()
        shipped = OLD_DRAFT_VOCABS | {hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
                                      for f in DRAFT_VOCABS.values() if (ROOT / "data" / f).exists()}
        if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
            return
        ok("draft layer: the token subset " + {"cjk": "with Chinese, Japanese and Korean",
                                               "cyrillic": "with the Cyrillic script",
                                               "fr": "for French"}.get(choice,
                                                                                          "for English and code (less VRAM)"))
    shutil.copyfile(new, dst)


def ensure_engine_for(cards, cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """The installed engine must have code for every card the model starts on: a card added later (--gpus with a
    different gfx generation, #128) or a new GPU in the PC otherwise stops the start with 'no kernel image'.  Such a
    card gets the engine compiled for all of them, before the start."""
    missing = [g for g in cards if not engine_runs_on(g)]
    if not missing:
        return cfg
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    meta = json.loads(info.read_text(encoding="utf-8"))
    say()
    say("  The installed engine has no code for " + ", ".join(f"{g['name']} ({g['arch']})" for g in missing) +
        ": it is compiled for " + ("these cards" if len(cards) > 1 else "it") + " now.")
    archs = sorted(set(meta.get("archs", [])) | {g["arch"] for g in cards})
    build_engine_hip({**cards[0], "archs": archs}, get_llama_cpp(), cfg.get("vision") or "none")
    dirs = json.loads(info.read_text(encoding="utf-8")).get("lib_dirs") or []
    cfg["lib_dirs"] = dirs + [d for d in cfg.get("lib_dirs") or [] if d not in dirs]
    write_config(cfg_path, cfg)
    return cfg


def write_run_script(model, cfg_path, port, open_browser=True):
    """run-<model>.sh: the server with this config; `open_browser` False (#609: --no-browser) leaves --open out."""
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port)] + (["--open"] if open_browser else [])
    script = ROOT / f"run-{model.lower()}.sh"
    script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                      encoding="utf-8")
    script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ the rope config
def derived_factor(ctx: int, trained: int = 262144) -> float:
    """The automatic extension factor: the FINAL context over the trained one, at least 1.

    Factor 1 removes the automatic expansion - the trained angles stand as they are - but it is not a
    switch for rope as a whole: an explicitly chosen method's settings keep their defined behavior.
    """
    return max(1.0, float(ctx) / float(trained))


def resolve_rope(ctx: int, scaling, scale, trained: int = 262144):
    """The rope config for the context ACTUALLY SERVED: (scaling, scale); scaling None = no scaling flags.

    An explicit --rope-scaling/--rope-scale always wins - a user-supplied factor is kept verbatim even
    when a reduction changed the context.  Past the trained range an omitted method defaults to yarn -
    llama.cpp's extension method: the trained angles survive on the high-frequency pairs and the
    magnitude correction keeps the attention temperature - and an omitted factor is derived from the
    final context (final / trained, at least 1), as is an explicitly chosen method's missing factor
    inside the trained range: factor 1, the trained angles, no expansion.  An explicit none is refused
    past the trained range (the setup will not configure a run it knows is out of spec) rather than
    silently overridden.
    """
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError("--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                             "trained 262144, so there is nothing to scale)")
        if scaling in (None, "none"):
            return None, None          # the stock model, by choice or by default
        return scaling, scale if scale is not None else derived_factor(ctx, trained)
    if scaling == "none":
        raise ValueError(f"a {ctx // 1024}K context is past the model's trained 262144, and --rope-scaling none "
                         "keeps the stock angles there - the model has never seen those positions, so the setup "
                         "refuses the combination instead of quietly overriding it. Pick --rope-scaling yarn or "
                         "linear, or rerun with --context 262144 or lower")
    return scaling or "yarn", scale if scale is not None else derived_factor(ctx, trained)


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), help="qwen = Qwen3.8-Flash-Next, swift = Swift 1.5")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--rope-scaling", choices=["none", "linear", "yarn"],
                    help="the RoPE extension for a context past the model's trained 262144: linear (position "
                         "interpolation) or yarn - llama.cpp's types. Omitted with a scaled context, the setup "
                         "picks yarn; 'none' is refused for such a context")
    ap.add_argument("--rope-scale", type=float,
                    help="the extension factor (default: the final context over the trained 262144, at least 1 - "
                         "1.5 for 384K, 2 for 512K, 1 inside the trained range)")
    ap.add_argument("--kv", choices=["int8", "q4_0", "k8v4"],
                    help="KV cache precision above 8K context: int8 (default), q4_0 (half the memory, a little less "
                         "precise) or k8v4 (hybrid: INT8 K + 4-bit V, 816 B/cell)")
    ap.add_argument("--vision", choices=["yes", "no", "none", "cpu"],
                    help="let the model read images on the CPU (cpu); `yes` asks for the GPU encoder, which this fork "
                         "does not build, so images stay off and setup says so")
    ap.add_argument("--vision-tokens", type=int, metavar="N",
                    help="the most image tokens a picture becomes (default 1024 with the encoder on the GPU, 300 on "
                         "the CPU): more reads small text and charts better, and takes longer to encode; remembered "
                         "for this model")
    ap.add_argument("--experimental-speed-projection", metavar="on|off|GGUF",
                    help="EXPERIMENTAL, off by default: the control vector in data/experimental-speed-projection "
                         "(or another GGUF) as a projection on layers 4-44; see docs/DETAILS.md")
    ap.add_argument("--port", type=int, help="the server's port (default: the one the install was set up with, 8080 for a new one)")
    ap.add_argument("--gpu", help="one GPU, numbered as setup lists the AMD cards (default: asked when several can "
                                  "be used; with --setup it is saved, when starting it is for that start only)")
    ap.add_argument("--gpus", help="several AMD GPUs sharing one model, as setup lists them: \"0,2\", or \"all\" "
                                   "(every card that can); the first is the main one. Saved, also when starting "
                                   "(see docs/MULTI_GPU.md)")
    ap.add_argument("--layer-split", help="with --gpus: where each later GPU's layers start (\"18\", \"16,32\"), one "
                                          "rising number per GPU after the first - not layers per card; default "
                                          "auto, placed from each GPU's free VRAM")
    ap.add_argument("--no-remote-expert-opt", action="store_true",
                    help="with two or more GPUs: leave out --remote-expert-opt, which setup adds there (#578)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other "
                                   "devices on your network (issue #26; set --api-key too)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0); several keys: key1,key2")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--data-dir", help="where the model files go (~70-120 GB): default Strata-data next to this folder, "
                                       "remembered for every Strata folder on this PC")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with every shard: "
                                       "<name>-00001-of-0000N.gguf ... -0000N-of-0000N.gguf)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the installed engine, Python packages and model settings as a start would, without "
                         "starting the model (update.sh runs it after a git pull)")
    ap.add_argument("--source", choices=SOURCES, default=None,
                    help="where the model files come from: auto (default: Hugging Face), huggingface or modelscope "
                         "(mainland China: the same files, checked against ModelScope's published SHA-256; "
                         "STRATA_SOURCE)")
    ap.add_argument("--inspect", nargs="+", metavar=("SOURCE", "VARIANT"),
                    help="what a GGUF is and whether Strata runs it, from its headers only (no download): a file, a "
                         "folder, a URL, ms:owner/repo (ModelScope) or hf:owner/repo, and optionally a variant name")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's settings for this PC (about 15-30 minutes, longer on a slow card), then start the model")
    ap.add_argument("--draft-vocab", choices=list(DRAFT_VOCABS),
                    help="the draft layer's tokens: cjk = with Chinese, Japanese and Korean (default), en = English "
                         "and code only (~110 MiB less VRAM, English answers 1-2%% faster), cyrillic = English, code "
                         "and the Cyrillic script (Ukrainian, Russian... answers decode ~30%% faster), fr = English, "
                         "code and French (French answers: 18%% more drafts accepted)")
    ap.add_argument("--low-ram", choices=["auto", "on", "off", "resident", "mmap"], default="auto",
                    help="read the model's experts from one file in its folder instead of copying them all into RAM "
                         "(for a PC with a big GPU and little RAM); auto: when the experts would not fit the RAM. In "
                         "this mode the experts the GPU does not hold are copied into RAM once when they fit (resident), "
                         "else read through the OS file cache (mmap); resident / mmap force one of the two")
    ap.add_argument("--resident-budget-gib", type=float, metavar="N",
                    help="UD-IQ4_XS: the GiB of its experts kept in RAM (default: the RAM less 24 GB, 40 on 64 GB; "
                         "more is kept as you choose, with a note)")
    ap.add_argument("--vram-reserve-mib", type=int, metavar="N",
                    help="VRAM in MiB the engine leaves free for other programs (a game, another model; the engine's "
                         "default: 700); the expert cache takes that much less")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="up to N requests decode together (batch slots, opt-in; default: one at a time, the others "
                         "wait). Each slot takes VRAM from the expert cache; setup says what it recommends")
    ap.add_argument("--kv-streaming", choices=["auto", "on", "off"], default="auto",
                    help="from a 64K context: keep the KV cache in RAM and only the attention's window in VRAM (more "
                         "experts fit on the GPU); auto: when the RAM has room for it")
    a = ap.parse_args()
    if os.name == "nt":
        fail("Strata-AMD-Linux runs on Linux only", "this fork has no Windows support")
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    if a.inspect:                                      # headers only: nothing is installed
        sys.exit(subprocess.run([sys.executable, str(ROOT / "tools" / "strata_inspect.py"), *a.inspect[:2]]).returncode)
    if a.resident_budget_gib is not None and not a.resident_budget_gib > 0:
        ap.error("--resident-budget-gib takes a number of GiB above 0, e.g. --resident-budget-gib 32")
    if a.vision_tokens is not None and a.vision_tokens < 1:
        ap.error("--vision-tokens takes a number of image tokens, 1 or more, e.g. --vision-tokens 768")
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error("--vram-reserve-mib takes a number of MiB, 0 or more, e.g. --vram-reserve-mib 2048")
    if a.gpu is not None:                             # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as setup lists the AMD cards, e.g. --gpu 1 (or --gpus 0,2), not {a.gpu!r}")
    say("Strata - Qwen3.8-Flash-Next on a normal PC (a GPU + system RAM + CPU)")
    data, elsewhere = data_folder(a.data_dir)          # the model files: in the data folder, found from any copy
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    # ---- 0. already installed: just start it
    have = installed_configs()
    if a.update:                                       # #475: update.sh - never starts the model
        return update_install(have, a)
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    adopted = None                                     # #629: the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of Strata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                adopted = prev
                a.kv = a.kv or ch["kv"]
                a.vision = a.vision or ch["vision"]
                a.experimental_speed_projection = a.experimental_speed_projection or ch["esp"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if a.vram_reserve_mib is None:          # #493: an explicit reserve set up before
                    a.vram_reserve_mib = ch.get("vram_reserve_mib")
                if isinstance(ch.get("gpu"), list):     # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(g) for g in ch["gpu"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = start_gpus(a.gpus) or a.gpu
    port = a.port or 8080                              # a new install's port (issue #32: --port for an existing one)
    if have and a.calibrate and not (a.setup or a.model or a.family or a.check):
        pick_cfg = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
            pick_cfg = have[int(ask("Tune which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        if not calibrate_config(pick_cfg):             # #447: said again where it is not lost above the start
            say()
            warn("this PC is NOT tuned: the tuning failed (the reason is above); the model "
                 + ("keeps" if a.no_start else "starts with") + " the default settings")
        return 0 if a.no_start else start(pick_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})

    # ---- 1. the PC
    step(1, "checking your PC")
    amd = gpus()
    say("  Your AMD GPUs:")
    for g in amd:
        say(f"    GPU {g['index']}: {g['name']}, {amd_mem_text(g)} - " + (amd_problem(g) or "can be used"))
    acc = amd_device_access_problem()                        # a warning only: recommend, never force
    if acc:
        warn(acc)
    if not amd:                        # the KFD topology is empty: name a Strix Halo the kernel sees
        for d in amd_pci_devices():
            if d["pci_id"] in STRIX_HALO_PCI_IDS:
                say(f"    The kernel lists an AMD Strix Halo (PCI 1002:{d['pci_id']:04x}) but /dev/kfd's topology is "
                    "empty: ROCm cannot use it yet (docs/STRIX_HALO.md: the amdgpu driver and /dev/kfd access, "
                    "your user in the render and video groups)")
    usable = [g for g in amd if amd_problem(g) is None]
    if not usable:
        fail("no AMD GPU Strata can use" + ("" if amd else " (the amdgpu driver's KFD topology is empty)"),
             f"Strata-AMD-Linux runs on AMD cards it supports ({AMD_CARDS}); an NVIDIA card is not supported")
    if a.gpus:                                     # a layer split across these cards, the first one the main
        chosen = amd_parse_gpus(a.gpus, amd)
        gpu = chosen[0]
    elif a.gpu is not None:
        gpu = next((g for g in usable if g["index"] == a.gpu), None)
        if gpu is None:
            fail(f"AMD GPU {a.gpu} cannot be used", "use one of: " + ", ".join(f"--gpu {g['index']}" for g in usable))
        chosen = [gpu]
    else:
        gpu = min(usable, key=amd_rank)            # a discrete card before an APU, then the most memory
        chosen = [gpu]
        if len(usable) > 1 and any(g.get("uma") for g in usable) != all(g.get("uma") for g in usable):
            others = [g for g in usable if g is not gpu]
            say(f"  Using GPU {gpu['index']} ({gpu['name']}); setup recommends the card with its own memory over "
                "the APU's shared memory. To use another: "
                + ", ".join(f"--gpu {g['index']} ({g['name']})" for g in others))
    # the engine is compiled for every chosen card's architecture
    gpu = {**gpu, "count": len(amd), "archs": sorted({g["arch"] for g in chosen})}
    chosen = [gpu] + chosen[1:]
    sel = [g["index"] for g in chosen]
    multi = sel if len(sel) > 1 else []
    a.gpu = gpu["index"] if len(amd) > 1 else a.gpu
    if multi:
        ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
    ok(f"GPU: {gpu['name']}, {amd_mem_text(gpu) if gpu.get('uma') else format(gpu['vram_gb'], '.1f') + ' GB VRAM'}, "
       f"{gpu['arch']} (AMD: docs/{'STRIX_HALO' if is_strix_halo(gpu) else 'AMD_HIP'}.md)")
    if gpu.get("uma"):
        for line in igpu_notes(gpu, ram_gb()):
            (warn if line.startswith("!") else say)(line.lstrip("!"))
    if gpu["vram_gb"] < 11:
        warn("less than 12 GB of " + ("GPU memory (this APU's carve-out + shared memory)" if gpu.get("uma") else "VRAM")
             + ": Strata will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    need = min(d["ram_gb"] for d in MODELS.values())
    low_ok = low_ram_fits("IQ1_M", ram, low_ram_vram(gpu)) and a.low_ram != "off"   # the smallest model, mapped
    if ram < need - 4 and not a.check and not low_ok:
        # every model keeps ALL its experts in RAM (23+ GB); VRAM only holds a copy of the most-used ones, so a
        # bigger GPU does not lower this.  The owner's rule: a stop by default, a risk the user can take (--model
        # with --yes, or y)
        confirm_risk(f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB: Strata keeps all of "
                     "the model's experts in RAM (23-50 GB, whatever the GPU), so the OS will page them from disk. "
                     "Expect it to be very slow, and it may not start at all.", bool(a.model), a.yes,
                     f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB",
                     "Strata keeps all of the model's experts in RAM (23-50 GB, whatever the GPU) and the GPU holds a "
                     "copy of the most-used ones: it needs 32 GB of RAM or more (48 GB for the full model); --model "
                     "NAME --yes installs one anyway")
        warn(f"going on with {ram:.0f} GB of RAM, as you chose")
    ram_msg = (f"RAM: {ram:.0f} GB" if ram >= need - 4 else f"RAM: {ram:.0f} GB (less than the {need} GB the smallest model needs)"
               + ("; the GPU's VRAM makes up for it (the low-RAM mode)" if ram < need - 4 and low_ok else ""))
    (ok if ram >= need - 4 or low_ok else warn)(ram_msg)       # #977: a RAM below every model's floor is not [ok]
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    floor = cpu_floor(avx2)
    if floor == "unsupported":
        fail("this CPU has neither AVX2 nor SSE4.2; Strata needs at least SSE4.2 (Intel Nehalem, 2008, or newer)")
    if not avx2:
        # #394 #595 #623: an older CPU gets an engine compiled here, whose CPU experts run on ggml-cpu's kernels for
        # this CPU.  Experimental: measured only on newer CPUs with the older path forced, and by users on a few
        # Xeons.  A warning, not a stop.
        warn(f"this CPU has no AVX2: Strata support for it is EXPERIMENTAL and slow. Setup compiles the engine on this "
             f"PC for {'AVX' if floor == 'avx' else 'SSE4.2'} (STRATA_ISA_FLOOR={floor}; 10-20 minutes, once), and the "
             "CPU's share of the experts runs on ggml-cpu's kernels, a few times slower than on an AVX2 CPU. "
             "See \"Older CPUs\" in docs/INSTALL.md")
    if a.check:
        say()
        any_fits = False
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
            if d.get("budget"):
                verdict = (("EXPERIMENTAL, " if d.get("experimental") else "") +
                           f"fits with {resident_budget_gib(m, ram)} GiB of its experts in RAM, the rest "
                           "read from the SSD" if ram >= d["ram_gb"] else "does not fit")
            elif low_ram_needed(m, ram) and low_ram_fits(m, ram, low_ram_vram(gpu)) and a.low_ram != "off":
                verdict = (f"fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, low_ram_vram(gpu)):.0f}% "
                           "of its experts, " + ("the rest stays in RAM)" if low_ram_resident(m, ram, low_ram_vram(gpu))
                                                 else "the rest is read from the SSD as needed)"))
            any_fits = any_fits or not verdict.startswith("does not fit")
            say(f"  {m:8s} needs ~{d['ram_gb']} GB RAM: {verdict}")
        if not any_fits:
            # #977: every size says "does not fit", so the verdict says so too (and the exit code, for scripts). It only
            # reports: --model NAME --yes still installs one anyway.
            say(f"\nThis PC cannot run Strata yet: no model size fits {ram:.0f} GB of RAM (the smallest needs about "
                f"{need} GB). --model NAME --yes installs one anyway, slowly.")
            return 1
        say("\nThis PC can run Strata. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    fams = list(FAMILIES)
    if a.family:
        family = a.family
    else:
        rec_fam = STRIX_HALO_FAMILY if strix_halo_recommends(gpu, ram) and STRIX_HALO_FAMILY in fams else fams[0]
        for i, f in enumerate(fams, 1):
            d = FAMILIES[f]
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}" + ("   [experimental]" if d.get("experimental") else "")
                + (f"   (recommended for Strix Halo: {STRIX_HALO_MODEL})" if f == rec_fam and rec_fam != fams[0] else ""))
        family = fams[int(ask("Which model?", [str(i) for i in range(1, len(fams) + 1)],
                              str(fams.index(rec_fam) + 1), a.yes)) - 1]
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    say()
    names = [m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))]
    names.sort(key=lambda m: bool(MODELS[m].get("experimental")))   # an experimental size last, never the default
    if a.model and a.model not in names:
        # #444: say which family has that size, and (with --gguf-dir) which files Strata can run at all
        elsewhere_fams = [f for f in FAMILIES if f in MODELS[a.model].get("families", ("qwen", "swift"))]
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names)
             + (f" (or {a.model}: " + ", ".join(f"--family {f} --model {a.model}" for f in elsewhere_fams) + ")"
                if elsewhere_fams else "")
             + (f".\n       {SUPPORTED_GGUFS}" if a.gguf_dir else ""))
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        if d.get("budget"):
            say(f"  {i}) {m} {d['about']}; download {d['download_gb']:.0f} GB, keeps ~"
                f"{resident_budget_gib(m, ram)} GB of its {d['arena_gb']:.0f} GB of experts in RAM{fit}")
            continue
        if low_ram_needed(m, ram) and low_ram_fits(m, ram, low_ram_vram(gpu)) and a.low_ram != "off":
            fit = (f"   <- fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, low_ram_vram(gpu)):.0f}%, "
                   + ("the rest in RAM)" if low_ram_resident(m, ram, low_ram_vram(gpu)) else "the rest from the SSD)"))
        say(f"  {i}) {m:8s} {d['about']}; download {d['download_gb']:.0f} GB, uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = str(names.index("IQ3_XXS") + 1) if ram >= 60 and "IQ3_XXS" in names else "1"
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    budget, split_no_budget = None, False
    if MODELS[model].get("budget"):
        # Unsloth's UD-IQ4_XS: a RAM budget of experts, the rest from the GGUF on the SSD - not the low-RAM mode (no
        # experts.bin: it would be another 77 GB on the disk), and one GPU (the budget mode has no layer split) unless
        # the RAM holds the GGUFs and 24 GB more: then several, without the budget, if asked for (#498)
        if MODELS[model].get("experimental"):
            warn(f"{model} is EXPERIMENTAL (docs/UNSLOTH_Q4.md): most of its experts are read from the SSD while it "
                 "answers, so it is several times slower than the 2-3-bit models; quality checked against llama.cpp")
        if ram < MODELS[model]["ram_gb"]:
            confirm_risk(f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC has {ram:.0f} GB: "
                         f"its RAM budget would be {resident_budget_gib(model, ram)} GiB, so nearly every expert is "
                         "read from the SSD while it answers (very slow), and it may run out of RAM",
                         bool(a.model), a.yes, f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC "
                         f"has {ram:.0f} GB", f"choose one of the 2-3-bit models, or --model {model} --yes to "
                         "install it anyway", "  Install it anyway?")
            warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
        budget = budget_choice(model, ram, a.resident_budget_gib)
        if multi and not unsloth_together(a, model, ram, gpu, chosen):
            multi, sel, chosen = [], [gpu["index"]], [gpu]
        split_no_budget = bool(multi)                         # #498: on several GPUs without the RAM budget
        if not split_no_budget:
            ok(f"RAM budget: {budget:g} GiB of {model}'s experts in RAM, the rest read from the model files on the SSD")
        if a.low_ram not in ("auto", "off"):
            warn(f"--low-ram {a.low_ram} does not apply to {model}: it always reads part of its experts from the files")
    elif a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib is for UD-IQ4_XS: {model} keeps all of its experts in RAM or in "
             "the low-RAM mode")
    low_ram = low_ram_wanted(model, ram, a.low_ram)
    # (the low-RAM mode's variant is decided once the context is known, below; #642: on several GPUs the resident
    # variant runs as on one card)
    if not low_ram and budget is None and ram < MODELS[model]["ram_gb"] - 4:
        confirm_paging(model, ram, a.low_ram, a.yes, bool(a.model))
    ok(f"size: {model}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    small = min(x["vram_gb"] for x in chosen)         # each card keeps its layers' KV of the whole context
    rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072
    if budget is not None:                             # UD-IQ4_XS: every GB of KV is a GB fewer of cached experts
        rec_ctx = 8192 if small < 14 else 32768
    # #406: the RAM rule is part of the recommendation (the smaller of the two), no longer a cap over the user's choice
    rec_ctx = min(rec_ctx, ram_ctx(model, ram, low_ram))
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            need_c = ctx_ram_need(model, c, low_ram)
            note = ("   (recommended for your GPU)" if c == rec_ctx else "") + \
                   ("   (experimental: setup adds rope scaling)" if c > 262144 else "") + \
                   (f"   (needs ~{need_c:.0f} GB RAM, this PC has {ram:.0f}: may run out of memory)"
                    if c > 131072 and need_c is not None and need_c > ram else "")
            say(f"  {i}) {c // 1024}K tokens{note}")
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    # #406 #364: a context past the RAM rule (an explicit --context, a pick in the list, or the earlier install's) is
    # kept, with what it risks.  It used to become 128K: users ran 256K fine where setup's estimate said no.
    need_gb = ctx_ram_need(model, ctx, low_ram)
    if need_gb is not None and ram < need_gb and ctx > 131072:
        warn(f"{ctx // 1024}K with {model} needs ~{need_gb:.0f} GB of RAM by setup's estimate "
             f"({MODELS[model]['arena_gb']:.0f} GB of experts + the context + room for the rest); this PC has "
             f"{ram:.0f}. Kept as you chose: it may be slower or run out of RAM under load. {rec_ctx // 1024}K is the "
             "recommended size.")
    scaling = a.rope_scaling
    if ctx > 262144 and scaling is None and not a.yes:
        # the interactive path: one question, yarn preselected (llama.cpp's extension method, recall-tested
        # here at 512K).  With --yes nothing prints: resolve_rope takes yarn below and the ok() line says so.
        say()
        say(f"  A {ctx // 1024}K context runs the model past its trained 262,144 positions: the rotary angles")
        say("  get rescaled (llama.cpp's RoPE extension). yarn keeps the trained angles on the high-frequency")
        say("  pairs and corrects the magnitudes; linear shrinks every angle. Override any time with")
        say("  --rope-scaling.")
        scaling = ask("RoPE extension method?", ["yarn", "linear"], "yarn", a.yes)
    try:
        scaling, rope_scale = resolve_rope(ctx, scaling, a.rope_scale)
    except ValueError as e:
        fail(str(e))
    if scaling is not None:
        origin = ("final context / trained 262144; override with --rope-scale" if a.rope_scale is None
                  else "as requested")
        ok(f"rope scaling: {scaling}, factor {rope_scale:g} ({origin})")
    ok(f"context: {ctx} tokens")
    # the KV cache (the model's memory of the conversation): 8-bit, or 4-bit after a Hadamard rotation (PR #21)
    kv = "fp16" if ctx <= 8192 else (a.kv or "int8")
    if ctx > 8192 and not a.kv and not a.yes:
        say()
        say("  KV cache precision (the model's memory of the conversation):")
        say("  1) 8-bit   (recommended: what every published number was measured with)")
        say("  2) 4-bit   half the memory (about 4% faster at 128K), but measurably less precise on long")
        say("             documents; long-context lookups (needle tests) still pass")
        kv = ["int8", "q4_0"][int(ask("KV cache?", ["1", "2"], "1", a.yes)) - 1]
    if ctx > 8192:
        ok(f"KV cache: {'8-bit' if kv == 'int8' else '4-bit (Hadamard-rotated)'}")
    vision = hip_vision(a.vision)
    ok("images: " + {"none": "off", "cpu": "on (encoder on the CPU)"}[vision])
    # The low-RAM mode's two variants.  resident: the experts the GPU's cache does not hold (and, as far as RAM allows,
    # the ones the prompt path borrows cache room from) are copied from the pack's experts.bin into RAM once, so
    # nothing is read from the SSD while it answers (engine 0.1.30, --resident-experts; the engine falls back to mmap
    # with a warning when they do not fit the RAM it finds free).  mmap: they are read through the OS file cache.
    # The GPU's share: its VRAM less the dense weights and buffers, this context's KV cache and the image encoder's room.
    resident = False
    if low_ram:
        arena = MODELS[model]["arena_gb"]
        vram = low_ram_vram(gpu) - (VISION["cpu"]["reserve_mib"] / 1024 if vision != "none" else 0)
        share = low_ram_gpu_share(model, vram, ctx, kv)
        rest = arena - low_ram_gpu_gb(model, vram, ctx, kv)
        resident = a.low_ram == "resident" or (a.low_ram != "mmap" and low_ram_resident(model, ram, vram, ctx, kv))
        if multi:      # every chosen card's share (the image encoder on the main one); #364 #384: the mapped variant
            held = min(arena, low_ram_gpu_gb(model, vram, ctx, kv) +
                       sum(low_ram_gpu_gb(model, low_ram_vram(x), ctx, kv) for x in chosen[1:]))
            share, rest = held / arena, arena - held
            # #642: the resident variant on a split, by the same rule as on one card - what no card holds fits the RAM
            resident = a.low_ram == "resident" or (a.low_ram != "mmap" and ram >= rest + LOW_RAM_HEADROOM_GB)
            if resident:
                ok(f"low-RAM mode on {len(chosen)} GPUs: they hold ~{100 * share:.0f}% of {model}'s experts "
                   f"({arena:.0f} GB) and the other ~{rest:.0f} GB stay in RAM ({ram:.0f} GB), read once from a copy "
                   "in the model folder")
            else:
                ok(f"low-RAM mode on {len(chosen)} GPUs: {model}'s experts ({arena:.0f} GB) are read from the model "
                   f"folder through the OS file cache instead of a copy in RAM ({ram:.0f} GB); the GPUs hold "
                   f"~{100 * share:.0f}% of them")
                if share < 0.6:
                    warn("most of the experts are read from the SSD while it answers: expect it to be much slower "
                         "than with enough RAM (a faster SSD and a smaller size help)")
        elif resident:
            ok(f"low-RAM mode: the GPU holds ~{100 * share:.0f}% of {model}'s experts ({arena:.0f} GB) and the other "
               f"~{rest:.0f} GB stay in RAM ({ram:.0f} GB), read once from a copy in the model folder")
        else:
            ok(f"low-RAM mode: {model}'s experts ({arena:.0f} GB) are read from the model folder through the OS file "
               f"cache instead of a copy in RAM ({ram:.0f} GB); the GPU holds ~{100 * share:.0f}% of them")
            if share < 0.6:
                warn("most of the experts are read from the SSD while it answers: expect it to be much slower than "
                     "with enough RAM (a faster SSD and a smaller size help)")
    # EXPERIMENTAL: the experimental-speed-projection control vector (data/experimental-speed-projection), off unless
    # chosen here; with it loaded, the web app and the API switch it off per request
    esp = None
    esp_choice = (a.experimental_speed_projection or "").strip()
    if family in ("qwen", "coder"):                   # the Coder: the same model's residual stream
        if not esp_choice:
            say()
            say("  EXPERIMENTAL - speed projection: a small control vector applied while the model runs (layers 4-44).")
            say("  It changes how the model answers: its package describes it as a refusal-direction projection (the")
            say("  model declines far fewer requests). Off unless you choose it; when on, the web app can switch it off")
            say("  per chat. Details: data/experimental-speed-projection/README.md")
            esp_choice = "on" if ask("Turn on the experimental speed projection?", ["y", "n"], "n", a.yes) == "y" else "off"
        if esp_choice.lower() not in ("off", "no", "n", "0"):
            esp = ESP_VECTOR if esp_choice.lower() in ("on", "yes", "y", "1") else Path(esp_choice).expanduser().resolve()
            if not esp.is_file():
                fail(f"the experimental speed projection's vector is missing: {esp}")
        ok("experimental speed projection: " + ("ON (experimental)" if esp else "off"))
    elif esp_choice.lower() not in ("", "off", "no", "n", "0"):
        warn("the experimental speed projection is made for the original Qwen3.8-Flash-Next, not Swift 1.5: left off"
             if family == "swift" else f"the experimental speed projection is not tested with {model}: left off")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    shards = gguf_dir_shards(models_dir, fam, model) if a.gguf_dir else \
        [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam) + 1)]
    problem = gguf_dir_problem(models_dir, shards[0], fam, model) if a.gguf_dir else None
    if problem:                                        # #444: files Strata cannot run, or another choice's files
        fail(*problem)
    if not a.gguf_dir and not all(sh.exists() and done(sh) for sh in shards):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            cand = [r / "models" / tag / sh.name for sh in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model files found in {models_dir}")
                break
    for s in shards:                                   # #173: a whole file copied in by hand has no finish mark
        if s.exists() and not done(s) and whole_shard(s):
            try:
                mark(s, "whole (checked against its own tensor directory)")
            except OSError as e:                       # a read-only folder (--gguf-dir on a share): the file is still whole
                warn(f"{s.name} is whole but its finish mark cannot be written ({e})")
    have_model = all(s.exists() and (done(s) or a.gguf_dir) for s in shards)
    # #425 (jctaborda): a download that resumes needs room only for what is still missing - the finished shards and
    # the .part files already on the disk count
    on_disk = sum(f.stat().st_size for s in shards for f in (s, s.with_name(s.name + ".part")) if f.is_file()) / 1e9
    to_fetch = 0 if a.gguf_dir or have_model else max(MODELS[model]["download_gb"] - on_disk, 0)
    # count only what step 6 will still write: a pack whose experts.bin is already there (the AVX-512 Q2_0
    # conversion, or the low-RAM mode's copy) and an existing MTP draft layer need no new room
    pack_now = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    pack_bin = (pack_now / "experts.bin").exists() and (pack_now / "index.txt").exists()
    mtp_have = find_in(roots, "mtp/rt/experts.bin") is not None
    q2_avx = model == "Q2_0" and avx512 and family == "qwen"
    need = to_fetch + (2 if mtp_have else 8) + \
        (40 if q2_avx and not pack_bin else 0) + (1 if vision != "none" else 0) + \
        (MODELS[model]["arena_gb"] + 1 if low_ram and not q2_avx and not pack_bin else 0)
    if free_gb(models_dir) < need:
        fail(f"not enough free disk space in {models_dir}: need ~{need:.0f} GB" +
             (f" ({on_disk:.0f} GB of the model is already there)" if on_disk >= 1 and not have_model else ""),
             "use --models-dir on a bigger drive")

    # ---- 3. python packages
    step(3, "Python packages")
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml, mtmd)")
    eng = build_engine_hip(gpu, llama, vision)
    meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    lib_dirs = meta.get("lib_dirs") or []
    engine_ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
    need_engine = MODELS[model].get("engine", UNSLOTH_ENGINE)
    if budget is not None and engine_ver < need_engine:      # checked before the 94 GB download
        fail(f"{model} needs engine {'.'.join(map(str, need_engine))} or newer; this one is {meta.get('version')}",
             "update Strata and run setup again")
    ok(f"engine: {eng / EXE}")

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}")
    if not a.gguf_dir:
        missing = [s.name for s in shards if not (s.exists() and done(s))]
        if missing:                                    # #495: files downloaded by hand go here, or --gguf-dir
            say(f"  The model files go in {models_dir}")
            say(f"  Files you already have: put them here with their original names ({', '.join(missing)}), or use "
                "--gguf-dir <their folder>.")
            if model_source() == "modelscope":
                say(f"  Downloading from ModelScope ({ms_endpoint()}); --source huggingface downloads from Hugging Face")
                warn("ModelScope serves the repositories' current files, not the pinned revisions: each file is checked "
                     "against the SHA-256 ModelScope itself publishes (self-attested); the MTP tensors against the "
                     "pinned checkpoint's own hashes")
            elif hf_endpoint() != HF_DEFAULT:
                say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all its sizes and the Coder: reuse one that is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf") if done(p)]
            if family in ("qwen", "coder") and s.name.endswith("00002-of-00002.gguf") and other and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model) + s.name, s)
    check_shards(shards)
    for s in shards:                                   # the Unsloth files: pinned sizes and SHA-256
        if s.name in fam.get("sha256", {}):
            verify_sha256(s, *fam["sha256"][s.name])
    ok("model files present")
    mmproj = Path(a.models_dir) / fam["mmproj"]
    if not mmproj.exists():
        mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
    if vision != "none":
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        else:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        ok(f"vision encoder: {mmproj}")
        if vision == "cpu" and "BF16" in mmproj.name:       # a tip only (recommend, never force; #625)
            say("       tip: on the CPU a Q8_0 copy of this encoder is a third smaller and about as exact (embedding "
                "cosine 0.999 vs BF16); see 'A Q8_0 encoder' in docs/DETAILS.md")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for Strata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if model == "Q2_0" and avx512 and family == "qwen":
        # the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a one-time ~40 GB conversion
        if not (pack / "index.txt").exists() or not (pack / "experts.bin").exists():   # index.txt is written last
            say("  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...")
            run([sys.executable, str(ROOT / "tools" / "strata_pack.py"), "build", "--gguf", str(shards[0]),
                 "--out", str(pack), "--skip-hash", "--force"], env=env)   # --force: #634 refuses an occupied pack
            run([sys.executable, str(ROOT / "tools" / "pack_index.py"), "--pack", str(pack)], env=env)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                 "--out", str(pack)], env=env)   # writes <pack>/tokenizer/
    elif not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        # every tensor as the GGUF stores it; the experts are read from the GGUF at start (seconds to build)
        # (Unsloth's packs: --compat-bf16 - the quants the engine cannot read become BF16, the form the engine reads)
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             *fam.get("pack_args", [])], env=env)
    if low_ram and not (pack / "experts.bin").exists():
        say(f"  Writing the experts into one file for the low-RAM mode (one time, {MODELS[model]['arena_gb']:.0f} GB) ...")
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             "--experts-bin"], env=env)
    ok(f"model prepared: {pack}")
    mtp = (find_in(roots, "mtp/rt/experts.bin") or data / "mtp/rt/experts.bin").parent.parent
    rt = mtp / "rt"
    corrupt = (rt / "experts.bin").exists() and mtp_corrupt(mtp, env)
    if corrupt:
        warn("some MTP tensors are not the checkpoint's (a download mirror that ignored range requests, #327): "
             "fetching them again and rebuilding the draft layer")
    if corrupt or not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)],
            env={**env, "STRATA_SOURCE": model_source()})
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    # a setup run again without --draft-vocab keeps the subset this model's config chose before (cyrillic, fr, en)
    draft_vocab = a.draft_vocab or saved_draft_vocab(ROOT / f"strata-{tag.lower()}.json")
    refresh_draft_vocab(rt, draft_vocab or "cjk")
    ok(f"MTP draft layer: {rt}")
    for line in draft_vocab_note(gpu.get("vram_gb", 0.0), draft_vocab):   # #474: a recommendation, nothing changes
        say("  " + line)

    # ---- 7. the start script
    step(7, "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
    # (a 4-shard file: the engine finds the PLE table's shard itself from shard 1, the measured setup)
    args = ["--pack", str(pack), "--native", str(shards[0]), *(["--ple-gguf", str(ple)] if len(shards) <= 2 else []),
            "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if scaling is not None:     # the resolved config: explicit flags as given, or the automatic yarn+factor
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if resident and a.low_ram != "resident" and engine_ver < RESIDENT_ENGINE:
        resident = False                               # an engine from before --resident-experts would refuse it
        ok(f"low-RAM mode: engine {meta.get('version')} has no resident variant yet; the experts are read through "
           "the OS file cache (run setup again after the next engine update)")
    if low_ram:   # the experts from the pack's experts.bin: the ones the GPU does not hold copied into RAM, or mapped
        args += ["--resident-experts" if resident else "--mmap-experts"]
    disk = rotational_disk(ple)                        # #605
    if disk:
        tensor = next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
        size = getattr(tensor, "expected_bytes", lambda: None)()
        table_gb = size / 1e9 if size else 28.8
        if ram >= MODELS[model]["ram_gb"] + table_gb + 4:
            args += ["--ple-io", "ram"]
            ok(f"the model is on a rotational disk ({disk}): its {table_gb:.0f} GB n-gram table is kept in RAM "
               "(--ple-io ram) - read from the disk at random, it can stall prompts for minutes (#605)")
        else:
            warn(f"the model is on a rotational disk ({disk}): its n-gram table is read from it at random, which can "
                 f"stall prompts for minutes (#605). An SSD is recommended; with ~{table_gb:.0f} GB more RAM, "
                 "--ple-io ram in the config's args keeps the table in RAM instead")
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 10.6 KB with K8V4, 7.5 KB with 4-bit, so only
    # when it fits.
    kv_ram_gb = kv_streaming_ram_gb(ctx, kv)      # the branches below are kv_streaming_wanted, with its messages
    # --kv-streaming on|off overrides the RAM test (the owner's rule).
    stream_fits = ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1
    if ctx >= 65536 and a.kv_streaming == "off":
        ok("KV streaming off, as you chose (--kv-streaming off): the KV cache stays in VRAM")
    elif ctx >= 65536 and (stream_fits or a.kv_streaming == "on"):
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
        if not stream_fits:
            warn(f"KV streaming needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
                 f"uses; this PC has {ram:.0f}. Kept as you chose (--kv-streaming on): it may page or run out of RAM "
                 "under load")
        if split_no_budget:                                   # #498: no budget to take it out of
            pass
        elif budget is not None and a.resident_budget_gib is None:   # its RAM comes out of the experts' budget
            budget = resident_budget_gib(model, ram, kv_ram_gb)
            ok(f"RAM budget: {budget} GiB (less the KV cache's RAM)")
        elif budget is not None and budget > resident_budget_gib(model, ram, kv_ram_gb):
            warn(f"the KV cache's {kv_ram_gb:.1f} GB of RAM come on top of your {budget:g} GiB RAM budget (setup "
                 f"would take them out of it: {resident_budget_gib(model, ram, kv_ram_gb)} GiB); kept as you chose")
    elif ctx >= 65536:   # #620: say why, so a regenerated config that lost --kv-resident is not a surprise
        ok(f"KV streaming off: it needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
           f"uses, and this PC has {ram:.0f}; the KV cache stays in VRAM (fewer cached experts). --kv-streaming on "
           "turns it on anyway")
    elif a.kv_streaming == "on":
        warn("--kv-streaming on: a context under 64K is not streamed (the attention's window holds all of it): off")
    if budget is not None and not split_no_budget:   # UD-IQ4_XS: the experts read from the GGUF in place, the most-used N
        args += ["--resident-budget-gib", f"{budget:g}"]   # GiB kept in RAM (#498: a layer split has no budget)
    if vision != "none":
        args += ["--vision", "--vram-reserve-mib", str(VISION["cpu"]["reserve_mib"])]
    if a.vram_reserve_mib is not None:                 # #493: VRAM left free for other programs (only when given)
        if "--vram-reserve-mib" in args:
            args[args.index("--vram-reserve-mib") + 1] = str(a.vram_reserve_mib)
        else:
            args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
        ok(f"VRAM kept free for other programs: {a.vram_reserve_mib} MiB (--vram-reserve-mib; the expert cache takes "
           "that much less)")
    if not multi and 0 < gpu.get("vram_gb", 0.0) < SMALL_CARD_GB:
        # #496: on a 6 GB card the expert cache can get no room at all; the engine lowers its own reserve when that
        # is what it takes, and says what is short when even that is not enough.  Setup only says what helps.
        for line in small_card_note(ctx, draft_vocab):   # a recommendation: nothing changes
            say("  " + line)
    elif a.vram_reserve_mib is None and linux_desktop():
        for line in desktop_reserve_note():              # #560 #516: a recommendation: nothing changes
            say("  " + line)
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(ROOT / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port}
    cfg["backend"] = "hip"
    # the dense prompt GEMMs through hipBLASLt with kernels measured on this GPU generation (tools/hip; +40-60%
    # prompt speed on the 7900 XTX): only a table for this card's arch AND the installed hipBLASLt version (the
    # engine refuses any other one and falls back to plain hipBLAS)
    table = hipblaslt_table(gpu["arch"], lib_dirs, meta.get("hipblaslt_version"))
    if table:
        cfg["env"] = {"STRATA_HIPBLASLT_TUNING": str(table)}
        if GFX1103_OPT_IN and gfx_arch_is(gpu["arch"], "gfx1103"):
            # the 780M's KFD queues are evicted (a GPU reset) while transparent huge pages move the pinned expert arena
            cfg.setdefault("env", {})["STRATA_NO_ARENA_THP"] = "1"
        if resident:   # ROCm: large page-locked host allocations can fail or be slow for the CPU; keep the copy pageable
            cfg.setdefault("env", {})["STRATA_RESIDENT_PIN"] = "0"
    if gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]                      # the engine is told this card (issue #51)
        cfg["gpus_asked"] = True                       # chosen at setup: not asked again at start
    if multi:                                          # a layer split across these cards (the server adds the flag)
        cfg["gpu"] = multi
        cfg["layer_split"] = a.layer_split or "auto"
        ok(f"layer split across GPUs {multi} ({cfg['layer_split']})")
        recommend_remote_expert_opt(cfg, off=a.no_remote_expert_opt)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if draft_vocab:
        cfg["draft_vocab"] = draft_vocab
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    # #465: requests at once - written only when given (else an earlier "parallel" is carried over); a recommendation
    streaming = "--kv-resident" in args
    if a.parallel is not None:
        if a.parallel >= 2:
            cfg["parallel"] = a.parallel
            for i, line in enumerate(parallel_note(a.parallel, [g.get("vram_gb", 0.0) for g in chosen],
                                                   MODELS[model]["arena_gb"], ctx, kv, streaming)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    if vision != "none":
        old_cfg = ROOT / f"strata-{tag.lower()}.json"
        vt = vision_tokens(a.vision_tokens, old_cfg if old_cfg.is_file() else adopted)
        # the encoder is the CPU one (the AMD engine has no GPU image encoder): "gpu" stays False for the server
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj), "model": str(shards[0]), "gpu": False,
                         "max_tokens": vt, "threads": max(1, (os.cpu_count() or 8) // 2)}
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    cal = setup_calibration(cfg)                       # #566: the calibration saved for this PC and model
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    else:                                              # #642: measured counts (a calibration) win over the rule
        cfg["args"] = recommend_pool_workers(cfg["args"])
        for line in two_socket_note(cpu_sockets()):    # bench tips: text only, the config is not touched
            say("  " + line)
    for line in bench_tips(cfg["args"], cfg.get("env"), ram, MODELS[model]["ram_gb"]):
        say("  " + line)
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None)
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run ./setup.sh (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.parallel is None:                             # #465: the opt-in, said once (nothing changes)
        for line in parallel_note(None, [g.get("vram_gb", 0.0) for g in chosen], MODELS[model]["arena_gb"], ctx, kv,
                                  "--kv-resident" in cfg["args"]):
            say("  " + line)
    if a.no_start:
        return 0
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
