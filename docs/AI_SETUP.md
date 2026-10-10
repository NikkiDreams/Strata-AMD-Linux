# Installing Strata for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked to
set up Strata on their PC. Strata runs the Qwen3.8-Flash-Next model locally on one AMD graphics card plus system RAM,
on Linux, and serves an OpenAI- and Anthropic-compatible API on `http://127.0.0.1:8080`. The human-oriented version of
everything below is [INSTALL.md](INSTALL.md). This fork runs on AMD cards on Linux only: a PC with an NVIDIA card or
with Windows is refused with a plain message.

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

## 0. Ground rules

- **Never expose the server without an API key.** Keep the default `--host 127.0.0.1`. Only if the user asks for
  access from other devices, use `--host 0.0.0.0` **together with** `--api-key <a long random secret>`, and give the
  user the key. Never put a tunnel or port forward in front of a server without a key.
- Do not change the user's system beyond what setup does (setup installs Python for the user account if needed,
  and everything else inside the Strata folder and its `Strata-data` folder). Ask before installing drivers.
- The model download is ~70 GB (94 GB for Unsloth's UD-IQ4_XS). Confirm the user is fine with that before you start,
  especially on a metered connection.
- Setup is long-running (an hour or more on a slow connection, plus 10-20 minutes for the engine build). Run it in the
  background or with a long timeout and poll its output; do not kill it because it is quiet for a while. It is
  resumable: running the same command again continues where it stopped.

## 1. Check the PC

The quickest way is setup's own check, which prints the AMD GPU(s), the driver, RAM, CPU, free disk and whether this PC
can run Strata, and installs nothing except Python and the `.venv` it needs to run (step 2 gets the repository first):

```
./setup.sh --check
```

To check by hand:

| What | Linux |
| --- | --- |
| OS | `cat /etc/os-release` |
| AMD GPU | `lspci \| grep -i -E 'vga\|display'`; VRAM: `cat /sys/class/drm/card*/device/mem_info_vram_total` (bytes); `rocm-smi` only if ROCm is installed |
| RAM | `free -g` |
| Free disk | `df -h .` |

Requirements (details: [INSTALL.md](INSTALL.md#what-you-need)):

- **GPU:** AMD Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700,
  RX 6800 / 6900 series, RX 6700 XT, RX 7600 / 7600 XT, or the Ryzen AI Max "Strix Halo" APU; 12 GB of VRAM or more.
  A card sharing the model needs 8 GB or more (docs/MULTI_GPU.md). Integrated Radeon graphics (other than Strix Halo)
  are not supported; the older cards and Strix Halo are experimental paths the user opts into
  ([OLDER_GPUS.md](OLDER_GPUS.md), [STRIX_HALO.md](STRIX_HALO.md)).
- **Driver:** the kernel's amdgpu driver, and this user must be able to open `/dev/kfd` and the render nodes (setup
  says so and gives the `usermod` command). If the driver is missing or too old, tell the user to update the kernel or
  the distribution's AMD driver package and restart; do not change drivers yourself unless they ask.
- **RAM:** 32 GB or more (see step 3). **Disk:** ~80 GB free, ideally on an NVMe SSD. **CPU:** x86-64 with AVX2.
  Without AVX2 (Xeon E5 v1/v2 and older) setup still installs, as an experimental and slow build it compiles on the PC
  (10-20 minutes): tell the user that before starting ([INSTALL.md](INSTALL.md#older-cpus-experimental)).
- **OS:** Linux (Ubuntu 22.04/24.04 are fully automatic). Windows is not supported by this fork.

If the PC does not meet them, say which part is missing and stop.

## 2. Get Strata

```
git clone https://github.com/Niko1221/Strata.git
```

Without git: download https://github.com/Niko1221/Strata/archive/refs/heads/main.zip and unzip it. Pick a drive with
enough free space: the model files go to a `Strata-data` folder **next to** the Strata folder (or `--data-dir <path>`
on another drive).

## 3. Pick the model

By the PC's RAM (ask the user whether they mainly want it for code - then the Coder is a good pick at any RAM size):

| RAM | `--family` / `--model` | Notes |
| --- | --- | --- |
| 32 GB | `--family coder` (size IQ1_M) | for code; with a 24 GB card Q2_0 and IQ2_XS also run (setup picks the low-RAM mode) |
| 48 GB | `--family qwen --model IQ2_XS` | or `Q2_0` (fastest) |
| 64 GB | `--family qwen --model IQ2_XS` (recommended) | `IQ3_XXS` / `IQ3_S` are slower and a bit better |
| 96 GB+ | `--family qwen --model IQ3_S` | `--family unsloth --model UD-IQ4_XS` (~4-bit, 94 GB, part of the experts read from the SSD under ~80 GB of RAM) |

`--family swift` (Swift 1.5, a fine-tune that thinks shorter; sizes IQ2_XS, IQ3_XXS, IQ3_S — its IQ3_S tier is
included at the pinned revision, #1651) is the alternative to `qwen`. With `--yes` and no `--model`, setup picks the
recommended size for the RAM itself. More: [MODELS.md](MODELS.md).

## 4. Run setup without questions

From the Strata folder:

```
./setup.sh --yes --family qwen --model IQ2_XS --no-start
```

The flags (all of them: `./setup.sh --help`):

| Flag | Meaning |
| --- | --- |
| `--yes` | take the recommended answer to every question (no prompts) |
| `--family qwen\|swift\|coder\|unsloth` | the model version |
| `--model Q2_0\|IQ2_XS\|IQ3_XXS\|IQ3_S\|IQ1_M\|UD-IQ4_XS` | the size (the Coder is IQ1_M, Unsloth UD-IQ4_XS) |
| `--context N` | context in tokens; default by VRAM: 32768 under 14 GB, 65536 under 20 GB, else 131072 |
| `--vision no\|none\|cpu` | read pictures on the CPU (`--vision cpu`); `yes` asks for the GPU encoder, which this fork does not build, so images stay off and setup says so |
| `--gpu N` / `--gpus 0,1` / `--gpus all` | one card, or several sharing the model (default: the card with the most memory) |
| `--layer-split "18"` | with `--gpus`: where each later card's layers start |
| `--data-dir PATH` | where the 70-120 GB of model files go |
| `--port N` | the server port (default 8080) |
| `--host H --api-key K` | listen beyond this PC; **only together with a key** |
| `--no-start` | install only, do not start the server |
| `--setup` | install another model or change settings of an installed one |
| `--check` | only check the PC |

`--no-start` is recommended for agents: the server runs in the foreground until its window is closed, which would
block your shell. Start it separately in step 6.

Notes:
- **Linux:** setup uses `sudo apt` (or dnf/pacman) to install Python with venv if it is missing, and needs
  `build-essential` and `git` to compile the engine. You cannot type the user's password: if a `sudo` step is needed,
  ask the user to run it (e.g. `sudo apt install python3-venv build-essential git`) and then rerun setup.
- **ROCm:** setup installs AMD's ROCm Python wheels into `.venv` (~10 GB, no sudo) unless a system ROCm 7 with `hipcc`
  and hipBLAS is already there; the engine is then compiled here for the card (10-20 minutes, once, and again after a
  `git pull` that changes the engine's source).
- **Images:** the CPU image encoder only (docs/AMD_HIP.md).

## 5. While it downloads, tell the user

- It downloads about 70 GB from Hugging Face; how long depends on their connection (at 100 Mbit/s roughly 1.5-2
  hours). It can be stopped and continues where it left off.
- Then it prepares the model for their PC (a few minutes) and compiles the engine for the card (10-20 minutes, once).
- **When the model starts, the PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
  loads 35-55 GB into RAM and locks part of it for the graphics card. That is normal; they should not close the window.
- Nothing leaves their PC: the model runs locally.

## 6. Start the server

Setup prints the start script it wrote (`start script: run-<model>.sh`). The name is the family tag plus the size,
lower case: `run-iq2_xs`, `run-swift-iq2_xs`, `run-coder-iq1_m`, `run-unsloth-ud-iq4_xs`.

```
nohup ./run-iq2_xs.sh > strata-server.out 2>&1 &
```

Or `./setup.sh` without flags, which starts the installed model (it asks which one when several are installed; add
`--yes` to take the first). The server opens the user's browser on `http://127.0.0.1:8080` when it is ready. Closing
its window (or stopping the process) stops the model.

## 7. Verify

The HTTP server answers once the model is loaded (30-90 s on later starts, a few minutes the first time). Poll:

```
curl http://127.0.0.1:8080/health
```

It returns `{"status": "ok", "model": ..., "max_context": ..., "images": ..., "api_key": ..., "loaded": true, ...}`.
Then:

```
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

With an API key, add `-H "Authorization: Bearer <key>"`. The server window prints a `ready: http://127.0.0.1:8080/v1`
line when it is ready; the engine log is `strata-<model>.log` in the Strata folder.

## 8. Connect the user's apps

- **Browser:** `http://127.0.0.1:8080` - Chat, a live Monitor, and About (settings and addresses).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8080/v1`, any API key (or the configured one),
  any model name.
- **Anthropic-compatible apps:** `http://127.0.0.1:8080/v1/messages`.
- **Codex CLI and other Responses API apps:** `http://127.0.0.1:8080/v1/responses` (stateless; Codex's
  `config.toml`: [DETAILS.md](DETAILS.md#the-responses-api-and-codex-cli)).
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (Strata ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the configured key).
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"` (default high).
- Strata answers one request at a time; `"parallel": N` in the model's config (or setup `--parallel N`) decodes up to
  N together ([BATCHING.md](BATCHING.md)). API details: [DETAILS.md](DETAILS.md#using-it).

## 9. When something fails

| Symptom | What to do |
| --- | --- |
| `no AMD GPU Strata can use` | The amdgpu driver lists no GPU, or the card is not one Strata supports (step 1). Check the card and the kernel, and that the user can open `/dev/kfd`; rerun setup. |
| `this user cannot open /dev/kfd` | `sudo usermod -aG render,video $USER`, then log out and in again; rerun. |
| Download or install stopped | Rerun the same setup command; it continues. |
| `port 8080 is already in use` | Strata is already running (check `/health`), or use `--port 8081`. |
| Very slow, disk busy, or "the engine stopped unexpectedly" | Not enough free RAM: close programs, or set up a smaller size (`--setup --model Q2_0` or IQ2_XS). |
| `prompt ... exceeds the context` | Rerun setup with `--setup --context <bigger>`. |
| Python or build tools could not be installed | Install what the message names (links are printed), then rerun. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and the [full table](DETAILS.md#troubleshooting). If it still fails,
collect `strata-<model>.log` and the setup output, and suggest an issue at https://github.com/Niko1221/Strata/issues.

## Alternative: the MCP server

Strata also has an MCP server, so an AI tool can check, install, start and stop Strata through tool calls
(`strata_status`, `strata_models`, `strata_install`, `strata_start`, `strata_stop`, `strata_logs`) instead of the
shell commands above. See [MCP_SERVER.md](MCP_SERVER.md).
