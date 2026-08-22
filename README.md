# Yojit — Local AI, set up right.

One command to discover, install, run, and keep local LLMs (MLX + llama.cpp) wired into [opencode](https://opencode.ai).

## What it does

- Classifies every model as `low`/`medium`/`high` risk based on real memory-safety math, not guesswork.
- Auto-picks the best-fitting quantization for your exact machine, for both MLX and llama.cpp equally.
- Searches Hugging Face, filters out non-chat models, and ranks by downloads plus agentic/tool-use signals.
- Keeps opencode's model list in sync automatically and launches straight into it, bound to the right model.
- Self-healing: installs missing prerequisites (`mlx-lm`, `llama-server`, `opencode`) on demand.

## Prerequisites

Python 3.9+ and pipx. If you don't have them:
```bash
brew install python3 pipx
```
Everything else (`mlx-lm`, `llama-server`, `opencode`) is installed on demand by `yojit init` itself.

## Install and run

```bash
git clone <this-repo>
cd yojit
```
```bash
pipx install -e .
```
```bash
yojit init
```

`pipx` installs `yojit` into its own isolated environment and puts the `yojit` command on your PATH — no system-Python errors, nothing to activate.

Prefer plain `pip` instead? `pip install -e .` works too, but on newer macOS/Homebrew Python it will refuse with an "externally-managed-environment" error unless you add `--break-system-packages`:
```bash
pip install --break-system-packages -e .
```

## Quick start

`yojit init` already covers first-time setup, suggests a model that fits your machine, and offers to launch straight into opencode. Beyond that:

Search Hugging Face and install a model:
```bash
yojit explore --query qwen3.6
```
Launch a model and open opencode:
```bash
yojit serve
```

## Where models are stored

When running from a dev checkout (this repo, installed editable), models live in `models/` right here in the repo — self-contained, gitignored so downloaded weights never get committed (only `models/.gitkeep` is tracked, to keep the folder present after a fresh clone).

If you install this as a real package (`pip`/`pipx` install of a built distribution rather than a git clone), there is no repo folder at runtime to default to. In that case you must set the location explicitly:
```bash
export YOJIT_HOME=/wherever/you/want
```
Running any command without a detectable repo checkout and without this variable set fails with a clear error telling you to set it — it will never silently pick a location on your machine.

## Commands

| Command | What it does |
|---|---|
| `init` | First-time setup: detect hardware, install prerequisites, sync config |
| `explore [--query TEXT]` | Search Hugging Face, rank by downloads, install |
| `install <repo> [--bits N] [--file X.gguf]` | Install a specific model |
| `list` | Show installed models, tiers, sizes, limits |
| `use <model>` | Set the default model |
| `serve [model] [--no-open]` | Launch a model (and opencode) |
| `stop` / `status` | Server lifecycle |
| `remove <model>` | Uninstall a model |
| `sync` | Force a config re-sync |
| `doctor` | Diagnostics |
| `upgrade` | Upgrade the tool, runtimes, and opencode |

## Safety model

Every installed model gets a `low`/`medium`/`high` tier based on what fraction of your *total* RAM its weights would occupy — not its absolute size. The same tier logic applies at any RAM size, from an 8GB machine to a 128GB one: `low` and `medium` are considered safe to serve; `high` is flagged RISKY in the `serve` picker and requires an explicit confirmation before launching.

The dividing line (~50% of RAM) isn't a theoretical guess — it's empirical. It was found by testing MLX models on a 24GB Apple Silicon Mac: every model whose weights exceeded that line crashed with a Metal out-of-memory error, regardless of context size, while models comfortably under it were rock-solid. Because the rule is expressed as a fraction rather than a fixed GB number, that finding carries over to any RAM size. See `src/yojit/classify.py` for the exact math.

The same 50% threshold is applied to llama.cpp as a conservative default, with the analogous safety flags in place (`--ctx-size`, `--mlock`, `--parallel 1`) — but llama.cpp's own memory behavior under real OOM pressure hasn't been independently stress-tested the way MLX's was. Treat the llama.cpp threshold as reasoned-by-analogy, not (yet) crash-verified on that backend specifically.

## Backend parity (MLX vs. llama.cpp)

MLX and llama.cpp are treated as first-class, symmetric citizens — not "MLX is the real one and llama.cpp is an afterthought":

- **Quant/bit auto-selection**: MLX picks the largest bit-width repo variant that still fits comfortably (`hf_explore.pick_best_fit`); GGUF picks the largest `.gguf` file in a repo that still fits comfortably (`hf_explore.pick_best_gguf_file`) — same principle, backend-appropriate mechanism (bit-width label vs. real file size, since GGUF quant-naming conventions vary too much across converters to parse reliably).
- **Architecture-aware context/output limits**: MLX reads `config.json`; GGUF parses the model's own binary header directly (`gguf_meta.py`) for real layer count, head count, and native context — not a guess.
- **Prerequisite self-install**: `mlx-lm` via pip, `llama-server` via Homebrew, `opencode` via Homebrew on macOS — actually installed on demand, not just detected and printed. Non-macOS or no Homebrew gets a clear manual-install pointer instead of a guessed command. (This doesn't cover Python or `pip`/`pipx` themselves — by the time any `yojit` command runs, the tool is already installed, so those are moot at that point, not gaps.)
- **`upgrade`**: upgrades both `mlx-lm` and `llama.cpp` when present, not just one.
- **Memory-safety launch flags**: MLX uses `--prefill-step-size`/`--prompt-cache-bytes` (tuned against real Metal OOM crashes); llama.cpp uses `--ctx-size`/`--mlock`/`--parallel 1` (the last one pins llama-server to a single request slot — it defaults to 4, each with its own KV cache sized to `--ctx-size`, which would silently multiply real memory usage past what the RAM math assumed).
- **All launch parameters are PC-spec-aware, for both backends**: every knob beyond context/output size is recomputed fresh from this machine's actual RAM headroom and CPU core count on every `serve` call (`classify.compute_launch_tuning`), never a fixed constant baked in for one developer's machine. Headroom (`RAM - model weight size - reserved OS memory`) is bucketed into tiers that scale prefill/batch chunking up as more memory is free to use, and down toward conservative values as it gets tight:
  - **MLX**: `--prefill-step-size` (512-8192, larger prefill chunks are faster but need more headroom) and `--prompt-cache-bytes` (scales with headroom, floored/ceilinged to sane bounds) both scale with headroom; `--decode-concurrency`/`--prompt-concurrency` are pinned to 1 (no backend here has verified-safe concurrent-request memory accounting yet); `--max-tokens` enforces the computed output limit server-side.
  - **llama.cpp**: `--threads` is CPU-core-count-minus-one (leaves a core for the OS); `--gpu-layers 999` forces full GPU offload; `--batch-size`/`--ubatch-size` scale with the same headroom tiers as MLX's prefill chunking, since they serve the same purpose (how much of the prompt gets processed per step).

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

The test suite (100+ tests) covers every module with mocked filesystem/network access — no test touches your real `~/Models` or `~/.config/opencode/opencode.json`. CI runs on every push via GitHub Actions across Linux and Apple Silicon runners.

## License

MIT
