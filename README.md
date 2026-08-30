# Yojit — Local AI, set up right.

One command to discover, install, run, and keep local LLMs (MLX + llama.cpp) wired into [opencode](https://opencode.ai).

## What it does

- Classifies every model as `low`/`medium`/`high` risk based on real memory-safety math, not guesswork.
- Auto-picks the best-fitting quantization for your exact machine, for both MLX and llama.cpp equally.
- Searches Hugging Face, filters out non-chat models, and ranks by downloads plus agentic/tool-use signals.
- Keeps opencode's model list in sync automatically and launches straight into it, bound to the right model.
- Self-healing: installs missing prerequisites (`mlx-vlm`, `llama-server`, `opencode`) on demand, into an isolated venv it owns (`~/.yojit/venv`) rather than system Python.
- Every MLX-format model — vision or plain text, dense or MoE — is served by `mlx-vlm`, a single universal backend.
- Per-model launch knobs LM Studio exposes in its Load panel — seed, KV-cache quantization, max concurrent predictions, context length — settable via `yojit config`.

## Prerequisites

Python 3.9+ and pipx. If you don't have them:
```bash
brew install python3 pipx
```
Everything else (`mlx-vlm`, `llama-server`, `opencode`) is installed on demand by `yojit init` itself.

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

When running from a dev checkout, models live in `models/` in the repo — gitignored, so weights never get committed.

If installed as a real package instead of a git clone, set the location explicitly:
```bash
export YOJIT_HOME=/wherever/you/want
```
Without a detectable repo checkout or this variable set, commands fail with a clear error rather than picking a location silently.

## Commands

| Command | What it does |
|---|---|
| `init` | First-time setup: detect hardware, install prerequisites, sync config |
| `explore [--query TEXT]` | Search Hugging Face, rank by downloads, install |
| `install <repo> [--bits N] [--file X.gguf]` | Install a specific model |
| `list` | Show installed models, tiers, sizes, limits |
| `use <model>` | Set the default model |
| `config <model> [--seed N] [--kv-cache-quant X] [--kv-group-size N] [--quantized-kv-start N] [--max-concurrent-predictions N] [--context N]` | Set per-model launch knobs |
| `serve [model] [--no-open]` | Launch a model (and opencode) |
| `stop` / `status` | Server lifecycle |
| `remove <model>` | Uninstall a model |
| `sync` | Force a config re-sync |
| `doctor` | Diagnostics |
| `upgrade` | Upgrade the tool, runtimes, and opencode |

## Safety model

Every installed model gets a `low`/`medium`/`high` tier based on what fraction of your *total* RAM its weights would occupy — not its absolute size. `low`/`medium` are safe to serve; `high` is flagged RISKY in the `serve` picker and needs explicit confirmation.

The ~50% dividing line is empirical: models above it risk an OOM crash regardless of context size, while models comfortably under it run reliably. Since it's a fraction, not a fixed GB number, it holds at any RAM size. See `src/yojit/classify.py` for the exact math.

The same threshold applies to llama.cpp as a conservative default, with its own equivalent safety flags (`--ctx-size`, `--mlock`, `--parallel 1`).

## Model coverage: mlx-vlm is the only MLX backend

Every MLX-format model — vision or plain text, dense or MoE — is served by `mlx-vlm`'s server. It's a strict superset of the older text-only `mlx-lm` server, with real CLI flags for KV-cache quantization, context-length caps, and max-concurrent-predictions that `mlx-lm` doesn't have — so there's no separate vision/text backend split anymore.

The runtime installs into `~/.yojit/venv`, a venv yojit owns exclusively, never system or Homebrew-managed Python — keeping install/upgrade side effects fully contained.

## Per-model settings (`yojit config`)

Mirrors the knobs LM Studio exposes in its model Load panel:

| Setting | MLX (`mlx-vlm`) | llama.cpp (GGUF) |
|---|---|---|
| Context length | `--max-kv-size`, real cap | `--ctx-size`, real cap |
| KV-cache quantization | `--kv-bits` / `--kv-group-size` / `--quantized-kv-start` | `--cache-type-k` / `--cache-type-v` |
| Max concurrent predictions | `--max-num-seqs` | `--parallel` |
| Seed | per-request only, no launch flag | `--seed`, real launch flag |

An override a model's current backend can't act on is simply inert, not an error. `yojit config <model>` with no flags prints the model's current overrides and context.

**KV-cache quantization defaults are computed per model per machine, never hardcoded.** `classify.default_kv_cache_overrides()` works out the real bytes-per-token cost from the model's own architecture against this machine's real headroom, and picks the highest-precision bit-width that still fits. `yojit config` always wins when explicitly set.

## Backend parity (MLX vs. llama.cpp)

MLX and llama.cpp are treated as first-class, symmetric citizens:

- **Quant/bit auto-selection**: MLX picks the largest bit-width variant that still fits comfortably; GGUF picks the largest `.gguf` file that fits, keyed on real file size since quant-naming conventions vary too much to parse.
- **Architecture-aware context/output limits**: MLX reads `config.json`; GGUF parses the model's own binary header (`gguf_meta.py`) for real layer/head counts and native context.
- **Prerequisite self-install**: `mlx-vlm` into yojit's isolated venv, `llama-server` and `opencode` via Homebrew — actually installed on demand, not just detected and printed.
- **`upgrade`**: upgrades both `mlx-vlm` and `llama.cpp` when present.
- **Memory-safety launch flags**: MLX uses `--prefill-step-size`/context caps; llama.cpp uses `--ctx-size`/`--mlock`/`--parallel 1` (pinned to one request slot, since each extra slot multiplies KV-cache memory use).
- **All launch parameters are spec-aware, for both backends**: every knob beyond context/output is recomputed fresh from real RAM headroom and CPU core count on every `serve` call, never a fixed constant.

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

The test suite covers every module with mocked filesystem/network access — no test touches your real `~/Models` or `~/.config/opencode/opencode.json`. CI runs on every push via GitHub Actions across Linux and Apple Silicon runners.

## License

MIT
