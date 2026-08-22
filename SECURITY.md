# Security

## Threat model

Yojit is a local tool that runs entirely on your own machine: it downloads
model weights from Hugging Face, starts a local inference server
(`mlx_lm.server` or `llama-server`) bound to `localhost`, and launches
[opencode](https://opencode.ai) against it. There is no telemetry, no
account system, and no remote server component.

### Network access

Yojit only talks to:
- **Hugging Face** (`huggingface.co`), to search for and download models.
- **Homebrew / PyPI**, indirectly, when installing prerequisites
  (`mlx-lm`, `llama-server`, `opencode`).

It is offline-by-default for serving: if Hugging Face is unreachable, it
runs entirely from the local model cache (see `server.apply_offline_posture`
in `src/yojit/server.py`).

### Local server

The inference server it starts binds to `localhost` only and is not
authenticated -- this mirrors how `mlx_lm.server` and `llama-server`
themselves behave. Do not expose the port they bind to (default `8080`) to
an untrusted network.

### Out of scope

- Vulnerabilities in `mlx-lm`, `llama.cpp`, or `opencode` themselves --
  report those upstream.
- Anything requiring local code execution as a precondition (if an attacker
  can already run arbitrary code on your machine, downloading a model isn't
  the interesting part).

## Reporting a vulnerability

Please open a private security advisory via GitHub
(Security tab -> "Report a vulnerability") rather than a public issue.
