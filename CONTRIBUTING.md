# Contributing to Yojit

Bug fixes, new backend support, better hardware-detection heuristics, and
documentation improvements are all welcome.

Before opening a PR for anything beyond a small fix, please open an issue
first to discuss the approach -- this is a young project and the
architecture (backend abstraction, PC-spec-aware tuning, resource-fit tiers)
is still settling.

## Developing Yojit

Requirements: Python 3.9+, pipx (recommended).

```bash
git clone https://github.com/abhisheksharma0994/yojit.git
cd yojit
pipx install -e .
pip install -e ".[dev]"
```

Run the test suite:
```bash
pytest -v
```

The test suite is fully offline by default (mocked filesystem/network) and
must stay that way -- no test should touch your real `~/.config/opencode`
or download a real model. See `tests/conftest.py` for the isolation
fixtures.

## Guiding principles (please read before changing core logic)

- **No hardcoded model recommendations.** Model suggestions must always come
  from the live Hugging Face ranking + RAM-fit math in `hf_explore.py` /
  `classify.py`, never a fixed list of "known good" models.
- **Backend parity.** Whatever MLX supports, llama.cpp should support too
  (and vice versa) -- both are first-class, PC-spec-aware backends. See the
  README's "Backend parity" section.
- **No fallback model storage location.** `manifest.models_root()` must
  never silently pick a directory the user didn't choose -- see its
  docstring.
- **Every safety threshold should be evidence-based**, not a guess. If you
  change a memory-safety constant in `classify.py`, say what real crash or
  benchmark data backs the new number.
