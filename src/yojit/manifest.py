"""models.json: the single source of truth for installed models."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

MODELS_ROOT_ENV_VAR = "YOJIT_HOME"


def _repo_root_if_dev_checkout() -> Path | None:
    """If this package is running from a real source checkout (an editable
    `pip install -e .` / `pipx install -e .` from a git clone), returns the
    repo root -- identified by a pyproject.toml sitting two directories above
    this file (src/yojit/manifest.py -> src/yojit -> src ->
    repo root). Returns None for a real installed package (e.g. from PyPI),
    where this file lives in site-packages and no such repo exists."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else None


def models_root() -> Path:
    """Root directory for manifest.json + store/ + tier symlinks.

    Priority: YOJIT_HOME env var (explicit override, and what makes
    this module unit-testable without touching a developer's real home
    directory) > <repo>/models/ when running from a dev checkout (keeps
    everything self-contained in one folder while hacking on the tool,
    gitignored so downloaded weights never get committed).

    There is deliberately NO fallback location (e.g. ~/Models) -- a real
    pip/pipx install of a built package has no repo folder at runtime, and
    silently picking some other directory in that case would be a surprise.
    Instead this raises a clear, actionable error telling the user to set
    YOJIT_HOME explicitly."""
    override = os.environ.get(MODELS_ROOT_ENV_VAR)
    if override:
        return Path(override)
    repo_root = _repo_root_if_dev_checkout()
    if repo_root:
        return repo_root / "models"
    raise RuntimeError(
        "Could not determine where to store models: this does not look like "
        "a dev checkout (no pyproject.toml found alongside the installed "
        f"package), so there is no repo-local models/ folder to use. Set "
        f"{MODELS_ROOT_ENV_VAR} to an explicit path, e.g.:\n"
        f"  export {MODELS_ROOT_ENV_VAR}=~/Models/yojit"
    )


def manifest_path() -> Path:
    return models_root() / "manifest.json"


def store_root() -> Path:
    return models_root() / "store"


def load() -> dict:
    p = manifest_path()
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "default_model": None, "models": {}}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"schema_version": SCHEMA_VERSION, "default_model": None, "models": {}}


def save(data: dict) -> None:
    models_root().mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(data, indent=2) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def add_model(model_id: str, entry: dict) -> None:
    data = load()
    entry.setdefault("added_at", now_iso())
    entry.setdefault("verified", False)
    data.setdefault("models", {})[model_id] = entry
    if not data.get("default_model"):
        data["default_model"] = model_id
    save(data)


def remove_model(model_id: str) -> dict | None:
    data = load()
    entry = data.get("models", {}).pop(model_id, None)
    if data.get("default_model") == model_id:
        remaining = list(data.get("models", {}).keys())
        data["default_model"] = remaining[0] if remaining else None
    save(data)
    return entry


def set_default(model_id: str) -> None:
    data = load()
    if model_id not in data.get("models", {}):
        raise KeyError(f"{model_id} is not installed")
    data["default_model"] = model_id
    save(data)


def get_default() -> str | None:
    return load().get("default_model")


def get_model(model_id: str) -> dict | None:
    return load().get("models", {}).get(model_id)


def list_models() -> dict:
    return load().get("models", {})
