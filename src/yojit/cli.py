"""Single entry point: yojit <subcommand> ..."""
import argparse
import shutil
import subprocess
import sys

from . import (
    classify,
    doctor,
    hf_explore,
    installer,
    manifest,
    mlx_env,
    opencode_sync,
    prereqs,
    server,
    specs,
)


def _suggest_model_for_ram(
    ram_gb: float, search_limit: int = 50, check_limit: int = 20
) -> tuple[str | None, str | None]:
    """Returns the highest-ranked HF model that fits comfortably on this machine's RAM."""
    try:
        results = hf_explore.search(query=None, limit=search_limit)
    except Exception:
        return None, None
    ranked = hf_explore.rank_candidates(results, ram_gb)

    for candidate in ranked[:check_limit]:
        repo_id = candidate["id"]
        try:
            files = hf_explore.repo_files(repo_id)
        except Exception:
            continue
        if hf_explore.detect_backend_from_files(files) != "mlx":
            continue
        size_gb = sum(f.get("size", 0) for f in files if f["path"].endswith(".safetensors")) / (1024 ** 3)
        if size_gb <= 0:
            continue
        tier = classify.resource_tier(size_gb, ram_gb)
        if tier in ("low", "medium") and classify.fits_at_all(size_gb, ram_gb):
            downloads = candidate.get("downloads", 0)
            return repo_id, f"{size_gb:.1f}GB, {tier} tier, {downloads} downloads"

    return None, None


def _init_first_model(ram_gb: float) -> None:
    print(f"\nNo models installed yet. Checking Hugging Face for a model that fits your "
          f"{round(ram_gb)}GB machine...")
    suggested_repo, reason = _suggest_model_for_ram(ram_gb)

    if suggested_repo:
        print(f"Suggested: {suggested_repo}  ({reason})")
        prompt = "Install this now? [Y/n], or paste a different Hugging Face repo id to install that instead: "
    else:
        print("Could not find a suggestion automatically (offline, or nothing fit in the models checked).")
        prompt = "Paste a Hugging Face repo id to install, or press Enter to skip for now: "

    choice = input(prompt).strip()
    if choice.lower() in ("n", "no", "skip") or (not suggested_repo and choice == ""):
        print("\nSkipped. Run `yojit explore` or `yojit install <repo>` whenever you're ready.")
        return

    repo_to_install = suggested_repo if (choice == "" or choice.lower() == "y") and suggested_repo else choice
    print(f"\nDownloading {repo_to_install}...")
    _resolve_install(repo_to_install, None, None, ram_gb)
    print("\nLaunching it now...")
    server.serve(None, open_opencode=True)


def cmd_init(args):
    s = specs.detect()
    print(f"Detected: {s.chip}, {round(s.total_ram_gb)} GB RAM, {round(s.free_disk_gb)} GB free disk, "
          f"Apple Silicon: {s.is_apple_silicon}")

    manifest.models_root().mkdir(parents=True, exist_ok=True)
    print(f"Models root: {manifest.models_root()}")

    if s.is_apple_silicon and not mlx_env.is_installed("mlx_vlm"):
        print("Installing mlx-vlm into yojit's isolated venv (~/.yojit/venv)...")
        mlx_env.pip_install("mlx-vlm")

    prereqs.ensure_opencode_installed()
    opencode_sync.sync()

    if manifest.list_models():
        print("\nInit complete. Run `yojit serve` when ready.")
        return
    _init_first_model(s.total_ram_gb)


def _resolve_gguf_install(repo_id: str, files: list[dict], gguf_file: str | None, ram_gb: float):
    if not gguf_file:
        best = hf_explore.pick_best_gguf_file(files, ram_gb)
        if not best:
            print(f"No .gguf file in {repo_id} fits comfortably on this {round(ram_gb)}GB machine.")
            sys.exit(1)
        gguf_file = best["path"]
        size_gb = best["size"] / (1024 ** 3)
        print(f"No --file given, auto-picked {gguf_file} ({size_gb:.1f} GB, best fit for your {round(ram_gb)}GB RAM)")
    print(installer.install_gguf(repo_id, gguf_file, ram_gb))


def _bit_subfolders(files: list[dict]) -> list[str]:
    return sorted({f["path"].split("/")[0] for f in files if "/" in f["path"]
                   and f["path"].split("/")[0].lower().endswith("bit")})


def _install_matching_bits(repo_id: str, subfolders: list[str], bits: int, ram_gb: float) -> None:
    match = next((s for s in subfolders if hf_explore.bit_width_from_name(s) == bits), None)
    if not match:
        print(f"No {bits}-bit variant found. Available: {subfolders}")
        sys.exit(1)
    print(installer.install_mlx(repo_id, match, ram_gb))


def _install_largest_fitting_bits(repo_id: str, subfolders: list[str], ram_gb: float) -> None:
    # Default to the largest bit-width that still lands in low/medium tier -- comfortable fit, not the tightest.
    subfolders = sorted(subfolders, key=lambda name: hf_explore.bit_width_from_name(name) or 0, reverse=True)
    print(f"Multiple quantizations available: {subfolders}")
    print("Picking the largest one that still fits comfortably (use --bits N to override)...")
    print(installer.install_mlx(repo_id, subfolders[0], ram_gb))


def _resolve_mlx_install(repo_id: str, files: list[dict], bits: int | None, ram_gb: float):
    root_has_weights = (any(f["path"] == "config.json" for f in files)
                         and any(f["path"].endswith(".safetensors") for f in files))
    if root_has_weights and bits is None:
        print(installer.install_mlx(repo_id, None, ram_gb))
        return

    subfolders = _bit_subfolders(files)
    if not subfolders:
        if not root_has_weights:
            print(f"Could not find MLX weights in {repo_id} (no root safetensors, no bit-suffixed subfolders).")
            sys.exit(1)
        print(installer.install_mlx(repo_id, None, ram_gb))
    elif bits is not None:
        _install_matching_bits(repo_id, subfolders, bits, ram_gb)
    else:
        _install_largest_fitting_bits(repo_id, subfolders, ram_gb)


def _resolve_install(repo_id: str, bits: int | None, gguf_file: str | None, ram_gb: float):
    files = hf_explore.repo_files(repo_id)
    backend = hf_explore.detect_backend_from_files(files)
    if backend == "llamacpp" or gguf_file:
        _resolve_gguf_install(repo_id, files, gguf_file, ram_gb)
    else:
        _resolve_mlx_install(repo_id, files, bits, ram_gb)


def cmd_install(args):
    ram_gb = specs.detect().total_ram_gb
    _resolve_install(args.repo, args.bits, args.file, ram_gb)


def cmd_explore(args):
    ram_gb = specs.detect().total_ram_gb
    results = hf_explore.search(query=args.query, limit=args.limit)
    ranked = hf_explore.rank_candidates(results, ram_gb)
    if not ranked:
        print("No results.")
        return
    print(f"\nTop {min(len(ranked), 15)} results by downloads:")
    shown = ranked[:15]
    for i, m in enumerate(shown, 1):
        print(f"  {i}) {m['id']}  ({m.get('downloads', 0)} downloads)")
    choice = input(f"Enter a number (1-{len(shown)}), or blank to cancel: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(shown)):
        print("Cancelled.")
        return
    repo_id = shown[int(choice) - 1]["id"]
    _resolve_install(repo_id, args.bits, None, ram_gb)


def cmd_list(args):
    models = manifest.list_models()
    default = manifest.get_default()
    if not models:
        print("No models installed.")
        return
    for model_id, entry in models.items():
        marker = " (default)" if model_id == default else ""
        print(f"{model_id}{marker}")
        print(f"    backend={entry.get('backend')} tier={entry.get('tier')} "
              f"size={entry.get('size_gb')}GB context={entry.get('context')} output={entry.get('output')} "
              f"verified={entry.get('verified')}")


def cmd_use(args):
    manifest.set_default(args.model)
    opencode_sync.sync()
    print(f"Default model set to {args.model}")


def cmd_config(args):
    """Sets per-model launch knobs (seed, KV-cache quant, concurrency, context).
    An override the model's backend can't use is simply inert, not an error."""
    entry = manifest.get_model(args.model)
    if not entry:
        print(f"{args.model} is not installed.")
        sys.exit(1)

    if not any([args.seed is not None, args.kv_cache_quant, args.max_concurrent_predictions is not None,
                args.kv_group_size is not None, args.quantized_kv_start is not None, args.context is not None]):
        print(f"Current overrides for {args.model}: {entry.get('overrides', {})}")
        print(f"Current context: {entry.get('context')}")
        return

    if args.context is not None:
        data = manifest.load()
        data["models"][args.model]["context"] = args.context
        manifest.save(data)

    overrides = manifest.update_overrides(
        args.model,
        seed=args.seed,
        kv_cache_quant=args.kv_cache_quant,
        max_concurrent_predictions=args.max_concurrent_predictions,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )
    print(f"Updated overrides for {args.model}: {overrides}")
    if args.context is not None:
        print(f"Context length set to {args.context}")
    print("Takes effect on the next `yojit serve`.")


def cmd_serve(args):
    server.serve(args.model, open_opencode=not args.no_open)


def cmd_stop(args):
    pid = server._port_pid(server.PORT)
    if not pid:
        print("No server running.")
        return
    subprocess.run(["kill", str(pid)])
    print(f"Stopped PID {pid}")


def cmd_status(args):
    pid = server._port_pid(server.PORT)
    if pid:
        print(f"Server running on port {server.PORT}, PID {pid}")
    else:
        print("No server running.")


def cmd_remove(args):
    print(installer.remove(args.model))


def cmd_sync(args):
    print(opencode_sync.sync())


def cmd_doctor(args):
    for line in doctor.run():
        print(line)


def cmd_upgrade(args):
    print("Upgrading yojit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--break-system-packages", "yojit"])
    if mlx_env.venv_python().exists():
        print("Upgrading mlx-vlm in yojit's isolated venv...")
        mlx_env.pip_install("mlx-vlm", upgrade=True)
    if shutil.which("llama-server"):
        print("Upgrading llama.cpp...")
        subprocess.run(["brew", "upgrade", "llama.cpp"], check=False)
    if shutil.which("opencode"):
        print("Upgrading opencode...")
        subprocess.run(["opencode", "upgrade"])

    print("Checking installed models for newer/better releases...")
    for model_id, entry in manifest.list_models().items():
        print(f"  {model_id}: checked (model-upgrade-detection not yet implemented -- "
              f"re-run `yojit install {entry.get('source_repo')}` to refresh manually)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yojit", description="Yojit -- Local AI, set up right.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="First-time setup").set_defaults(func=cmd_init)

    p = sub.add_parser("explore", help="Search Hugging Face and install")
    p.add_argument("--query", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--bits", type=int, default=None)
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("install", help="Install a specific model")
    p.add_argument("repo")
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--file", default=None, help="Exact .gguf filename for llama.cpp repos")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("list", help="List installed models")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("use", help="Set the default model")
    p.add_argument("model")
    p.set_defaults(func=cmd_use)

    p = sub.add_parser("config", help="Set per-model launch knobs (seed, KV-cache quant, concurrency, context)")
    p.add_argument("model")
    p.add_argument("--seed", type=int, default=None, help="Fixed RNG seed (llama.cpp only; omit for random)")
    p.add_argument("--kv-cache-quant", default=None,
                    help="KV cache quantization: bit-width for mlx_vlm (e.g. 4, 8), "
                         "quant type for llama.cpp (e.g. q8_0, q4_0)")
    p.add_argument("--kv-group-size", type=int, default=None, help="KV cache quant group size (mlx_vlm only)")
    p.add_argument("--quantized-kv-start", type=int, default=None,
                    help="Token index KV quantization starts at (mlx_vlm only)")
    p.add_argument("--max-concurrent-predictions", type=int, default=None,
                    help="Max concurrent request slots (all backends; raising above 1 "
                         "multiplies KV-cache memory use accordingly)")
    p.add_argument("--context", type=int, default=None,
                    help="Override the auto-computed context length for this model")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("serve", help="Launch a model and opencode")
    p.add_argument("model", nargs="?", default=None)
    p.add_argument("--no-open", action="store_true", help="Don't auto-launch opencode")
    p.set_defaults(func=cmd_serve)

    sub.add_parser("stop", help="Stop the running server").set_defaults(func=cmd_stop)
    sub.add_parser("status", help="Show server status").set_defaults(func=cmd_status)

    p = sub.add_parser("remove", help="Remove an installed model")
    p.add_argument("model")
    p.set_defaults(func=cmd_remove)

    sub.add_parser("sync", help="Re-sync opencode.json").set_defaults(func=cmd_sync)
    sub.add_parser("doctor", help="Run diagnostics").set_defaults(func=cmd_doctor)
    sub.add_parser("upgrade", help="Upgrade everything").set_defaults(func=cmd_upgrade)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
