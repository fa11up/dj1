"""
dj1 — Pipeline Orchestrator
════════════════════════════
Runs the full 5-module pipeline end-to-end via a single command:

    uv run python main.py --config config/session.yaml
    uv run python main.py --config config/session.yaml --from-module 4
    uv run python main.py --config config/session.yaml --preview
    uv run python main.py --config config/session.yaml --quality

Modules:
  1  Ingestor        — scan local music library → tracks.json
  2  Analyzer        — BPM, key, energy, beat grid → analysis.json
  3  Spotify Bridge  — match Spotify playlist to local files → matched_tracks.json
  4  Planner         — build setlist with energy curves → session.json
  5  Renderer        — render continuous mix → output MP3/WAV

Each module can still be run individually via `uv run python -m modules.<name>`.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import yaml


# ── Colorama (optional) ──────────────────────────────────────────────────────

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def green(s):  return Fore.GREEN  + str(s) + Style.RESET_ALL
    def cyan(s):   return Fore.CYAN   + str(s) + Style.RESET_ALL
    def yellow(s): return Fore.YELLOW + str(s) + Style.RESET_ALL
    def red(s):    return Fore.RED    + str(s) + Style.RESET_ALL
    def bold(s):   return Style.BRIGHT + str(s) + Style.RESET_ALL
    def dim(s):    return Style.DIM   + str(s) + Style.RESET_ALL
except ImportError:
    def green(s):  return str(s)
    def cyan(s):   return str(s)
    def yellow(s): return str(s)
    def red(s):    return str(s)
    def bold(s):   return str(s)
    def dim(s):    return str(s)


def _rule(char="═", width=60):
    print(char * width)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path):
    """Load and validate session.yaml, returning the full config dict."""
    path = Path(config_path)
    if not path.exists():
        print(red(f"  ✗ Config not found: {path.resolve()}"))
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    if not config:
        print(red("  ✗ Config file is empty."))
        sys.exit(1)

    return config


def resolve_paths(config, config_path):
    """
    Resolve all paths from config, using config file location as base
    for relative paths. Returns a dict of resolved path strings.
    """
    paths_cfg  = config.get("paths", {})
    config_dir = Path(config_path).parent.parent  # config/ -> project root

    music_dir  = Path(paths_cfg.get("music_dir", "~/Music")).expanduser()
    data_dir   = Path(paths_cfg.get("data_dir", "./data"))
    output_dir = Path(paths_cfg.get("output_dir", "./output"))

    # Make relative paths relative to project root
    if not data_dir.is_absolute():
        data_dir = config_dir / data_dir
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir

    return {
        "music_dir":           str(music_dir),
        "data_dir":            str(data_dir),
        "output_dir":          str(output_dir),
        "tracks_json":         str(data_dir / "tracks.json"),
        "analysis_json":       str(data_dir / "analysis.json"),
        "matched_tracks_json": str(data_dir / "matched_tracks.json"),
        "session_json":        str(data_dir / "session.json"),
        "analysis_cache":      str(data_dir / ".analysis_cache"),
        "stems_cache":         str(data_dir / ".stems_cache"),
    }


# ── Module runners ────────────────────────────────────────────────────────────

def run_module_1(paths, config):
    """Module 1: Ingestor — scan library."""
    from modules.ingestor import run as ingestor_run
    return ingestor_run(
        music_dir=paths["music_dir"],
        output_path=paths["tracks_json"],
    )


def run_module_2(paths, config):
    """Module 2: Analyzer — audio analysis."""
    from modules.analyzer import run as analyzer_run
    return analyzer_run(
        tracks_path=paths["tracks_json"],
        output_path=paths["analysis_json"],
        cache_dir=paths["analysis_cache"],
        n_jobs=4,
    )


def run_module_3(paths, config):
    """Module 3: Spotify Bridge — playlist matching (optional)."""
    spotify_cfg = config.get("spotify", {})
    enabled     = spotify_cfg.get("enabled", False)

    if not enabled:
        print(f"\n  {dim('Spotify integration disabled — skipping Module 3.')}")
        print(f"  {dim('Planner will build setlist from full library.')}\n")
        return None

    playlist_ids = spotify_cfg.get("playlist_ids", [])
    threshold    = spotify_cfg.get("match_threshold", 80)

    if not playlist_ids:
        print(yellow("  ⚠  spotify.enabled is true but no playlist_ids configured — skipping."))
        return None

    from modules.spotify_bridge import run as spotify_run

    # Run for the first playlist (primary seed source)
    # Future: could merge multiple playlists
    playlist_id = playlist_ids[0]
    return spotify_run(
        playlist_id=playlist_id,
        tracks_path=paths["tracks_json"],
        output_path=paths["matched_tracks_json"],
        threshold=threshold,
    )


def run_module_4(paths, config):
    """Module 4: Planner — build setlist."""
    from modules.planner import run as planner_run
    return planner_run(
        analysis_path=paths["analysis_json"],
        matched_tracks_path=paths["matched_tracks_json"],
        config_path=args_config_path,  # pass the original config path
        output_path=paths["session_json"],
    )


def run_module_5(paths, config, preview_mode=False, quality_mode=False, output_file=None):
    """Module 5: Renderer — render the mix."""
    from modules.renderer import run as renderer_run

    # Build output path from config if not specified
    if output_file is None:
        output_cfg = config.get("output", {})
        fmt        = output_cfg.get("format", "mp3")
        suffix     = "_preview" if preview_mode else ""
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = str(Path(paths["output_dir"]) / f"mix_{timestamp}{suffix}.{fmt}")

    return renderer_run(
        session_path=paths["session_json"],
        output_path=output_file,
        preview_mode=preview_mode,
        quality_mode=quality_mode,
        stems_cache_dir=paths["stems_cache"],
    )


def run_creative(paths, config, preview_mode=False, quality_mode=False, output_file=None):
    """Creative mode: replaces Modules 4+5 when session.mode == 'creative'."""
    from modules.creative import run as creative_run

    if output_file is None:
        output_cfg = config.get("output", {})
        fmt        = output_cfg.get("format", "mp3")
        suffix     = "_preview" if preview_mode else "_creative"
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = str(Path(paths["output_dir"]) / f"mix_{timestamp}{suffix}.{fmt}")

    return creative_run(
        analysis_path=paths["analysis_json"],
        config_path=args_config_path,
        stems_cache_dir=paths["stems_cache"],
        output_path=output_file,
        preview_mode=preview_mode,
        quality_mode=quality_mode,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

# Module registry — label, runner function, requires previous output?
MODULES = [
    (1, "Ingestor",       run_module_1),
    (2, "Analyzer",       run_module_2),
    (3, "Spotify Bridge", run_module_3),
    (4, "Planner",        run_module_4),
    (5, "Renderer",       run_module_5),
]

# Global reference to config path (needed by planner which reads yaml directly)
args_config_path = None


def run_pipeline(config_path, from_module=1, preview_mode=False,
                 quality_mode=False, output_file=None):
    """
    Run the full pipeline from `from_module` through Module 5.
    """
    global args_config_path
    args_config_path = config_path

    pipeline_start = time.time()

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print(bold("  🎛️  dj1 — Autonomous DJ Engine"))
    _rule("═")
    print(f"  {dim('Config:')}      {config_path}")
    print(f"  {dim('Start from:')}  Module {from_module}")
    print(f"  {dim('Mode:')}        {'PREVIEW' if preview_mode else 'QUALITY' if quality_mode else 'DEFAULT'}")
    print(f"  {dim('Time:')}        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _rule("─")

    # ── Load config ───────────────────────────────────────────────────────────
    config = load_config(config_path)
    paths  = resolve_paths(config, config_path)

    # Ensure directories exist
    Path(paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Run modules ───────────────────────────────────────────────────────────
    results = {}

    session_mode = config.get("session", {}).get("mode", "sequential")

    for num, label, runner in MODULES:
        if num < from_module:
            print(f"\n  {dim(f'Module {num}: {label} — skipped (--from-module {from_module})')}")
            continue

        print(f"\n{'─' * 60}")
        print(bold(f"  ▶ Module {num}: {label}"))
        print(f"{'─' * 60}")

        module_start = time.time()
        try:
            if num == 4 and session_mode == "creative":
                # Creative mode replaces Modules 4+5 in one call
                label = "Creative Mode (4+5)"
                result = run_creative(paths, config, preview_mode, quality_mode, output_file)
                module_elapsed = time.time() - module_start
                results[5] = {"status": "ok", "elapsed": module_elapsed, "result": result}
                print(f"\n  {green('✓')} Creative mode completed in {module_elapsed:.1f}s")
                break  # skip Module 5
            elif num == 5:
                result = runner(paths, config, preview_mode, quality_mode, output_file)
            else:
                result = runner(paths, config)

            module_elapsed = time.time() - module_start
            results[num] = {"status": "ok", "elapsed": module_elapsed, "result": result}
            print(f"\n  {green('✓')} Module {num} completed in {module_elapsed:.1f}s")

        except SystemExit as e:
            module_elapsed = time.time() - module_start
            results[num] = {"status": "error", "elapsed": module_elapsed, "error": str(e)}
            print(f"\n  {red('✗')} Module {num} failed: {e}")
            # Module 3 (Spotify) failure is non-fatal — continue without seeds
            if num == 3:
                print(f"  {yellow('→ Continuing without Spotify seeds.')}")
                continue
            else:
                break

        except Exception as e:
            module_elapsed = time.time() - module_start
            results[num] = {"status": "error", "elapsed": module_elapsed, "error": str(e)}
            print(f"\n  {red('✗')} Module {num} failed with exception: {e}")
            if num == 3:
                print(f"  {yellow('→ Continuing without Spotify seeds.')}")
                continue
            else:
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    pipeline_elapsed = time.time() - pipeline_start

    print(f"\n{'═' * 60}")
    print(bold("  📊 Pipeline Summary"))
    print(f"{'═' * 60}")

    for num, label, _ in MODULES:
        if num in results:
            r = results[num]
            status_icon = green("✓") if r["status"] == "ok" else red("✗")
            print(f"  {status_icon} Module {num}: {label:<20} {r['elapsed']:>6.1f}s")
        elif num < from_module:
            print(f"  {dim('—')} Module {num}: {label:<20} {dim('skipped')}")

    print(f"{'─' * 60}")
    print(f"  {bold('Total time:')}  {pipeline_elapsed:.1f}s")

    # Track count from analysis
    if 2 in results and results[2]["status"] == "ok":
        analysis = results[2].get("result", {})
        stats = analysis.get("stats", {})
        if stats:
            print(f"  {bold('Tracks analyzed:')}  {stats.get('successful', '?')}/{stats.get('total_tracks', '?')}")

    # Setlist info from planner
    if 4 in results and results[4]["status"] == "ok":
        planner_out = results[4].get("result", {})
        if isinstance(planner_out, dict):
            n_tracks = planner_out.get("total_tracks", "?")
            curve    = planner_out.get("curve_type", "?")
            print(f"  {bold('Setlist:')}  {n_tracks} tracks, curve: {curve}")

    # Output file from renderer
    if 5 in results and results[5]["status"] == "ok":
        output_path = results[5].get("result")
        if output_path:
            print(f"  {bold('Output:')}  {output_path}")

    _rule("═")
    print()

    # Return success status
    all_ok = all(r["status"] == "ok" for r in results.values()
                 if results.get(num, {}).get("status") != "skipped")
    return all_ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="dj1 — Autonomous DJ Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py --config config/session.yaml
  uv run python main.py --config config/session.yaml --preview
  uv run python main.py --config config/session.yaml --quality
  uv run python main.py --config config/session.yaml --from-module 4
  uv run python main.py --config config/session.yaml --from-module 5 --preview
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="./config/session.yaml",
        help="Path to session.yaml config file (default: ./config/session.yaml)",
    )
    parser.add_argument(
        "--from-module", "-f",
        type=int, choices=[1, 2, 3, 4, 5], default=1,
        help="Start pipeline from this module (1–5). Earlier outputs must exist.",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Fast render: no time-stretching, shorter fades, 128k MP3",
    )
    parser.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Full render pipeline: beat-align + stretch + EQ, 320k MP3 or WAV",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Override output file path for the rendered mix",
    )

    args = parser.parse_args()

    if args.preview and args.quality:
        parser.error("--preview and --quality are mutually exclusive")

    success = run_pipeline(
        config_path=args.config,
        from_module=args.from_module,
        preview_mode=args.preview,
        quality_mode=args.quality,
        output_file=args.output,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()