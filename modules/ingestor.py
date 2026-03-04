"""
Module 1: Library Ingestor
──────────────────────────
Scans a local music folder recursively, extracts metadata from audio files,
and writes a normalized track registry (tracks.json) for downstream modules.

Supported formats: MP3, FLAC, WAV, AAC, M4A, OGG

Dependencies: stdlib only + (optional) mutagen for richer tag extraction.
If mutagen is not installed, WAV files are parsed natively via the stdlib
`wave` module, and other formats fall back to filename-only metadata.
Install mutagen for full tag support: pip install mutagen
"""

import os
import sys
import json
import wave
import struct
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# ── Optional mutagen import ───────────────────────────────────────────────────
try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# ── Colorama for pretty output (available in environment) ─────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def green(s):  return Fore.GREEN + s + Style.RESET_ALL
    def cyan(s):   return Fore.CYAN  + s + Style.RESET_ALL
    def yellow(s): return Fore.YELLOW + s + Style.RESET_ALL
    def red(s):    return Fore.RED   + s + Style.RESET_ALL
    def bold(s):   return Style.BRIGHT + s + Style.RESET_ALL
    def dim(s):    return Style.DIM  + s + Style.RESET_ALL
except ImportError:
    def green(s):  return s
    def cyan(s):   return s
    def yellow(s): return s
    def red(s):    return s
    def bold(s):   return s
    def dim(s):    return s

# ── Supported formats ─────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_track_id(filepath: str) -> str:
    """Stable unique ID based on absolute filepath."""
    return hashlib.md5(filepath.encode("utf-8")).hexdigest()[:12]


def format_duration(seconds) -> str:
    """Convert seconds to mm:ss string."""
    if not seconds:
        return "0:00"
    seconds = float(seconds)
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def clean_tag(value) -> str | None:
    """
    Extract a clean string from a mutagen tag value.
    Mutagen returns tags as lists or special objects depending on format.
    """
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value).strip() or None


def _parse_wav_stdlib(filepath: Path) -> dict:
    """Parse a WAV file using stdlib wave module. Returns partial metadata dict."""
    result = {}
    try:
        with wave.open(str(filepath), "rb") as wf:
            frames      = wf.getnframes()
            rate        = wf.getframerate()
            channels    = wf.getnchannels()
            sampwidth   = wf.getsampwidth()
            duration    = frames / float(rate) if rate > 0 else None
            result["duration_seconds"] = round(duration, 2) if duration else None
            result["sample_rate_hz"]   = rate
            result["channels"]         = channels
            result["bitrate_kbps"]     = round((rate * channels * sampwidth * 8) / 1000, 1)
    except Exception as e:
        result["parse_error"] = f"WAV stdlib parse failed: {e}"
    return result


def extract_metadata(filepath: Path) -> dict:
    """
    Extract normalized metadata from an audio file.
    Uses mutagen when available; falls back to stdlib for WAV,
    and gracefully handles errors for all other formats.
    Returns a dict regardless — missing fields are None.
    """
    ext = filepath.suffix.lower()
    meta = {
        "track_id":          make_track_id(str(filepath.resolve())),
        "filepath":          str(filepath.resolve()),
        "filename":          filepath.name,
        "extension":         ext,
        "title":             None,
        "artist":            None,
        "album":             None,
        "genre":             None,
        "year":              None,
        "track_number":      None,
        "duration_seconds":  None,
        "duration_formatted": None,
        "bitrate_kbps":      None,
        "sample_rate_hz":    None,
        "channels":          None,
        "file_size_mb":      None,
        "has_missing_tags":  False,
        "parse_error":       None,
    }

    # File size (always attempt, doesn't require the file to be valid audio)
    try:
        meta["file_size_mb"] = round(filepath.stat().st_size / (1024 * 1024), 2)
    except Exception:
        pass

    # ── Mutagen path (richer, multi-format) ───────────────────────────────────
    if MUTAGEN_AVAILABLE:
        try:
            audio = MutagenFile(str(filepath), easy=True)

            if audio is None:
                meta["parse_error"] = "mutagen could not parse file (unsupported or corrupt)"
            else:
                if hasattr(audio, "info"):
                    info = audio.info
                    meta["duration_seconds"]  = round(info.length, 2) if getattr(info, "length", None) else None
                    meta["duration_formatted"] = format_duration(meta["duration_seconds"])
                    meta["bitrate_kbps"]       = round(info.bitrate / 1000, 1) if getattr(info, "bitrate", None) else None
                    meta["sample_rate_hz"]     = getattr(info, "sample_rate", None)
                    meta["channels"]           = getattr(info, "channels", None)

                meta["title"]        = clean_tag(audio.get("title"))
                meta["artist"]       = clean_tag(audio.get("artist"))
                meta["album"]        = clean_tag(audio.get("album"))
                meta["genre"]        = clean_tag(audio.get("genre"))
                meta["track_number"] = clean_tag(audio.get("tracknumber"))

                year_raw = (audio.get("date") or audio.get("year") or audio.get("originaldate"))
                if year_raw:
                    year_str = clean_tag(year_raw)
                    if year_str:
                        meta["year"] = year_str[:4] if len(year_str) >= 4 else year_str

        except Exception as e:
            meta["parse_error"] = str(e)

    # ── Stdlib WAV fallback ────────────────────────────────────────────────────
    elif ext == ".wav":
        wav_data = _parse_wav_stdlib(filepath)
        meta.update(wav_data)
        if meta["duration_seconds"]:
            meta["duration_formatted"] = format_duration(meta["duration_seconds"])

    else:
        meta["parse_error"] = "mutagen not installed; only WAV files can be parsed without it"

    # Flag tracks missing core metadata
    core_tags = ["title", "artist", "duration_seconds"]
    meta["has_missing_tags"] = any(meta[t] is None for t in core_tags)

    return meta


def scan_library(music_dir: str | Path) -> tuple[list[dict], dict]:
    """
    Recursively scan music_dir for supported audio files.
    Returns (tracks list, summary stats dict).
    """
    music_path = Path(music_dir).expanduser().resolve()

    if not music_path.exists():
        raise FileNotFoundError(f"Music directory not found: {music_path}")
    if not music_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {music_path}")

    # Discover all supported files
    found_files = []
    for ext in SUPPORTED_EXTENSIONS:
        found_files.extend(music_path.rglob(f"*{ext}"))
        found_files.extend(music_path.rglob(f"*{ext.upper()}"))

    # Deduplicate (case-insensitive glob can double-hit on some systems)
    found_files = list({str(f): f for f in found_files}.values())
    found_files.sort(key=lambda f: str(f).lower())

    if not found_files:
        print(yellow(f"⚠  No supported audio files found in {music_path}"))
        return [], {}

    print(f"\n{bold(green(f'🎵 Found {len(found_files)} audio files — scanning...'))}\n")

    tracks = []
    errors = []

    total = len(found_files)
    _BAR_WIDTH = 40
    _LINE_WIDTH = 80  # pad to fully overwrite previous line on \r
    for i, f in enumerate(found_files):
        pct = int((i + 1) / total * _BAR_WIDTH)
        bar = "█" * pct + "░" * (_BAR_WIDTH - pct)
        label = f"  [{bar}] {i+1}/{total}  {f.name[:30]}"
        print(f"\r{label:<{_LINE_WIDTH}}", end="", flush=True)
        meta = extract_metadata(f)
        tracks.append(meta)
        if meta["parse_error"]:
            errors.append((f.name, meta["parse_error"]))
    print()  # newline after progress bar

    # ── Summary stats ─────────────────────────────────────────────────────────
    valid_durations = [t["duration_seconds"] for t in tracks if t["duration_seconds"]]
    total_seconds = sum(valid_durations)
    total_hours = total_seconds / 3600

    by_ext = {}
    for t in tracks:
        by_ext[t["extension"]] = by_ext.get(t["extension"], 0) + 1

    missing_tags = [t for t in tracks if t["has_missing_tags"]]

    stats = {
        "scanned_at": datetime.now().isoformat(),
        "music_dir": str(music_path),
        "total_tracks": len(tracks),
        "total_duration_seconds": round(total_seconds, 2),
        "total_duration_formatted": format_duration(total_seconds),
        "total_hours": round(total_hours, 2),
        "tracks_with_missing_tags": len(missing_tags),
        "tracks_with_errors": len(errors),
        "formats": by_ext,
        "errors": errors,
    }

    return tracks, stats


def _rule(char="─", width=60):
    print(char * width)


def print_summary(stats: dict, tracks: list):
    """Print a formatted summary table to the console."""
    print()
    _rule("═")
    print(bold("  📊 Library Scan Summary"))
    _rule("═")
    rows = [
        ("Total Tracks",    str(stats["total_tracks"])),
        ("Total Duration",  f"{stats['total_duration_formatted']} ({stats['total_hours']}h)"),
        ("Formats Found",   ", ".join(f"{ext}({n})" for ext, n in stats["formats"].items())),
        ("Missing Tags",    str(stats["tracks_with_missing_tags"])),
        ("Parse Errors",    str(stats["tracks_with_errors"])),
    ]
    for label, value in rows:
        print(f"  {cyan(label + ':'):<28} {value}")
    _rule()

    if tracks:
        print()
        _rule("─")
        print(bold("  🎵 Track Sample (first 10)"))
        _rule("─")
        header = f"  {'ID':<14} {'Title':<32} {'Artist':<24} {'Duration'}"
        print(dim(header))
        _rule("─")
        for t in tracks[:10]:
            title    = (t["title"] or t["filename"])[:30]
            artist   = (t["artist"] or "Unknown")[:22]
            duration = t["duration_formatted"] or "?"
            print(f"  {dim(t['track_id']):<14} {title:<32} {cyan(artist):<24} {green(duration)}")
        _rule("─")

    if stats["errors"]:
        print()
        print(yellow("⚠  Parse Errors:"))
        for filename, err in stats["errors"]:
            print(f"  {red('•')} {filename}: {err}")


def run(music_dir: str, output_path: str = "./data/tracks.json") -> dict:
    """
    Main entry point for Module 1.
    Returns the full output dict (tracks + stats).
    """
    _rule("═")
    print(bold("  🎛️  AutoDJ — Module 1: Library Ingestor"))
    _rule("═")

    tracks, stats = scan_library(music_dir)

    if not tracks:
        return {"tracks": [], "stats": stats}

    print_summary(stats, tracks)

    output = {"tracks": tracks, "stats": stats}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(green(f"  ✅ tracks.json written → {out_path.resolve()}"))
    _rule("═")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDJ Module 1 — Library Ingestor")
    parser.add_argument(
        "--music-dir", "-m",
        required=True,
        help="Path to your local music folder (scanned recursively)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/tracks.json",
        help="Output path for tracks.json (default: ./data/tracks.json)"
    )
    args = parser.parse_args()

    run(music_dir=args.music_dir, output_path=args.output)