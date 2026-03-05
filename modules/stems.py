"""
Module 5b: Stem Separator
─────────────────────────
Wraps Facebook's demucs (htdemucs model) to separate tracks into 4 stems:
  drums | bass | other (melody/synths) | vocals

Stems are cached per track_id under data/.stems_cache/{track_id}/.
The renderer uses these stems for stem_blend transitions instead of
EQ filter approximations.

Requirements:
  demucs must be installed and available in PATH.
  Install via: pip install demucs  (use Python 3.12 or earlier — demucs
  has no wheels for Python 3.14 as of 2026-03).

  Or install via homebrew / conda and ensure `demucs` is in your shell PATH.

Usage:
  Called by renderer.py when any slot has transition_hint == "stem_blend".
  Never called directly.
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

log = logging.getLogger(__name__)

STEM_NAMES  = ("drums", "bass", "other", "vocals")
DEMUCS_MODEL = "htdemucs"

# ── Availability check ────────────────────────────────────────────────────────

def demucs_available() -> bool:
    """Return True if `demucs` command is found in PATH."""
    return shutil.which("demucs") is not None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _stem_dir(cache_dir: str, track_id: str) -> Path:
    return Path(cache_dir) / track_id


def _stems_cached(cache_dir: str, track_id: str) -> bool:
    d = _stem_dir(cache_dir, track_id)
    return d.exists() and all((d / f"{name}.wav").exists() for name in STEM_NAMES)


def _stem_paths(cache_dir: str, track_id: str) -> dict[str, str]:
    d = _stem_dir(cache_dir, track_id)
    return {name: str(d / f"{name}.wav") for name in STEM_NAMES}


# ── Core separation ───────────────────────────────────────────────────────────

def separate_track(
    filepath: str,
    cache_dir: str,
    track_id: str,
) -> Optional[dict[str, str]]:
    """
    Separate one track into 4 stems using demucs (htdemucs model).
    Results are cached in cache_dir/{track_id}/.

    Returns:
        {drums: path, bass: path, other: path, vocals: path} on success
        None if demucs is unavailable or separation fails
    """
    if not demucs_available():
        log.warning(f"stems: demucs not found in PATH — skipping {Path(filepath).name}")
        return None

    # Return cached results if present
    if _stems_cached(cache_dir, track_id):
        log.debug(f"stems: cache hit for {track_id}")
        return _stem_paths(cache_dir, track_id)

    stem_dir = _stem_dir(cache_dir, track_id)
    stem_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Stems: separating {Path(filepath).name} (may take ~1 min)...")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "demucs",
            "--name", DEMUCS_MODEL,
            "--out", tmp,
            filepath,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,   # 10 min hard limit
            )
        except subprocess.TimeoutExpired:
            log.error(f"stems: demucs timed out for {filepath}")
            return None
        except Exception as e:
            log.error(f"stems: demucs subprocess error: {e}")
            return None

        if result.returncode != 0:
            log.error(f"stems: demucs failed (exit {result.returncode}): {result.stderr[-500:]}")
            return None

        # Demucs writes to: tmp/htdemucs/{track_stem_name}/{drum,bass,...}.wav
        # Find the output dir (demucs uses the audio filename without extension)
        track_stem_name = Path(filepath).stem
        demucs_out = Path(tmp) / DEMUCS_MODEL / track_stem_name
        if not demucs_out.exists():
            log.error(f"stems: expected demucs output dir not found: {demucs_out}")
            return None

        # Copy stems to cache
        for name in STEM_NAMES:
            src = demucs_out / f"{name}.wav"
            dst = stem_dir / f"{name}.wav"
            if not src.exists():
                log.error(f"stems: missing stem file {src}")
                return None
            shutil.copy2(str(src), str(dst))

    elapsed = time.time() - t0
    print(f"  Stems: done ({elapsed:.0f}s) → {stem_dir}")
    return _stem_paths(cache_dir, track_id)


# ── Bulk processing ───────────────────────────────────────────────────────────

def run_stems_for_setlist(
    setlist: list[dict],
    cache_dir: str,
) -> dict[str, dict[str, str]]:
    """
    Separate stems for all tracks involved in stem_blend transitions.
    Processes both the incoming track AND its predecessor (outgoing track).

    Returns:
        {track_id: {drums: path, bass: path, other: path, vocals: path}}
        Empty dict if demucs is unavailable.
    """
    if not demucs_available():
        print("  Stems: demucs not found — stem_blend transitions will degrade to long_blend")
        return {}

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Collect track_ids needed for stem separation
    needed: set[str] = set()
    for i, slot in enumerate(setlist):
        if slot.get("transition_hint") == "stem_blend":
            needed.add(slot["track_id"])           # incoming
            if i > 0:
                needed.add(setlist[i - 1]["track_id"])  # outgoing

    if not needed:
        return {}

    print(f"\n  Stems: separating {len(needed)} track(s) for stem_blend transitions...")

    results: dict[str, dict[str, str]] = {}
    for slot in setlist:
        tid = slot["track_id"]
        if tid not in needed:
            continue
        if tid in results:
            continue
        stem_paths = separate_track(
            filepath=slot["filepath"],
            cache_dir=cache_dir,
            track_id=tid,
        )
        if stem_paths:
            results[tid] = stem_paths

    return results


def run_stems_for_creative(
    tracks_used: list[str],
    analyses: list[dict],
    cache_dir: str,
) -> dict[str, dict[str, str]]:
    """
    Separate stems for all tracks referenced in a creative mode plan.

    Args:
        tracks_used: list of track_ids from creative_plan["tracks_used"]
        analyses: full analyses list (needed for filepath lookup)
        cache_dir: stems cache directory

    Returns:
        {track_id: {drums: path, bass: path, other: path, vocals: path}}
        Empty dict if demucs is unavailable.
    """
    if not demucs_available():
        print("  Stems: demucs not found — creative mode requires demucs")
        return {}

    if not tracks_used:
        return {}

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Build filepath lookup
    filepath_by_id = {a["track_id"]: a.get("filepath", "") for a in analyses}

    print(f"\n  Stems: separating {len(tracks_used)} track(s) for creative mode...")

    results: dict[str, dict[str, str]] = {}
    for track_id in tracks_used:
        filepath = filepath_by_id.get(track_id, "")
        if not filepath or not Path(filepath).exists():
            log.warning(f"stems: filepath not found for {track_id} — skipping")
            continue
        stem_paths = separate_track(
            filepath=filepath,
            cache_dir=cache_dir,
            track_id=track_id,
        )
        if stem_paths:
            results[track_id] = stem_paths

    return results


# ── Audio loaders ─────────────────────────────────────────────────────────────

def load_stems(
    stem_paths: dict[str, str],
    frame_rate: int = 44100,
    channels: int = 2,
) -> Optional[dict[str, AudioSegment]]:
    """
    Load stem WAV files as pydub AudioSegments, normalised to frame_rate/channels.
    Returns None if any stem file is missing or fails to load.
    """
    segs: dict[str, AudioSegment] = {}
    for name, path in stem_paths.items():
        try:
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(frame_rate).set_channels(channels)
            segs[name] = seg
        except Exception as e:
            log.error(f"stems: failed to load {name} from {path}: {e}")
            return None
    return segs
