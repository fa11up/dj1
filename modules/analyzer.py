"""
Module 2: Audio Analyzer
────────────────────────
Performs deep audio analysis on each track from tracks.json and writes
analysis.json — the feature data that drives all downstream DJ logic.

Extracted per track:
  - BPM (tempo) + confidence score
  - Beat grid (timestamps of every beat)
  - Musical key + mode (Camelot Wheel position)
  - Energy profile (RMS over time + mean/peak/dynamic range)
  - Intro / outro zones (where to cue in/out)
  - Spectral centroid (brightness proxy)
  - Danceability estimate (beat regularity + energy stability)

Results are cached to disk — re-running only processes new/changed files.
Parallel processing via joblib for large libraries.

Usage:
    uv run python -m modules.analyzer                        # uses data/tracks.json
    uv run python -m modules.analyzer --tracks data/tracks.json --jobs 4
"""

import os
import sys
import json
import time
import hashlib
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa

# Suppress librosa/numba warnings that clutter output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning)

from joblib import Parallel, delayed

# ── Colorama ──────────────────────────────────────────────────────────────────
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


# ── Constants ─────────────────────────────────────────────────────────────────

# Camelot Wheel: maps (chroma_index, is_major) → Camelot position string
# chroma_index: 0=C, 1=C#/Db, 2=D, 3=D#/Eb, 4=E, 5=F,
#               6=F#/Gb, 7=G, 8=G#/Ab, 9=A, 10=A#/Bb, 11=B
CAMELOT_MAJOR = {
    0: "8B",  1: "3B",  2: "10B", 3: "5B",  4: "12B", 5: "7B",
    6: "2B",  7: "9B",  8: "4B",  9: "11B", 10: "6B", 11: "1B",
}
CAMELOT_MINOR = {
    0: "5A",  1: "12A", 2: "7A",  3: "2A",  4: "9A",  5: "4A",
    6: "11A", 7: "6A",  8: "1A",  9: "8A",  10: "3A", 11: "10A",
}

# Key names for human-readable output
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Intro/outro detection — look this far into track for cue points
INTRO_SEARCH_SEC  = 60   # scan first 60s for intro end
OUTRO_SEARCH_SEC  = 60   # scan last 60s for outro start

# Energy profile: number of time segments to sample across the track
ENERGY_SEGMENTS = 32
SECTION_MIN_DURATION_SEC = 4.0     # discard sections shorter than this
SECTION_ENERGY_THRESHOLD = 0.25    # below this normalised energy = "low"
SECTION_HIGH_THRESHOLD   = 0.70    # above this = "high"

# Cache version — bump this to invalidate all cached analyses
CACHE_VERSION = "3.0"

# BPM normalisation target range — covers House (120–130), Techno (125–140),
# DnB (160–175), Jungle, and most electronic music at native tempo.
BPM_NORM_LO = 80.0
BPM_NORM_HI = 175.0

def normalise_bpm(bpm, lo=BPM_NORM_LO, hi=BPM_NORM_HI):
    """
    Correct half-time / double-time BPM detection errors.

    librosa sometimes reports BPM at half or double the true tempo:
      - 156 BPM DnB track detected as 78  → should double to 156
      - 128 BPM House track detected as 256 → should halve to 128

    Strategy: if BPM falls outside the target range (80–175), repeatedly
    halve or double until it lands inside.

    Returns (normalised_bpm, was_corrected: bool).
    """
    if bpm is None or bpm <= 0:
        return bpm, False
    original = bpm
    # Double up if too slow
    attempts = 0
    while bpm < lo and attempts < 4:
        bpm *= 2
        attempts += 1
    # Halve if too fast
    attempts = 0
    while bpm > hi and attempts < 4:
        bpm /= 2
        attempts += 1
    # Final sanity — if still outside range, return original unchanged
    if not (lo <= bpm <= hi):
        return original, False
    corrected = abs(bpm - original) > 0.5
    return round(bpm, 2), corrected

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rule(char="─", width=60):
    print(char * width)


def file_hash(filepath: str) -> str:
    """
    Fast file fingerprint using size + mtime.
    Good enough for cache invalidation — avoids reading the full file.
    """
    stat = Path(filepath).stat()
    raw = f"{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def camelot(key_index: int, is_major: bool) -> str:
    """Convert librosa key index + mode to Camelot Wheel position."""
    if is_major:
        return CAMELOT_MAJOR.get(key_index, "?")
    return CAMELOT_MINOR.get(key_index, "?")


def detect_intro_outro(y, sr, duration, sections=None):
    """
    Estimate where the musical content starts (intro_end) and where it begins
    to wind down (outro_start).

    If structural sections are available (P2-6), uses them for more precise
    detection. Falls back to the original RMS energy thresholding otherwise.

    Returns (intro_end_sec, outro_start_sec).
    """
    intro_end   = 0.0
    outro_start = duration

    # ── Try sections-based detection first ────────────────────────────────────
    if sections:
        # Find the end of the last "intro" section
        intro_sections = [s for s in sections if s["label"] == "intro"]
        if intro_sections:
            intro_end = intro_sections[-1]["end_sec"]

        # Find the start of the first "outro" section
        outro_sections = [s for s in sections if s["label"] == "outro"]
        if outro_sections:
            outro_start = outro_sections[0]["start_sec"]

        # If we got useful values from sections, use them
        if intro_end > 0 or outro_start < duration:
            intro_end   = max(0.0, min(intro_end, duration * 0.4))
            outro_start = max(duration * 0.5, min(outro_start, duration))
            if intro_end < outro_start:
                return round(intro_end, 2), round(outro_start, 2)

    # ── Fallback: original RMS energy thresholding ────────────────────────────
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    threshold = np.max(rms) * 0.15

    # Intro
    intro_end = 0.0
    search_end = min(INTRO_SEARCH_SEC, duration * 0.4)
    intro_frames = times <= search_end
    if intro_frames.any():
        intro_rms = rms[intro_frames]
        intro_times = times[intro_frames]
        above = np.where(intro_rms > threshold)[0]
        if len(above) > 0:
            intro_end = float(intro_times[above[0]])

    # Outro
    outro_start = duration
    search_start = max(0, duration - OUTRO_SEARCH_SEC)
    outro_frames = times >= search_start
    if outro_frames.any():
        outro_rms = rms[outro_frames]
        outro_times = times[outro_frames]
        above = np.where(outro_rms > threshold)[0]
        if len(above) > 0:
            outro_start = float(outro_times[above[-1]])

    intro_end   = max(0.0, min(intro_end, duration * 0.4))
    outro_start = max(duration * 0.5, min(outro_start, duration))

    return round(intro_end, 2), round(outro_start, 2)


def detect_sections(y, sr, beat_times, energy_curve, duration):
    """
    Detect structural sections of a track (intro, verse, chorus, breakdown, outro)
    using a combination of spectral clustering and energy analysis.

    Uses librosa's spectral features to find segment boundaries via self-similarity,
    then labels each segment based on its energy characteristics relative to the
    track's overall energy profile.

    Labelling heuristic:
      - First section with low energy → "intro"
      - Last section with low energy  → "outro"
      - Sections with high energy     → "chorus" (or "drop" if very high)
      - Sections with medium energy   → "verse"
      - Sections with energy dip surrounded by high sections → "breakdown"

    Args:
        y:            audio signal (mono, float)
        sr:           sample rate
        beat_times:   list of beat timestamps in seconds
        energy_curve: normalised energy curve (32 segments, 0–1)
        duration:     total track duration in seconds

    Returns:
        list of section dicts: [{label, start_sec, end_sec, energy, energy_category}]
        Returns empty list on failure.
    """
    try:
        # ── Compute features for segmentation ─────────────────────────────────
        # Use MFCCs — they capture timbral changes well for structural boundaries
        hop_length = 512
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

        # Self-similarity via recurrence matrix
        # Stack delta-MFCCs for richer features
        mfcc_delta = librosa.feature.delta(mfcc)
        features = np.vstack([mfcc, mfcc_delta])

        # Detect segment boundaries using spectral clustering on the
        # self-similarity matrix (novelty-based segmentation)
        bounds = librosa.segment.agglomerative(features, k=None)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)

        # Add track start and end
        bound_times = np.concatenate([[0.0], bound_times, [duration]])
        bound_times = np.unique(np.sort(bound_times))

        # ── Build raw sections ────────────────────────────────────────────────
        raw_sections = []
        for i in range(len(bound_times) - 1):
            start = float(bound_times[i])
            end   = float(bound_times[i + 1])
            if (end - start) < SECTION_MIN_DURATION_SEC:
                continue  # skip tiny sections

            # Compute energy for this section from the energy curve
            if energy_curve and len(energy_curve) > 0:
                n_segs = len(energy_curve)
                start_idx = int(start / duration * n_segs)
                end_idx   = int(end / duration * n_segs)
                start_idx = max(0, min(start_idx, n_segs - 1))
                end_idx   = max(start_idx + 1, min(end_idx, n_segs))
                section_energy = float(np.mean(energy_curve[start_idx:end_idx]))
            else:
                # Compute directly from audio
                start_sample = int(start * sr)
                end_sample   = min(int(end * sr), len(y))
                if end_sample > start_sample:
                    section_audio = y[start_sample:end_sample]
                    rms = librosa.feature.rms(y=section_audio, hop_length=hop_length)[0]
                    section_energy = float(np.mean(rms)) if len(rms) > 0 else 0.0
                    # Normalise roughly to 0–1
                    peak_rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
                    peak = float(np.max(peak_rms)) if len(peak_rms) > 0 else 1.0
                    section_energy = section_energy / peak if peak > 0 else 0.0
                else:
                    section_energy = 0.0

            raw_sections.append({
                "start_sec": round(start, 2),
                "end_sec":   round(end, 2),
                "energy":    round(section_energy, 4),
            })

        if not raw_sections:
            return []

        # ── Merge very short adjacent sections with similar energy ────────────
        merged = [raw_sections[0]]
        for sec in raw_sections[1:]:
            prev = merged[-1]
            energy_diff = abs(prev["energy"] - sec["energy"])
            combined_dur = sec["end_sec"] - prev["start_sec"]
            # Merge if very similar energy and combined duration is reasonable
            if energy_diff < 0.15 and combined_dur < duration * 0.4:
                merged[-1] = {
                    "start_sec": prev["start_sec"],
                    "end_sec":   sec["end_sec"],
                    "energy":    round((prev["energy"] + sec["energy"]) / 2, 4),
                }
            else:
                merged.append(sec)

        # ── Label sections ────────────────────────────────────────────────────
        sections = []
        n = len(merged)
        for i, sec in enumerate(merged):
            e = sec["energy"]

            # Categorise energy level
            if e < SECTION_ENERGY_THRESHOLD:
                energy_cat = "low"
            elif e > SECTION_HIGH_THRESHOLD:
                energy_cat = "high"
            else:
                energy_cat = "mid"

            # Positional + energy-based labelling
            is_first = (i == 0)
            is_last  = (i == n - 1)
            is_early = (i <= 1)
            is_late  = (i >= n - 2)

            # Check for breakdown: low energy surrounded by higher sections
            prev_energy = merged[i - 1]["energy"] if i > 0 else 0.0
            next_energy = merged[i + 1]["energy"] if i < n - 1 else 0.0
            is_dip = (e < prev_energy * 0.6 and e < next_energy * 0.6
                      and prev_energy > SECTION_ENERGY_THRESHOLD)

            if is_first and energy_cat == "low":
                label = "intro"
            elif is_last and energy_cat == "low":
                label = "outro"
            elif is_early and energy_cat == "low":
                label = "intro"
            elif is_late and energy_cat == "low":
                label = "outro"
            elif is_dip:
                label = "breakdown"
            elif energy_cat == "high":
                label = "chorus"
            elif energy_cat == "mid":
                label = "verse"
            else:
                label = "bridge"

            sections.append({
                "label":           label,
                "start_sec":       sec["start_sec"],
                "end_sec":         sec["end_sec"],
                "energy":          sec["energy"],
                "energy_category": energy_cat,
            })

        return sections

    except Exception:
        # Segmentation failed — return empty (non-fatal)
        return []


def energy_profile(y: np.ndarray, sr: int, n_segments: int = ENERGY_SEGMENTS) -> dict:
    """
    Compute a coarse energy profile across the track.
    Splits the track into n_segments and returns RMS energy per segment,
    normalized to 0–1. Also returns mean, peak, and dynamic range.
    """
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Downsample rms to n_segments buckets
    indices = np.linspace(0, len(rms) - 1, n_segments, dtype=int)
    curve = rms[indices].tolist()

    peak = float(np.max(rms)) if len(rms) > 0 else 0.0
    if peak > 0:
        curve_norm = [round(v / peak, 4) for v in curve]
    else:
        curve_norm = [0.0] * n_segments

    mean_energy = float(np.mean(rms))
    dynamic_range = float(np.max(rms) - np.min(rms)) if len(rms) > 0 else 0.0

    return {
        "curve":          curve_norm,       # normalized energy over time (0–1)
        "mean":           round(mean_energy, 6),
        "peak":           round(peak, 6),
        "dynamic_range":  round(dynamic_range, 6),
    }


def danceability_score(
    tempo: float,
    beat_times: list[float],
    energy_mean: float,
    energy_dynamic_range: float,
) -> float:
    """
    Estimate danceability (0–1) from beat regularity and energy stability.

    - Beat regularity: how consistent the inter-beat intervals are
      (low variance = steady groove = more danceable)
    - Energy stability: moderate dynamic range is good; extreme variance
      or total flatness both reduce danceability
    - Tempo sweet spot: 100–140 BPM scores highest

    This is a heuristic approximation of what Spotify's danceability measures.
    """
    if not beat_times or len(beat_times) < 4:
        return 0.5  # not enough data — return neutral score

    # Beat regularity: coefficient of variation of inter-beat intervals
    ibi = np.diff(beat_times)
    cv = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else 1.0
    regularity_score = max(0.0, 1.0 - cv * 3)  # CV of 0.33 → score of 0

    # Tempo sweet spot: 100–140 BPM = 1.0, falls off outside
    if 100 <= tempo <= 140:
        tempo_score = 1.0
    elif tempo < 100:
        tempo_score = max(0.3, tempo / 100)
    else:
        tempo_score = max(0.3, 1.0 - (tempo - 140) / 100)

    # Energy stability: penalize extremes (dead flat or wildly dynamic)
    # Normalize dynamic range to 0–1 range (rough calibration)
    dr_norm = min(energy_dynamic_range / 0.3, 1.0)
    energy_score = 1.0 - abs(dr_norm - 0.5) * 0.6

    score = (regularity_score * 0.5) + (tempo_score * 0.3) + (energy_score * 0.2)
    return round(min(1.0, max(0.0, score)), 3)


# ── Core analysis function ────────────────────────────────────────────────────

def analyze_track(track: dict, cache_dir: Path) -> dict:
    """
    Analyze a single track. Returns an analysis result dict.
    Checks cache first — only loads audio if cache is stale or missing.

    This function is designed to be called in parallel via joblib.
    It does NOT print anything (to avoid garbled parallel output) —
    status is collected and printed by the caller.
    """
    filepath = track["filepath"]
    track_id = track["track_id"]

    result = {
        "track_id":       track_id,
        "filepath":       filepath,
        "filename":       track["filename"],
        "analyzed_at":    None,
        "cache_hit":      False,
        "analysis_error": None,
        # Features
        "bpm":                 None,
        "bpm_raw":             None,
        "bpm_normalised":      None,
        "bpm_was_corrected":   None,
        "bpm_confidence":      None,
        "beat_times":          None,
        "beat_count":          None,
        "key_index":           None,
        "key_name":            None,
        "mode":                None,       # "major" | "minor"
        "camelot":             None,
        "energy":              None,       # full energy profile dict
        "intro_end_sec":       None,
        "outro_start_sec":      None,
        "spectral_centroid_mean": None,
        "danceability":        None,
        "sections": None
    }

    # ── File existence check (before cache key — file_hash calls stat()) ────────
    if not Path(filepath).exists():
        result["analysis_error"] = f"File not found: {filepath}"
        return result

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)  # ensure cache dir exists
    cache_key  = f"{track_id}_{file_hash(filepath)}_{CACHE_VERSION}"
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cache_hit"] = True
            return cached
        except Exception:
            pass  # corrupt cache — re-analyze

    # ── Load audio ────────────────────────────────────────────────────────────
    try:
        # mono=True: all analysis is mono; sr=None preserves native sample rate
        y, sr = librosa.load(filepath, mono=True, sr=None)
    except Exception as e:
        result["analysis_error"] = f"Load failed: {e}"
        return result

    duration = float(len(y) / sr)

    try:
        # ── Tempo + beat tracking ─────────────────────────────────────────────
        # onset_envelope is used by both tempo and beat tracker for consistency
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_arr, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            trim=False,
        )
        # librosa ≥0.10 returns tempo as a 1-element array
        bpm = float(tempo_arr[0]) if hasattr(tempo_arr, "__len__") else float(tempo_arr)
        bpm = round(bpm, 2)

        beat_times_arr = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = [round(float(t), 4) for t in beat_times_arr]

        # BPM confidence: based on tempo strength at the detected period
        # librosa's tempo confidence is the normalized peak of the tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempo_period_frames = librosa.time_to_frames(60.0 / bpm, sr=sr) if bpm > 0 else 1
        tempo_period_frames = min(tempo_period_frames, tempogram.shape[0] - 1)
        bpm_confidence = round(float(np.mean(tempogram[tempo_period_frames])), 4)

        # ── Key detection ─────────────────────────────────────────────────────
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_index = int(np.argmax(chroma_mean))
        key_name  = KEY_NAMES[key_index]

        # Mode detection: compare major vs minor template correlation
        # Major template: [1,0,1,0,1,1,0,1,0,1,0,1] (Ionian scale degrees)
        # Minor template: [1,0,1,1,0,1,0,1,1,0,1,0] (Aeolian scale degrees)
        major_template = np.array([1,0,1,0,1,1,0,1,0,1,0,1], dtype=float)
        minor_template = np.array([1,0,1,1,0,1,0,1,1,0,0,1], dtype=float)

        # Roll templates to match detected key
        maj = np.roll(major_template, key_index)
        min_ = np.roll(minor_template, key_index)

        # Correlation with normalized chroma
        chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-8)
        major_score = float(np.dot(chroma_norm, maj / maj.sum()))
        minor_score = float(np.dot(chroma_norm, min_ / min_.sum()))
        is_major    = major_score >= minor_score
        mode        = "major" if is_major else "minor"
        camelot_pos = camelot(key_index, is_major)

        # ── Energy profile ────────────────────────────────────────────────────
        eng = energy_profile(y, sr)

        # ── Spectral centroid ─────────────────────────────────────────────────
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = round(float(np.mean(sc)), 2)

        # ── Danceability ──────────────────────────────────────────────────────
        dance = danceability_score(bpm, beat_times, eng["mean"], eng["dynamic_range"])

        bpm_raw = bpm
        bpm, bpm_was_corrected = normalise_bpm(bpm)

        # ── Structural sections (P2-6) ──────────────────────────────────────
        sections = detect_sections(
            y, sr, beat_times,
            eng.get("curve") or eng.get("curve", []),
            duration,
        )

         # ── Intro / outro detection ───────────────────────────────────────────
        intro_end, outro_start = detect_intro_outro(y, sr, duration, sections)

        # ── Assemble result ───────────────────────────────────────────────────
        result.update({
            "analyzed_at":             datetime.now().isoformat(),
            "bpm":                     bpm,
            "bpm_raw":                 bpm_raw,
            "bpm_normalised":          bpm,
            "bpm_was_corrected":       bpm_was_corrected,
            "bpm_confidence":          bpm_confidence,
            "beat_times":              beat_times,
            "beat_count":              len(beat_times),
            "key_index":               key_index,
            "key_name":                key_name,
            "mode":                    mode,
            "camelot":                 camelot_pos,
            "energy":                  eng,
            "intro_end_sec":           intro_end,
            "outro_start_sec":         outro_start,
            "spectral_centroid_mean":  spectral_centroid_mean,
            "danceability":            dance,
            "sections":                sections,
        })

    except Exception as e:
        result["analysis_error"] = f"Analysis failed: {e}"
        result["analyzed_at"]    = datetime.now().isoformat()
        return result

    # ── Write cache ───────────────────────────────────────────────────────────
    try:
        cache_file.write_text(json.dumps(result))
    except Exception:
        pass  # cache write failure is non-fatal

    return result


# ── Parallel runner ───────────────────────────────────────────────────────────

def analyze_library(
    tracks: list[dict],
    cache_dir: Path,
    n_jobs: int = -1,
) -> tuple[list[dict], dict]:
    """
    Analyze all tracks in parallel. Returns (results list, stats dict).

    n_jobs: number of parallel workers.
        -1 = use all CPU cores
         1 = single-threaded (easier to debug)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(tracks)

    print(f"\n{bold(green(f'🔬 Analyzing {total} tracks...'))} "
          f"{dim('(cached tracks will be instant)')}\n")

    start = time.time()

    # Run analysis — joblib handles parallelism + progress
    # prefer="threads" avoids pickling issues with numpy arrays in some setups;
    # for CPU-bound work "processes" is faster but requires picklable functions.
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(analyze_track)(track, cache_dir) for track in tracks
    )

    elapsed = time.time() - start

    # ── Stats ─────────────────────────────────────────────────────────────────
    successful  = [r for r in results if r["analysis_error"] is None]
    failed      = [r for r in results if r["analysis_error"] is not None]
    cache_hits  = [r for r in results if r["cache_hit"]]

    bpms = [r["bpm"] for r in successful if r["bpm"]]
    keys = [r["camelot"] for r in successful if r["camelot"]]

    stats = {
        "analyzed_at":     datetime.now().isoformat(),
        "total_tracks":    total,
        "successful":      len(successful),
        "failed":          len(failed),
        "cache_hits":      len(cache_hits),
        "elapsed_seconds": round(elapsed, 1),
        "bpm_range":       [round(min(bpms), 1), round(max(bpms), 1)] if bpms else None,
        "bpm_mean":        round(float(np.mean(bpms)), 1) if bpms else None,
        "key_distribution": {k: keys.count(k) for k in sorted(set(keys))},
        "errors":          [(r["filename"], r["analysis_error"]) for r in failed],
    }

    return results, stats


# ── Display ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], stats: dict):
    """Print analysis summary to console."""
    print()
    _rule("═")
    print(bold("  🔬 Analysis Summary"))
    _rule("═")

    rows = [
        ("Tracks analyzed",  f"{stats['successful']}/{stats['total_tracks']}"),
        ("Cache hits",       str(stats["cache_hits"])),
        ("Failed",           str(stats["failed"])),
        ("Time elapsed",     f"{stats['elapsed_seconds']}s"),
        ("BPM range",        f"{stats['bpm_range'][0]}–{stats['bpm_range'][1]}" if stats["bpm_range"] else "n/a"),
        ("Mean BPM",         str(stats["bpm_mean"]) if stats["bpm_mean"] else "n/a"),
    ]
    for label, value in rows:
        color = red if (label == "Failed" and stats["failed"] > 0) else cyan
        print(f"  {color(label + ':'):<28} {value}")
    _rule()

    # Sample table
    successful = [r for r in results if r["analysis_error"] is None]
    if successful:
        print()
        _rule("─")
        print(bold("  🎵 Analysis Sample (first 10)"))
        _rule("─")
        header = f"  {'Filename':<30} {'BPM':>6}  {'Key':<5}  {'Camelot':<8}  {'Dance':>6}  {'Intro':>6}"
        print(dim(header))
        _rule("─")
        for r in successful[:10]:
            name     = r["filename"][:28]
            bpm      = f"{r['bpm']:.1f}" if r["bpm"] else "?"
            key      = f"{r['key_name']} {r['mode'][:3]}" if r["key_name"] else "?"
            cam      = r["camelot"] or "?"
            dance    = f"{r['danceability']:.2f}" if r["danceability"] is not None else "?"
            intro    = f"{r['intro_end_sec']:.1f}s" if r["intro_end_sec"] is not None else "?"
            print(f"  {name:<30} {cyan(bpm):>6}  {key:<5}  {green(cam):<8}  {dance:>6}  {dim(intro):>6}")
        _rule("─")

    if stats["errors"]:
        print()
        print(yellow("⚠  Analysis Errors:"))
        for filename, err in stats["errors"]:
            print(f"  {red('•')} {filename}: {err}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    tracks_path: str = "./data/tracks.json",
    output_path: str = "./data/analysis.json",
    cache_dir:   str = "./data/.analysis_cache",
    n_jobs:      int = -1,
) -> dict:
    """
    Main entry point for Module 2.
    Reads tracks.json, analyzes all tracks, writes analysis.json.
    Returns the full output dict.
    """
    _rule("═")
    print(bold("  🎛️  AutoDJ — Module 2: Audio Analyzer"))
    _rule("═")

    # Load tracks registry from Module 1
    tracks_file = Path(tracks_path)
    if not tracks_file.exists():
        print(red(f"  ✗ tracks.json not found at {tracks_file.resolve()}"))
        print(red("  → Run Module 1 first: uv run python -m modules.ingestor --music-dir <path>"))
        sys.exit(1)

    with open(tracks_file) as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        print(yellow("  ⚠  No tracks found in tracks.json — nothing to analyze."))
        return {"analyses": [], "stats": {}}

    print(f"\n  {dim('Tracks registry:')} {tracks_path}")
    print(f"  {dim('Cache directory:')} {cache_dir}")
    print(f"  {dim('Parallel workers:')} {'all cores' if n_jobs == -1 else n_jobs}\n")

    results, stats = analyze_library(
        tracks=tracks,
        cache_dir=Path(cache_dir),
        n_jobs=n_jobs,
    )

    print_summary(results, stats)

    # Write output
    output = {"analyses": results, "stats": stats}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(green(f"  ✅ analysis.json written → {out_path.resolve()}"))
    _rule("═")

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDJ Module 2 — Audio Analyzer")
    parser.add_argument("--tracks",  "-t", default="./data/tracks.json",
                        help="Path to tracks.json from Module 1")
    parser.add_argument("--output",  "-o", default="./data/analysis.json",
                        help="Output path for analysis.json")
    parser.add_argument("--cache",   "-c", default="./data/.analysis_cache",
                        help="Directory for cached analysis results")
    parser.add_argument("--jobs",    "-j", type=int, default=-1,
                        help="Parallel workers (-1 = all cores, 1 = single-threaded)")
    args = parser.parse_args()

    run(
        tracks_path=args.tracks,
        output_path=args.output,
        cache_dir=args.cache,
        n_jobs=args.jobs,
    )