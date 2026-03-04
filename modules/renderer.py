"""
Module 5: Renderer
──────────────────
Reads session.json and renders a continuous audio mix to a single MP3 or WAV.

Transition pipeline (per slot pair):
  1. Load both tracks via pydub
  2. Trim to intro_end → outro_start (cue points from Module 2)
  3. Beat-align: find nearest beat boundary for splice (quality mode)
  4. BPM time-stretch incoming track to match outgoing (pyrubberband, quality mode)
  5. EQ swap: low-pass outgoing + high-pass incoming during overlap (pedalboard)
  6. Volume crossfade over the overlap window
  7. Concatenate into running mix buffer

Chaos factor modulates which steps are applied:
  chaos 0.0–0.3  → full pipeline (beat-align + stretch + EQ + fade)
  chaos 0.3–0.6  → EQ + fade (no stretch)
  chaos 0.6–0.8  → volume fade only
  chaos 0.8–1.0  → hard cut (or quick fade, randomly)

Modes:
  --preview   fast mode: no time-stretching, shorter fades, 128k MP3
  --quality   full pipeline, 320k MP3 or WAV

Usage:
    uv run python -m modules.renderer
    uv run python -m modules.renderer --preview
    uv run python -m modules.renderer --quality --output output/mix.wav
"""

import sys, json, math, random, argparse, warnings, tempfile, time
from pathlib import Path
from datetime import datetime

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

warnings.filterwarnings("ignore")

# Optional imports — degrade gracefully
try:
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
except Exception:
    RUBBERBAND_AVAILABLE = False

try:
    from pedalboard import Pedalboard, LowpassFilter, HighpassFilter, Gain
    PEDALBOARD_AVAILABLE = True
except Exception:
    PEDALBOARD_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 44100
CHANNELS        = 2

# EQ band crossover frequencies (Hz)
# Bass: 0–200Hz  |  Mids: 200–800Hz  |  Highs: 800Hz+
EQ_BASS_HZ   = 200.0
EQ_MID_HZ    = 800.0

# Energy threshold — below this RMS we consider a region "soft"
SOFT_ENERGY_THRESHOLD = 0.03

# Transition phase lengths in bars (at tempo)
# Phase 2 (highs only): BPM-adaptive — 8 bars at fast tempo, 16 at slow
# Phase 3 (bass swap):  chaos-driven — 2 bars low chaos, 4 bars high chaos
# Phase 4 (outro):      same as phase 2
PHASE_HIGHS_BARS_FAST = 8    # BPM >= 120
PHASE_HIGHS_BARS_SLOW = 16   # BPM <  120
PHASE_SWAP_BARS_TIGHT = 2    # chaos < 0.5
PHASE_SWAP_BARS_LOOSE = 4    # chaos >= 0.5
PHASE_OUTRO_BARS      = 16   # outgoing fades after bass swap

# Looping (P2-1): extend outgoing track if outro is too short for a full blend
LOOP_BARS_DEFAULT  = 4     # default loop length in bars
LOOP_BARS_LONG     = 8     # longer loop for slow tempos
LOOP_MAX_REPEATS   = 4     # max times to repeat the loop segment
LOOP_MIN_CONTENT_MS = 8000 # minimum outgoing content after p2_start before looping kicks in

# quick_fade and crossfade simple durations (ms) — no staging
QUICK_FADE_MS  = 4_000
CROSSFADE_MS   = 12_000

# Minimum tail on a hard_cut so there's never dead silence
HARD_CUT_TAIL_MS = 2_000

# Phrase boundary: 8 bars × 4 beats = 32 beats
PHRASE_BEATS = 32


# ── Transition helpers ───────────────────────────────────────────────────────

def bpm_blend_ms(bpm, preview_mode=False):
    """
    Compute long_blend total duration from BPM (phases 2+3+4 combined).
    Returns milliseconds. Preview mode halves phase lengths.
    """
    if not bpm or bpm <= 0:
        bpm = 120.0
    sec_per_bar   = 4 * 60.0 / bpm
    highs_bars    = PHASE_HIGHS_BARS_FAST if bpm >= 120 else PHASE_HIGHS_BARS_SLOW
    outro_bars    = PHASE_OUTRO_BARS
    swap_bars     = PHASE_SWAP_BARS_TIGHT  # default; overridden at render time
    total_bars    = highs_bars + swap_bars + outro_bars
    if preview_mode:
        total_bars = max(8, total_bars // 2)
    return int(total_bars * sec_per_bar * 1000)


def bars_to_ms(bars, bpm):
    """Convert a bar count to milliseconds at the given BPM."""
    if not bpm or bpm <= 0:
        bpm = 120.0
    return int(bars * 4 * 60.0 / bpm * 1000)


def loop_segment(seg, loop_bars, bpm, beat_times=None):
    """
    Extract the last `loop_bars` bars from a segment and loop them to create
    an extended version. Returns the looped audio (just the loop repetitions,
    NOT the original segment).

    The loop point snaps to the nearest beat boundary for a clean splice.
    Uses simple pydub concatenation — the loop is musically identical content
    repeated, so no time-stretching is needed.

    Args:
        seg:        AudioSegment to extract loop from
        loop_bars:  number of bars to loop (typically 4 or 8)
        bpm:        tempo for bar duration calculation
        beat_times: list of beat timestamps in seconds (for beat-snapping)

    Returns:
        (loop_audio, loop_point_ms):
            loop_audio    = AudioSegment of the repeated loop
            loop_point_ms = position in original seg where the loop was taken from
    """
    loop_ms = bars_to_ms(loop_bars, bpm)

    # Don't loop if the segment is too short
    if len(seg) < loop_ms * 1.5:
        return AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE), len(seg)

    # Find the loop start point — ideally on a beat/phrase boundary
    raw_loop_start = len(seg) - loop_ms

    if beat_times:
        # Snap to nearest beat boundary at or before raw_loop_start
        target_sec = raw_loop_start / 1000.0
        candidates = [b for b in beat_times if b <= target_sec]
        if candidates:
            loop_start_ms = int(candidates[-1] * 1000)
        else:
            loop_start_ms = raw_loop_start
    else:
        loop_start_ms = raw_loop_start

    loop_start_ms = max(0, loop_start_ms)

    # Extract the loop region
    loop_region = seg[loop_start_ms:]

    # Repeat the loop region (up to LOOP_MAX_REPEATS times)
    # Apply a tiny crossfade at each join to avoid clicks
    loop_audio = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    xfade_ms = min(50, len(loop_region) // 4)  # tiny crossfade for seamless joins

    for i in range(LOOP_MAX_REPEATS):
        if len(loop_audio) == 0:
            loop_audio = loop_region
        else:
            if xfade_ms > 0 and len(loop_audio) > xfade_ms and len(loop_region) > xfade_ms:
                loop_audio = loop_audio.append(loop_region, crossfade=xfade_ms)
            else:
                loop_audio = loop_audio + loop_region

    return loop_audio, loop_start_ms


def find_phrase_boundary(beat_times_sec, target_ms, phrase_beats=PHRASE_BEATS):
    """
    Find the next clean phrase boundary (every phrase_beats beats) at or
    after target_ms. Falls back to nearest beat if no phrase boundary found
    within a reasonable window (8 bars).
    Returns boundary position in ms.
    """
    if not beat_times_sec:
        return target_ms

    target_sec = target_ms / 1000.0
    beats      = [b for b in beat_times_sec if b >= target_sec]

    if not beats:
        return target_ms

    # Find beats that align to phrase boundaries (every phrase_beats beats)
    all_beats   = beat_times_sec
    phrase_boundaries = [
        all_beats[i] for i in range(0, len(all_beats), phrase_beats)
        if all_beats[i] >= target_sec
    ]

    if phrase_boundaries:
        # Take the next phrase boundary after target
        return int(phrase_boundaries[0] * 1000)

    # No phrase boundary found — snap to nearest beat
    arr = [b for b in beat_times_sec if b >= target_sec]
    return int(arr[0] * 1000) if arr else target_ms


def find_breakdown_point(energy_curve, outro_start_ratio=0.6):
    """
    Scan the track's energy curve (list of 0-1 floats, 32 segments from Module 2)
    for the lowest energy point in the latter portion of the track
    (after outro_start_ratio through the track).

    This finds breakdowns, drops, or quiet moments near the outro — ideal
    transition start points. Returns the segment index of the best point,
    or None if no energy curve is available.
    """
    if not energy_curve or len(energy_curve) < 4:
        return None
    n          = len(energy_curve)
    start_idx  = int(n * outro_start_ratio)
    window     = energy_curve[start_idx:]
    if not window:
        return None
    min_val    = min(window)
    min_idx    = start_idx + window.index(min_val)
    return min_idx


def find_breakdown_from_sections(sections, outro_start_ratio=0.55, track_duration_ms=None):
    """
    Find the best breakdown point using structural section data from P2-6.

    Looks for sections labelled "breakdown" in the latter portion of the track.
    If no breakdown is found, looks for any low-energy section near the end.

    Returns breakdown position in milliseconds, or None if no suitable point found.
    """
    if not sections or not track_duration_ms:
        return None

    track_dur_sec = track_duration_ms / 1000.0
    search_start  = track_dur_sec * outro_start_ratio

    # Priority 1: explicitly labelled breakdowns after the search start
    breakdowns = [
        s for s in sections
        if s["label"] == "breakdown" and s["start_sec"] >= search_start
    ]
    if breakdowns:
        # Pick the one closest to the end (best for transition timing)
        best = max(breakdowns, key=lambda s: s["start_sec"])
        return int(best["start_sec"] * 1000)

    # Priority 2: any low-energy section in the latter portion
    low_energy = [
        s for s in sections
        if s.get("energy_category") == "low" and s["start_sec"] >= search_start
        and s["label"] not in ("intro",)  # don't use intro sections
    ]
    if low_energy:
        best = max(low_energy, key=lambda s: s["start_sec"])
        return int(best["start_sec"] * 1000)

    # Priority 3: transition between chorus→verse or chorus→bridge
    for i in range(len(sections) - 1):
        curr = sections[i]
        nxt  = sections[i + 1]
        if (nxt["start_sec"] >= search_start
            and curr.get("energy_category") == "high"
            and nxt.get("energy_category") in ("mid", "low")):
            return int(nxt["start_sec"] * 1000)

    return None


def segment_idx_to_ms(seg_idx, total_ms, n_segments=32):
    """Convert a Module 2 energy segment index to a millisecond position."""
    return int(seg_idx / n_segments * total_ms)


def compute_crossover_hz(spectral_centroid_mean):
    """
    Derive the bass/mid EQ crossover frequency from the track's spectral centroid.

    Spectral centroid is the "brightness" centre of the spectrum.
    Typical values:
      Techno/DnB  → low centroid (dark, bass-heavy) → lower crossover (~80Hz)
      House       → mid centroid → medium crossover (~120Hz)
      Bright/HiNRG → high centroid → higher crossover (~180Hz)

    We scale linearly: centroid 500Hz → 60Hz crossover
                       centroid 4000Hz → 200Hz crossover
    Clamped to 60–250Hz.
    """
    if not spectral_centroid_mean or spectral_centroid_mean <= 0:
        return EQ_BASS_HZ  # fallback

    # Linear interpolation: map centroid 500–4000 → crossover 60–200
    lo_c, hi_c = 500.0,  4000.0
    lo_x, hi_x = 60.0,   200.0
    ratio = (spectral_centroid_mean - lo_c) / (hi_c - lo_c)
    ratio = max(0.0, min(1.0, ratio))
    return round(lo_x + ratio * (hi_x - lo_x), 1)


def apply_highpass_adaptive(seg, cutoff_hz, chaos):
    """
    Apply a highpass filter with shape adapted to chaos level.
    Low chaos  → hard filter (steep, decisive — Linkwitz-Riley style via cascaded biquads)
    High chaos → gentle shelf (gradual rolloff — sounds more natural/blended)

    Pedalboard doesn't expose filter order directly, so we simulate:
      hard   = cascade two HighpassFilters at same cutoff (steeper rolloff)
      shelf  = single HighpassFilter at lower cutoff (softer knee)
    """
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        audio = seg_to_float(seg)
        if chaos < 0.4:
            # Hard: cascade two filters for steep rolloff
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=cutoff_hz),
                HighpassFilter(cutoff_frequency_hz=cutoff_hz),
            ])
        else:
            # Shelf: single filter at 60% of cutoff for gentle knee
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=cutoff_hz * 0.6),
            ])
        return float_to_seg(board(audio, SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def apply_lowpass_adaptive(seg, cutoff_hz, chaos):
    """Adaptive lowpass — hard at low chaos, shelf at high chaos."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        audio = seg_to_float(seg)
        if chaos < 0.4:
            board = Pedalboard([
                LowpassFilter(cutoff_frequency_hz=cutoff_hz),
                LowpassFilter(cutoff_frequency_hz=cutoff_hz),
            ])
        else:
            board = Pedalboard([
                LowpassFilter(cutoff_frequency_hz=cutoff_hz * 1.4),
            ])
        return float_to_seg(board(audio, SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def measure_tail_energy(seg, window_sec=3.0):
    """
    Measure RMS energy of the last window_sec of a segment.
    Returns a float 0–1 (normalised to int16 range).
    """
    window_ms = int(window_sec * 1000)
    tail      = seg[-min(window_ms, len(seg)):]
    samples   = np.array(tail.get_array_of_samples(), dtype=np.float32)
    samples  /= (2 ** (tail.sample_width * 8 - 1))
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


def measure_head_energy(seg, window_sec=3.0):
    """
    Measure RMS energy of the first window_sec of a segment.
    Returns a float 0–1 (normalised to int16 range).
    """
    window_ms = int(window_sec * 1000)
    head      = seg[:min(window_ms, len(seg))]
    samples   = np.array(head.get_array_of_samples(), dtype=np.float32)
    samples  /= (2 ** (head.sample_width * 8 - 1))
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


def energy_aware_hint(hint, outgoing_seg, incoming_seg):
    """
    Override the planner's transition hint based on actual audio energy
    at the splice point.

    Rules:
      - Both ends soft  → long_blend (gentle overlap, no jarring cut)
      - Outgoing soft, incoming loud → crossfade (let incoming build in)
      - Outgoing loud, incoming soft → crossfade (ease the landing)
      - Both loud + hint is hard_cut → keep hard_cut (intentional drop)
      - Otherwise → keep planner hint
    """
    tail_energy = measure_tail_energy(outgoing_seg)
    head_energy = measure_head_energy(incoming_seg)

    tail_soft = tail_energy < SOFT_ENERGY_THRESHOLD
    head_soft = head_energy < SOFT_ENERGY_THRESHOLD

    if tail_soft and head_soft:
        return "long_blend"
    if tail_soft or head_soft:
        # One side is soft — a hard cut would sound broken
        if hint == "hard_cut":
            return "crossfade"
        return hint
    # Both loud — respect the planner
    return hint


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_track(filepath, preview_mode=False):
    """
    Load an audio file as a pydub AudioSegment.
    Normalises to stereo 44100Hz. Returns None on failure.
    """
    try:
        seg = AudioSegment.from_file(filepath)
        seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
        return seg
    except Exception as e:
        print(f"  ⚠  Could not load {Path(filepath).name}: {e}")
        return None


def trim_to_cues(seg, intro_end_sec, outro_start_sec):
    """
    Trim a segment to its DJ cue points (intro_end → outro_start).
    Falls back gracefully if cue points are missing or out of range.
    """
    duration_ms = len(seg)
    start_ms = max(0, int((intro_end_sec or 0) * 1000))
    end_ms   = min(duration_ms, int((outro_start_sec or duration_ms / 1000) * 1000))

    # Sanity: need at least 10 seconds of content
    if end_ms - start_ms < 10_000:
        return seg

    return seg[start_ms:end_ms]


# ── Beat alignment ────────────────────────────────────────────────────────────

def nearest_beat_ms(position_ms, beat_times_sec):
    """
    Given a position in ms and a list of beat timestamps (seconds),
    return the nearest beat position in ms.
    """
    if not beat_times_sec:
        return position_ms
    position_sec = position_ms / 1000.0
    beat_arr     = np.array(beat_times_sec)
    idx          = int(np.argmin(np.abs(beat_arr - position_sec)))
    return int(beat_arr[idx] * 1000)


# ── Time stretching ───────────────────────────────────────────────────────────

def time_stretch_segment(seg, from_bpm, to_bpm, max_stretch=0.15):
    """
    Time-stretch a pydub AudioSegment from from_bpm to to_bpm using pyrubberband.
    Clamps stretch ratio to ±max_stretch (default 15%) to avoid artifacts.
    Returns the original segment if stretch is out of range or rubberband unavailable.
    """
    if not RUBBERBAND_AVAILABLE or not from_bpm or not to_bpm:
        return seg

    ratio = from_bpm / to_bpm
    # Clamp: don't stretch more than max_stretch in either direction
    ratio = max(1 - max_stretch, min(1 + max_stretch, ratio))

    if abs(ratio - 1.0) < 0.01:
        return seg  # negligible — skip

    try:
        # Convert pydub → numpy float32
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= (2 ** (seg.sample_width * 8 - 1))  # normalise to -1..1

        if seg.channels == 2:
            samples = samples.reshape(-1, 2)
            stretched_l = pyrb.time_stretch(samples[:, 0], SAMPLE_RATE, ratio)
            stretched_r = pyrb.time_stretch(samples[:, 1], SAMPLE_RATE, ratio)
            stretched   = np.stack([stretched_l, stretched_r], axis=1).flatten()
        else:
            stretched = pyrb.time_stretch(samples, SAMPLE_RATE, ratio)

        # Convert back to int16 pydub segment
        stretched = np.clip(stretched, -1.0, 1.0)
        pcm = (stretched * (2 ** 15 - 1)).astype(np.int16)
        return AudioSegment(
            pcm.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=seg.channels,
        )
    except Exception as e:
        print(f"    ⚠  Time-stretch failed ({e}) — using original tempo")
        return seg


# ── EQ helpers ───────────────────────────────────────────────────────────────

def seg_to_float(s):
    """Convert pydub AudioSegment to float32 numpy array (channels, samples)."""
    arr = np.array(s.get_array_of_samples(), dtype=np.float32)
    arr /= (2 ** (s.sample_width * 8 - 1))
    if s.channels == 2:
        return arr.reshape(-1, 2).T
    return arr.reshape(1, -1)


def float_to_seg(arr, channels):
    """Convert float32 numpy array back to pydub AudioSegment."""
    if channels == 2:
        arr = arr.T.flatten()
    else:
        arr = arr.flatten()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * (2 ** 15 - 1)).astype(np.int16)
    return AudioSegment(pcm.tobytes(), frame_rate=SAMPLE_RATE,
                        sample_width=2, channels=channels)


def apply_highpass(seg, cutoff_hz):
    """Apply a highpass filter to a segment (cuts bass below cutoff_hz)."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_hz)])
        return float_to_seg(board(seg_to_float(seg), SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def apply_lowpass(seg, cutoff_hz):
    """Apply a lowpass filter to a segment (cuts highs above cutoff_hz)."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        board = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_hz)])
        return float_to_seg(board(seg_to_float(seg), SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


# ── Transition strategy ──────────────────────────────────────────────────────

def transition_strategy(hint, chaos, preview_mode, bpm=None):
    """
    Return a strategy dict describing how to execute the transition.

    For long_blend:  staged 3-phase DJ transition (the real deal)
    For crossfade:   simple overlapping volume fade with EQ at low chaos
    For quick_fade:  short volume fade, no EQ
    For hard_cut:    sequential tail-fade + head-fade, no overlap

    Returns:
        {
          "method":        "staged" | "crossfade" | "quick_fade" | "hard_cut"
          "do_stretch":    bool
          "do_phrase_align": bool
          # staged only:
          "highs_ms":      int   phase 2 duration
          "swap_ms":       int   phase 3 duration
          "outro_ms":      int   phase 4 duration
          # simple only:
          "fade_ms":       int
        }
    """
    bpm = bpm or 120.0

    do_stretch      = RUBBERBAND_AVAILABLE and not preview_mode and chaos < 0.5
    do_phrase_align = not preview_mode and chaos < 0.7

    if hint == "long_blend":
        highs_bars = PHASE_HIGHS_BARS_FAST if bpm >= 120 else PHASE_HIGHS_BARS_SLOW
        swap_bars  = PHASE_SWAP_BARS_TIGHT if chaos < 0.5 else PHASE_SWAP_BARS_LOOSE
        outro_bars = PHASE_OUTRO_BARS
        if preview_mode:
            highs_bars = max(4, highs_bars // 2)
            swap_bars  = 1
            outro_bars = max(4, outro_bars // 2)
        return {
            "method":          "staged",
            "do_stretch":      do_stretch,
            "do_phrase_align": do_phrase_align,
            "highs_ms":        bars_to_ms(highs_bars, bpm),
            "swap_ms":         bars_to_ms(swap_bars,  bpm),
            "outro_ms":        bars_to_ms(outro_bars, bpm),
        }

    if hint == "crossfade":
        fade_ms = min(CROSSFADE_MS, CROSSFADE_MS // 2 if preview_mode else CROSSFADE_MS)
        return {
            "method":          "crossfade",
            "do_stretch":      do_stretch,
            "do_phrase_align": False,
            "fade_ms":         fade_ms,
            "do_eq":           PEDALBOARD_AVAILABLE and chaos < 0.5,
        }

    if hint == "quick_fade":
        return {
            "method":          "quick_fade",
            "do_stretch":      False,
            "do_phrase_align": False,
            "fade_ms":         QUICK_FADE_MS,
            "do_eq":           False,
        }

    # hard_cut
    return {
        "method":          "hard_cut",
        "do_stretch":      False,
        "do_phrase_align": False,
        "fade_ms":         HARD_CUT_TAIL_MS,
        "do_eq":           False,
    }


# ── Core render ───────────────────────────────────────────────────────────────

def _stretch_incoming(incoming, bpm_in, bpm_out, window_ms):
    """Time-stretch the first window_ms + buffer of incoming to match bpm_out."""
    if not bpm_in or not bpm_out or abs(bpm_in - bpm_out) < 1:
        return incoming
    buf      = 4_000
    head     = incoming[:window_ms + buf]
    rest     = incoming[window_ms + buf:]
    return time_stretch_segment(head, bpm_in, bpm_out) + rest


def render_staged(outgoing, incoming, slot_out, slot_in, strategy, chaos=0.3):
    """
    Full DJ transition — 4 phases:

      Phase 1  outgoing plays alone, transition held until breakdown point
               + next phrase boundary (the "right moment")
      Phase 2  incoming highs enter (bass cut); outgoing plays full-range
               (crowd hears a new layer on top, bass stays locked to outgoing)
      Phase 3  bass swap — outgoing bass cut, incoming bass brought in
               (tight at low chaos, loose at high chaos)
      Phase 4  incoming plays full-range; outgoing highs/mids fade out

    EQ crossover derived from spectral centroid (adaptive per track).
    Filter shape: hard (cascaded) at low chaos, shelf (gentle) at high chaos.
    """
    bpm        = slot_out.get("actual_bpm") or 120.0
    beat_times = slot_out.get("beat_times") or []
    highs_ms   = strategy["highs_ms"]
    swap_ms    = strategy["swap_ms"]
    outro_ms   = strategy["outro_ms"]
    total_ms   = highs_ms + swap_ms + outro_ms

    # ── Adaptive EQ crossover from spectral centroid ──────────────────────────
    centroid   = slot_out.get("spectral_centroid_mean")
    bass_hz    = compute_crossover_hz(centroid)
    # Highs cutoff: 4× bass crossover, clamped to 400–1200Hz
    highs_hz   = max(400.0, min(1200.0, bass_hz * 4))

    # ── Find breakdown point — prefer structural sections (P2-6) ─────────────
    sections = slot_out.get("sections") or []
    breakdown_ms_from_sections = find_breakdown_from_sections(
        sections, outro_start_ratio=0.55, track_duration_ms=len(outgoing)
    )

    if breakdown_ms_from_sections is not None and beat_times:
        # Sections found a breakdown — use it
        phrase_ms = find_phrase_boundary(beat_times, breakdown_ms_from_sections)
        phrase_ms = min(phrase_ms, int(len(outgoing) * 0.82))
        total_available = len(outgoing) - phrase_ms
        if total_available >= total_ms * 0.5:
            if total_available < total_ms:
                scale    = total_available / total_ms
                highs_ms = int(highs_ms * scale)
                swap_ms  = max(int(swap_ms * scale), 500)
                outro_ms = int(outro_ms * scale)
                total_ms = highs_ms + swap_ms + outro_ms

    # ── Fallback: energy curve breakdown detection ────────────────────────────
    elif breakdown_ms_from_sections is None:
        energy_curve = (slot_out.get("energy") or {}).get("curve") or []
        breakdown_idx = find_breakdown_point(energy_curve, outro_start_ratio=0.55)
        if breakdown_idx is not None and beat_times:
            breakdown_ms = segment_idx_to_ms(breakdown_idx, len(outgoing))
            # Now find the next phrase boundary at or after the breakdown
            phrase_ms = find_phrase_boundary(beat_times, breakdown_ms)
            phrase_ms = min(phrase_ms, int(len(outgoing) * 0.82))
            total_available = len(outgoing) - phrase_ms
            if total_available >= total_ms * 0.5:
                # Good — we have room. Redistribute if needed.
                if total_available < total_ms:
                    scale    = total_available / total_ms
                    highs_ms = int(highs_ms * scale)
                    swap_ms  = max(int(swap_ms * scale), 500)
                    outro_ms = int(outro_ms * scale)
                    total_ms = highs_ms + swap_ms + outro_ms
            # else: breakdown too late in track, fall through to normal phrase align

    # ── Phrase align the splice point (if breakdown search didn't already) ────
    elif strategy["do_phrase_align"] and beat_times:
        target_ms  = max(0, len(outgoing) - total_ms)
        aligned_ms = find_phrase_boundary(beat_times, target_ms)
        aligned_ms = min(aligned_ms, int(len(outgoing) * 0.80))
        total_available = len(outgoing) - aligned_ms
        if total_available < total_ms and total_ms > 0:
            scale    = total_available / total_ms
            highs_ms = int(highs_ms * scale)
            swap_ms  = max(int(swap_ms  * scale), 500)
            outro_ms = int(outro_ms * scale)
            total_ms = highs_ms + swap_ms + outro_ms

    # ── Clamp everything to available audio ───────────────────────────────────
    max_out  = max(4000, len(outgoing) - 2000)
    max_in   = max(4000, len(incoming) - 2000)
    total_ms = min(total_ms, max_out, max_in)
    # Ensure phases fit inside total_ms with minimum viable swap
    swap_ms  = max(500, min(swap_ms,  total_ms // 4))
    highs_ms = max(500, min(highs_ms, (total_ms - swap_ms) // 2))
    outro_ms = max(500, total_ms - highs_ms - swap_ms)
    # Re-clamp total to actual sum of phases
    total_ms = highs_ms + swap_ms + outro_ms
    total_ms = min(total_ms, max_out, max_in)
    # Final phase redistribution to honour clamped total
    phase_sum = highs_ms + swap_ms + outro_ms
    if phase_sum > total_ms and phase_sum > 0:
        ratio    = total_ms / phase_sum
        highs_ms = max(500, int(highs_ms * ratio))
        swap_ms  = max(500, int(swap_ms  * ratio))
        outro_ms = max(500, total_ms - highs_ms - swap_ms)

    # ── Time-stretch incoming to match outgoing BPM ───────────────────────────
    if strategy["do_stretch"]:
        bpm_in  = slot_in.get("actual_bpm")
        bpm_out = slot_out.get("actual_bpm")
        incoming = _stretch_incoming(incoming, bpm_in, bpm_out, total_ms)

    # ── Looping: extend outgoing if outro too short for blend (P2-1) ──────────
    p2_start_tentative = max(0, len(outgoing) - total_ms)
    tail_available_ms  = len(outgoing) - p2_start_tentative

    # Decide loop length: 8 bars for slow BPM, 4 bars for fast
    loop_bars = LOOP_BARS_LONG if bpm < 120 else LOOP_BARS_DEFAULT
    loop_ms   = bars_to_ms(loop_bars, bpm)

    # Loop if: (a) not enough outgoing content for the blend, or
    #          (b) always loop to keep energy steady during transition
    needs_loop = (tail_available_ms < total_ms) or (tail_available_ms < total_ms * 1.2)

    if needs_loop and loop_ms > 0 and len(outgoing) > loop_ms * 1.5:
        loop_audio, loop_point_ms = loop_segment(
            outgoing, loop_bars, bpm, beat_times
        )
        if len(loop_audio) > 0:
            # Splice: keep outgoing up to the loop point, then append loop
            # This extends the outgoing track with repeated bars
            outgoing = outgoing[:loop_point_ms] + loop_audio

    # ── Slice regions ──────────────────────────────────────────────────────────
    # outgoing: body | phase2_region | phase3_region | (ends)
    p2_start = max(0, len(outgoing) - (highs_ms + swap_ms + outro_ms))
    p3_start = p2_start + highs_ms
    p4_start = p3_start + swap_ms

    out_body  = outgoing[:p2_start]                # plays alone
    out_p2    = outgoing[p2_start:p3_start]        # full range, incoming highs on top
    out_p3    = outgoing[p3_start:p4_start]        # bass cut here (swap point)
    out_p4    = outgoing[p4_start:]                # highs fade out, incoming takes over

    inc_p2    = incoming[:highs_ms]                # highs only (bass cut)
    inc_p3    = incoming[highs_ms:highs_ms+swap_ms] # bass comes in
    inc_p4    = incoming[highs_ms+swap_ms:highs_ms+swap_ms+outro_ms]  # full range
    inc_body  = incoming[highs_ms+swap_ms+outro_ms:]  # rest of track, unmodified

    # ── Phase 2: incoming highs only over outgoing full range ─────────────────
    # Strip bass+mids from incoming — crowd hears a shimmering layer on top
    inc_p2_hi  = apply_highpass_adaptive(inc_p2, highs_hz, chaos)
    overlap_p2 = out_p2.overlay(inc_p2_hi)

    # ── Phase 3: bass swap — decisive, tight window ───────────────────────────
    # Cut outgoing bass, bring in incoming bass simultaneously.
    # Two basslines playing together = mud. This is the critical handoff.
    out_p3_nobase = apply_highpass_adaptive(out_p3, bass_hz, chaos)
    inc_p3_full   = inc_p3  # incoming enters full range (bass now present)
    overlap_p3    = out_p3_nobase.overlay(inc_p3_full)

    # ── Phase 4: outgoing highs fade out, incoming full range ─────────────────
    # Outgoing highs linger briefly then dissolve — incoming fully takes over
    out_p4_hi  = apply_highpass_adaptive(out_p4, highs_hz, chaos).fade_out(outro_ms)
    overlap_p4 = inc_p4.overlay(out_p4_hi)

    return out_body + overlap_p2 + overlap_p3 + overlap_p4 + inc_body


def render_transition(outgoing, incoming, slot_out, slot_in, chaos, preview_mode):
    """
    Entry point for all transitions.
    Determines strategy via energy + hint + chaos, dispatches to the
    appropriate render function.
    """
    hint     = slot_in.get("transition_hint", "crossfade")
    hint     = energy_aware_hint(hint, outgoing, incoming)
    bpm      = slot_out.get("actual_bpm") or slot_in.get("actual_bpm")
    strategy = transition_strategy(hint, chaos, preview_mode, bpm=bpm)
    method   = strategy["method"]

    # ── Staged DJ transition ──────────────────────────────────────────────────
    if method == "staged":
        return render_staged(outgoing, incoming, slot_out, slot_in, strategy, chaos=chaos)

    # ── Simple crossfade ──────────────────────────────────────────────────────
    fade_ms = strategy["fade_ms"]
    fade_ms = max(0, min(fade_ms, len(outgoing) - 1000, len(incoming) - 1000))

    if fade_ms <= 0:
        return outgoing + incoming

    if method == "hard_cut":
        # Sequential tail-fade + head-fade — breath between tracks, no overlap
        out_body = outgoing[:-fade_ms]
        out_tail = outgoing[-fade_ms:].fade_out(fade_ms)
        in_head  = incoming[:fade_ms].fade_in(fade_ms)
        in_body  = incoming[fade_ms:]
        return out_body + out_tail + in_head + in_body

    # crossfade / quick_fade — overlap the fade regions
    if strategy.get("do_stretch"):
        bpm_in  = slot_in.get("actual_bpm")
        bpm_out = slot_out.get("actual_bpm")
        incoming = _stretch_incoming(incoming, bpm_in, bpm_out, fade_ms)

    if strategy.get("do_eq") and PEDALBOARD_AVAILABLE:
        centroid     = slot_out.get("spectral_centroid_mean")
        bass_hz      = compute_crossover_hz(centroid)
        out_tail_raw = outgoing[-fade_ms:]
        in_head_raw  = incoming[:fade_ms]
        out_tail_eq  = apply_highpass_adaptive(out_tail_raw, bass_hz, chaos)
        in_head_eq   = apply_highpass_adaptive(in_head_raw,  bass_hz, chaos)
        outgoing  = outgoing[:-fade_ms] + out_tail_eq
        incoming  = in_head_eq + incoming[fade_ms:]

    out_body = outgoing[:-fade_ms]
    out_tail = outgoing[-fade_ms:].fade_out(fade_ms)
    in_head  = incoming[:fade_ms].fade_in(fade_ms)
    in_body  = incoming[fade_ms:]
    overlap  = out_tail.overlay(in_head)
    return out_body + overlap + in_body


def render_session(setlist, chaos, preview_mode, progress_cb=None):
    """
    Render all slots into a single AudioSegment.
    progress_cb(position, total, filename) called per track if provided.
    """
    mix      = None
    prev_seg = None
    prev_slot = None
    total    = len(setlist)

    for i, slot in enumerate(setlist):
        if progress_cb:
            progress_cb(i, total, slot["filename"])

        seg = load_track(slot["filepath"], preview_mode)
        if seg is None:
            print(f"  ⚠  Skipping {slot['filename']} (could not load)")
            continue

        seg = trim_to_cues(seg, slot.get("intro_end_sec"), slot.get("outro_start_sec"))

        # Normalise loudness per track
        seg = pydub_normalize(seg)

        if mix is None:
            # First track — no transition
            mix = seg
        else:
            mix = render_transition(mix, seg, prev_slot, slot, chaos, preview_mode)

        prev_seg  = seg
        prev_slot = slot

    return mix


# ── Export ────────────────────────────────────────────────────────────────────

def export_mix(mix, output_path, preview_mode):
    """Export the final mix to MP3 or WAV based on file extension and mode."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt     = out.suffix.lstrip(".").lower() or "mp3"
    bitrate = "128k" if preview_mode else "320k"

    export_kwargs = {"format": fmt}
    if fmt == "mp3":
        export_kwargs["bitrate"] = bitrate

    mix.export(str(out), **export_kwargs)
    size_mb = out.stat().st_size / (1024 * 1024)
    return size_mb


# ── Display ───────────────────────────────────────────────────────────────────

def print_progress(position, total, filename, start_time):
    elapsed  = time.time() - start_time
    pct      = int((position / max(total, 1)) * 40)
    bar      = "█" * pct + "░" * (40 - pct)
    eta_str  = ""
    if position > 0:
        eta = elapsed / position * (total - position)
        eta_str = f"  ETA {int(eta)}s"
    print(f"\r  [{bar}] {position}/{total}  {filename[:30]}{eta_str}".ljust(78),
          end="", flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    session_path  = "./data/session.json",
    output_path   = None,
    preview_mode  = False,
    quality_mode  = False,
):
    print("─" * 60 + "\n  AutoDJ — Module 5: Renderer\n" + "─" * 60)

    if not Path(session_path).exists():
        sys.exit(f"✗ {session_path} not found — run Module 4 first.")

    session  = json.loads(Path(session_path).read_text())
    setlist  = session["setlist"]
    chaos    = float(session.get("chaos_factor", 0.3))
    duration = session.get("duration_min", 60)

    if not setlist:
        sys.exit("✗ Setlist is empty — re-run Module 4.")

    # Default output path
    if output_path is None:
        fmt         = "mp3"
        suffix      = "_preview" if preview_mode else ""
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"./output/mix_{timestamp}{suffix}.{fmt}"

    mode_str = "PREVIEW (fast)" if preview_mode else "QUALITY (full pipeline)"

    print(f"\n  Tracks:       {len(setlist)}")
    print(f"  Duration:     ~{duration} min")
    print(f"  Chaos:        {chaos}")
    print(f"  Mode:         {mode_str}")
    print(f"  Output:       {output_path}")
    print(f"\n  Libraries:    rubberband={'✓' if RUBBERBAND_AVAILABLE else '✗ (no stretch)'}  "
          f"pedalboard={'✓' if PEDALBOARD_AVAILABLE else '✗ (no EQ)'}  "
          f"librosa={'✓' if LIBROSA_AVAILABLE else '✗ (no beat-align)'}")

    # Warn about degraded pipeline
    if not preview_mode and not quality_mode:
        if not RUBBERBAND_AVAILABLE:
            print(f"\n  ⚠  pyrubberband not available — time-stretching disabled")
        if not PEDALBOARD_AVAILABLE:
            print(f"  ⚠  pedalboard not available — EQ swap disabled")

    print(f"\n  {'─' * 56}")

    start_time = time.time()

    def progress(pos, total, fname):
        print_progress(pos + 1, total, fname, start_time)

    mix = render_session(setlist, chaos, preview_mode, progress_cb=progress)

    print(f"\r  {'─' * 56}")

    if mix is None:
        sys.exit("✗ Render failed — no audio could be loaded from setlist.")

    print(f"\n  Exporting...")
    size_mb  = export_mix(mix, output_path, preview_mode)
    elapsed  = time.time() - start_time
    duration_actual = len(mix) / 1000 / 60

    print(f"  ✅ Mix rendered in {elapsed:.1f}s")
    print(f"  📁 {output_path}  ({size_mb:.1f} MB, {duration_actual:.1f} min)")
    print("─" * 60)

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDJ Module 5 — Renderer")
    parser.add_argument("--session",  "-s", default="./data/session.json")
    parser.add_argument("--output",   "-o", default=None)
    parser.add_argument("--preview",  "-p", action="store_true",
                        help="Fast mode: no stretching, 128k MP3, shorter fades")
    parser.add_argument("--quality",  "-q", action="store_true",
                        help="Full pipeline: beat-align + stretch + EQ, 320k")
    args = parser.parse_args()

    if args.preview and args.quality:
        parser.error("--preview and --quality are mutually exclusive")

    run(session_path=args.session, output_path=args.output,
        preview_mode=args.preview, quality_mode=args.quality)