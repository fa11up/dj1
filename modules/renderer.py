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
    from pedalboard import (
        Pedalboard, LowpassFilter, HighpassFilter, Gain,
        Reverb, Delay, Chorus, Phaser,
    )
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
BPM_BLEND_SAFETY_PCT = 0.05

# ── New-transition constants ───────────────────────────────────────────────────

# filter_sweep: spectral wipe — mirror filter sweeps hand frequency spectrum from
#               outgoing to incoming.  Together they always sum to ~full spectrum.
SWEEP_BARS      = 24      # total overlap bars
SWEEP_STEPS     = 32      # chunks for smooth filter curve (more = smoother)
SWEEP_LOW_HZ    = 40.0    # lowest frequency in sweep range
SWEEP_HIGH_HZ   = 18000.0 # highest frequency in sweep range

# reverb_wash: outgoing blooms into reverb; incoming fades in beneath the wash
REVERB_ROOM_SIZE    = 0.78
REVERB_DAMPING      = 0.40
REVERB_WET          = 0.50
REVERB_DRY          = 0.50
REVERB_WASH_BARS    = 16

# tension_drop: rising highpass sweeps bass away over N bars → hard cut to drop
TENSION_BUILD_BARS  = 16

# harmonic_blend: chorus-widened EQ phase swap for adjacent-key tracks
# Longer highs phase than long_blend gives more harmonic breathing room
HARMONIC_HIGHS_BARS_FAST = 16   # BPM ≥ 120
HARMONIC_HIGHS_BARS_SLOW = 24   # BPM < 120

# loop_roll: classic loop-roll — loop the last N bars while incoming fades in underneath
LOOP_ROLL_BARS = 4       # bars per loop cell (phrase-aligned)
LOOP_ROLL_REPS = 3       # how many times to repeat the cell during the fade-in

# loop_stutter: buffer/stutter roll — halving loop length (4→2→1 bars) before the drop
LOOP_STUTTER_BARS      = 4   # starting bar count; halves each stage
LOOP_STUTTER_REPS      = 2   # repetitions at each stage (4×2 + 2×2 + 1×2 bars total)
LOOP_STUTTER_XFADE_MS  = 20  # tiny crossfade between stutter stitches (tight = more click)


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

    for _ in range(LOOP_MAX_REPEATS):
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


# ── Effect helpers ────────────────────────────────────────────────────────────

def apply_reverb(seg, room_size=REVERB_ROOM_SIZE, damping=REVERB_DAMPING,
                 wet=REVERB_WET, dry=REVERB_DRY):
    """Apply reverb bloom via pedalboard. Falls back silently."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        board = Pedalboard([Reverb(room_size=room_size, damping=damping,
                                   wet_level=wet, dry_level=dry)])
        return float_to_seg(board(seg_to_float(seg), SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def apply_delay(seg, delay_seconds=0.25, feedback=0.35, mix=0.35):
    """Apply slapback/echo delay via pedalboard."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        board = Pedalboard([Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix)])
        return float_to_seg(board(seg_to_float(seg), SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def apply_chorus(seg, rate_hz=0.6, depth=0.15, mix=0.45):
    """Apply chorus for harmonic widening during blends."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        board = Pedalboard([Chorus(rate_hz=rate_hz, depth=depth, mix=mix)])
        return float_to_seg(board(seg_to_float(seg), SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def apply_phaser(seg, rate_hz=0.5, mix=0.40):
    """Apply phaser for psychoacoustic width."""
    if not PEDALBOARD_AVAILABLE:
        return seg
    try:
        board = Pedalboard([Phaser(rate_hz=rate_hz, mix=mix)])
        return float_to_seg(board(seg_to_float(seg), SAMPLE_RATE), seg.channels)
    except Exception:
        return seg


def pitch_shift_seg(seg, semitones):
    """
    Pitch-shift by semitones using pyrubberband. Clamped to ±6 st.
    Returns original segment if rubberband unavailable or shift < 0.05 st.
    """
    if not RUBBERBAND_AVAILABLE or abs(semitones) < 0.05:
        return seg
    semitones = max(-6.0, min(6.0, semitones))
    try:
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= (2 ** (seg.sample_width * 8 - 1))
        if seg.channels == 2:
            samples = samples.reshape(-1, 2)
            l = pyrb.pitch_shift(samples[:, 0], SAMPLE_RATE, semitones)
            r = pyrb.pitch_shift(samples[:, 1], SAMPLE_RATE, semitones)
            shifted = np.stack([l, r], axis=1).flatten()
        else:
            shifted = pyrb.pitch_shift(samples.flatten(), SAMPLE_RATE, semitones)
        shifted = np.clip(shifted, -1.0, 1.0)
        pcm = (shifted * (2 ** 15 - 1)).astype(np.int16)
        return AudioSegment(pcm.tobytes(), frame_rate=SAMPLE_RATE,
                            sample_width=2, channels=seg.channels)
    except Exception:
        return seg


def camelot_semitone_distance(cam_out, cam_in):
    """
    Return (semitones, direction) for pitch-shifting cam_in toward cam_out.

    Only the two close-key relationships are returned:
      Same position         → (0,  0)   no shift needed
      Relative major/minor  → (3, ±1)   A↔B same number = 3 semitones

    Adjacent number (perfect 5th = 7 semitones) → (None, 0): too audible to shift.
    """
    if not cam_out or not cam_in:
        return None, 0
    try:
        num_out, let_out = int(cam_out[:-1]), cam_out[-1].upper()
        num_in,  let_in  = int(cam_in[:-1]),  cam_in[-1].upper()
    except Exception:
        return None, 0
    if num_out == num_in and let_out == let_in:
        return 0, 0
    if num_out == num_in and let_out != let_in:
        # Relative pair: A→B = shift incoming up 3 st; B→A = shift down 3 st
        direction = 1 if let_in == 'B' else -1
        return 3, direction
    return None, 0


def sweep_filter(seg, start_hz, end_hz, mode, n_steps=SWEEP_STEPS):
    """
    Smooth filter sweep: divide the segment into n_steps chunks, applying a
    progressively changing cutoff between start_hz and end_hz (log interpolation).

    mode: 'lowpass'  — cutoff controls the top of the passband (outgoing: full→sub)
          'highpass' — cutoff controls the bottom of the passband (incoming: air→full)

    Falls back to the unprocessed segment if pedalboard is unavailable.
    """
    if not PEDALBOARD_AVAILABLE or not len(seg):
        return seg
    chunk_ms = len(seg) / n_steps
    if chunk_ms < 1:
        return seg
    result = AudioSegment.empty()
    for i in range(n_steps):
        t      = i / max(1, n_steps - 1)
        cutoff = start_hz * ((end_hz / start_hz) ** t)   # log interp
        cutoff = max(20.0, min(20000.0, cutoff))
        s_ms   = int(i * chunk_ms)
        e_ms   = int((i + 1) * chunk_ms) if i < n_steps - 1 else len(seg)
        chunk  = seg[s_ms:e_ms]
        if not chunk:
            continue
        try:
            audio = seg_to_float(chunk)
            Flt   = LowpassFilter if mode == 'lowpass' else HighpassFilter
            chunk = float_to_seg(
                Pedalboard([Flt(cutoff_frequency_hz=cutoff)])(audio, SAMPLE_RATE),
                chunk.channels,
            )
        except Exception:
            pass
        result += chunk
    return result


# ── New render functions ──────────────────────────────────────────────────────

def render_filter_sweep(mix, incoming_seg, slot_out, slot_in, strategy, chaos):
    """
    Spectral wipe: outgoing and incoming swap the frequency spectrum via mirror
    filter sweeps — at any moment they sum to approximately the full spectrum.

      Outgoing : lowpass  falling 18 kHz → 40 Hz  (full sound → sub-only → gone)
      Incoming : highpass falling 18 kHz → 40 Hz  (silent → full sound)

    BPM delta up to 8% is acceptable because the sweep masks tempo differences.
    Best for high-energy sections, momentum buildups, and climax transitions.
    """
    bpm        = slot_out.get("actual_bpm") or 120.0
    sweep_ms   = strategy["sweep_ms"]
    beat_times = slot_out.get("beat_times") or []
    sections   = slot_out.get("sections")   or []

    if strategy.get("do_stretch"):
        bpm_in = slot_in.get("actual_bpm")
        if bpm_in and bpm:
            incoming_seg = time_stretch_segment(incoming_seg, bpm_in, bpm)

    bd_ms = find_breakdown_from_sections(sections, track_duration_ms=len(mix))
    if bd_ms is None or len(mix) - bd_ms < sweep_ms:
        bd_ms = max(0, len(mix) - sweep_ms)
    if strategy.get("do_phrase_align") and beat_times:
        bd_ms = find_phrase_boundary(beat_times, bd_ms)
    sweep_start = max(0, min(bd_ms, len(mix) - sweep_ms))

    out_body  = mix[:sweep_start]
    out_sweep = mix[sweep_start : sweep_start + sweep_ms]
    in_sweep  = incoming_seg[:sweep_ms]
    in_body   = incoming_seg[sweep_ms:]

    if not PEDALBOARD_AVAILABLE:
        out_fade = out_sweep.fade_out(sweep_ms)
        in_fade  = in_sweep.fade_in(sweep_ms)
        return out_body + out_fade.overlay(in_fade) + in_body

    # Mirror pair — log sweep in opposite directions
    out_swept = sweep_filter(out_sweep, SWEEP_HIGH_HZ, SWEEP_LOW_HZ, 'lowpass')
    in_swept  = sweep_filter(in_sweep,  SWEEP_HIGH_HZ, SWEEP_LOW_HZ, 'highpass')
    # Brief volume envelope at edges prevents digital click
    edge_ms   = min(1500, sweep_ms // 8)
    out_swept = out_swept.fade_out(edge_ms)
    in_swept  = in_swept.fade_in(edge_ms)
    return out_body + out_swept.overlay(in_swept) + in_body


def render_reverb_wash(mix, incoming_seg, slot_out, slot_in, strategy, chaos):
    """
    Reverb dissolution: at the breakdown, outgoing blooms into heavy reverb.
    Incoming fades in beneath the wash, creating an atmospheric bridge.

    The reverb tail extends over two halves — moderate then heavy — so the
    bloom grows naturally rather than snapping to maximum wet at the splice.
    Works for any BPM delta because the reverb masks tempo differences.
    Best for mood shifts, key changes, and atmospheric / melancholic passages.
    """
    bpm        = slot_out.get("actual_bpm") or 120.0
    wash_ms    = strategy["wash_ms"]
    beat_times = slot_out.get("beat_times") or []
    sections   = slot_out.get("sections")   or []

    if strategy.get("do_stretch"):
        bpm_in = slot_in.get("actual_bpm")
        if bpm_in and bpm:
            incoming_seg = time_stretch_segment(incoming_seg, bpm_in, bpm)

    bd_ms = find_breakdown_from_sections(sections, track_duration_ms=len(mix))
    if bd_ms is None or len(mix) - bd_ms < wash_ms:
        bd_ms = max(0, len(mix) - wash_ms)
    if strategy.get("do_phrase_align") and beat_times:
        bd_ms = find_phrase_boundary(beat_times, bd_ms)
    wash_start = max(0, min(bd_ms, len(mix) - wash_ms))

    out_body = mix[:wash_start]
    out_tail = mix[wash_start : wash_start + wash_ms]
    in_body  = incoming_seg

    if not PEDALBOARD_AVAILABLE:
        fade_ms      = min(wash_ms, len(out_tail), len(in_body))
        out_tail_fad = out_tail[:fade_ms].fade_out(fade_ms)
        in_head_fad  = in_body[:fade_ms].fade_in(fade_ms)
        return out_body + out_tail_fad.overlay(in_head_fad) + in_body[fade_ms:]

    # Reverb grows over two halves (moderate → heavy) then fades out
    half          = max(1, len(out_tail) // 2)
    out_first     = apply_reverb(out_tail[:half], room_size=0.55, wet=0.30, dry=0.70)
    out_second    = apply_reverb(out_tail[half:], room_size=REVERB_ROOM_SIZE,
                                 wet=REVERB_WET, dry=REVERB_DRY)
    out_wash      = (out_first + out_second).fade_out(wash_ms)

    # Incoming fades in over the full wash window
    in_fade_ms    = min(wash_ms, len(in_body))
    in_head       = in_body[:in_fade_ms].fade_in(in_fade_ms)
    in_rest       = in_body[in_fade_ms:]

    # Pad in_head to match out_wash length for clean overlay
    pad = len(out_wash) - len(in_head)
    if pad > 0:
        in_head = in_head + AudioSegment.silent(duration=pad,
                                                frame_rate=SAMPLE_RATE).set_channels(CHANNELS)
    else:
        in_head = in_head[:len(out_wash)]

    return out_body + out_wash.overlay(in_head) + in_rest


def render_harmonic_blend(mix, incoming_seg, slot_out, slot_in, strategy, chaos):
    """
    Harmonically conscious phase-swap blend.

    Same 4-phase structure as long_blend but with two enhancements:
      1. Chorus widening on the overlap phases — blurs key differences,
         widens stereo image, makes the blend feel larger than a simple EQ swap.
      2. Pitch correction: if the tracks are a relative major/minor pair
         (same Camelot number, different letter = 3 semitones), the incoming's
         blend window is pitch-shifted to match the outgoing's key. After the
         overlap resolves, the incoming plays at its natural pitch.

    Select for: Camelot-adjacent tracks, BPM delta < 6%, peak-of-set moments.
    """
    bpm        = slot_out.get("actual_bpm") or 120.0
    highs_ms   = strategy["highs_ms"]
    swap_ms    = strategy["swap_ms"]
    outro_ms   = strategy["outro_ms"]
    beat_times = slot_out.get("beat_times") or []
    sections   = slot_out.get("sections")   or []

    centroid = slot_out.get("spectral_centroid_mean")
    bass_hz  = compute_crossover_hz(centroid)
    highs_hz = max(400.0, min(1200.0, bass_hz * 4))

    # Pitch-correct incoming blend window for relative major/minor pairs
    cam_out  = slot_out.get("camelot")
    cam_in   = slot_in.get("camelot")
    semitones, direction = camelot_semitone_distance(cam_out, cam_in)
    if semitones and semitones > 0 and RUBBERBAND_AVAILABLE:
        blend_len    = highs_ms + swap_ms
        in_blend     = pitch_shift_seg(incoming_seg[:blend_len], -semitones * direction)
        incoming_seg = in_blend + incoming_seg[blend_len:]

    if strategy.get("do_stretch"):
        bpm_in = slot_in.get("actual_bpm")
        if bpm_in and bpm:
            incoming_seg = time_stretch_segment(incoming_seg, bpm_in, bpm)

    total_ms = highs_ms + swap_ms + outro_ms
    bd_ms = find_breakdown_from_sections(sections, track_duration_ms=len(mix))
    if bd_ms is None or len(mix) - bd_ms < total_ms:
        bd_ms = max(0, len(mix) - total_ms)
    if strategy.get("do_phrase_align") and beat_times:
        bd_ms = find_phrase_boundary(beat_times, bd_ms)
    p2_start = max(0, min(bd_ms, len(mix) - total_ms))
    p3_start = p2_start + highs_ms
    p4_start = p3_start + swap_ms

    out_body = mix[:p2_start]
    out_p2   = mix[p2_start:p3_start]
    out_p3   = mix[p3_start:p4_start]
    out_p4   = mix[p4_start:]

    inc_p2   = incoming_seg[:highs_ms]
    inc_p3   = incoming_seg[highs_ms : highs_ms + swap_ms]
    inc_p4   = incoming_seg[highs_ms + swap_ms : highs_ms + swap_ms + outro_ms]
    inc_body = incoming_seg[highs_ms + swap_ms + outro_ms:]

    # Phase 2: incoming highs + chorus widening over outgoing full
    inc_p2_hi  = apply_highpass_adaptive(inc_p2, highs_hz, 0.0)
    inc_p2_ch  = apply_chorus(inc_p2_hi)
    out_p2_ch  = apply_chorus(out_p2)
    overlap_p2 = out_p2_ch.overlay(inc_p2_ch)

    # Phase 3: bass swap — tight, no chorus (decisiveness)
    out_p3_nb  = apply_highpass_adaptive(out_p3, bass_hz, 0.0)
    overlap_p3 = out_p3_nb.overlay(inc_p3)

    # Phase 4: outgoing highs fade with phaser shimmer, incoming takes over
    out_p4_hi  = apply_highpass_adaptive(out_p4, highs_hz, 0.0).fade_out(outro_ms)
    out_p4_ph  = apply_phaser(out_p4_hi)
    overlap_p4 = inc_p4.overlay(out_p4_ph)

    return out_body + overlap_p2 + overlap_p3 + overlap_p4 + inc_body


def render_tension_drop(mix, incoming_seg, slot_out, slot_in, strategy, chaos):
    """
    Build-and-release tension: a rising highpass filter sweeps the bass out of
    the outgoing track over N bars, creating mounting anticipation. At the phrase
    boundary, the mix hard-cuts to the incoming track's drop — no overlap.

    The cut IS the release. Works best when incoming starts on a hard downbeat.
    A brief (2-bar) fade-in on the incoming head prevents a digital click at
    the splice, while preserving the punch of the drop.

    Select for: high-energy climax moments, genre pivots, incoming tracks that
    open on a big drop. Chaos factor 0.5+. Any BPM delta is acceptable.
    """
    bpm        = slot_out.get("actual_bpm") or 120.0
    build_ms   = strategy["build_ms"]
    beat_times = slot_out.get("beat_times") or []

    build_start = max(0, len(mix) - build_ms)
    if strategy.get("do_phrase_align") and beat_times:
        build_start = find_phrase_boundary(beat_times, build_start)
    build_start = max(0, min(build_start, len(mix) - build_ms))

    out_body  = mix[:build_start]
    out_build = mix[build_start:]

    if not PEDALBOARD_AVAILABLE:
        return out_body + out_build.fade_out(len(out_build)) + incoming_seg

    # Rising highpass: track's natural bass crossover → 8 kHz (full tension)
    centroid    = slot_out.get("spectral_centroid_mean")
    bass_hz     = compute_crossover_hz(centroid)
    out_tension = sweep_filter(out_build, bass_hz, 8000.0, 'highpass', n_steps=SWEEP_STEPS)

    # Slight gain bump during the build (excitement, compensates for lost bass energy)
    try:
        audio       = seg_to_float(out_tension)
        out_tension = float_to_seg(
            Pedalboard([Gain(gain_db=2.5)])(audio, SAMPLE_RATE),
            out_tension.channels,
        )
    except Exception:
        pass

    # Hard cut: brief 2-bar fade-in on incoming to avoid click, preserves punch
    in_fade_ms = min(bars_to_ms(2, bpm), len(incoming_seg) // 4)
    in_head    = incoming_seg[:in_fade_ms].fade_in(in_fade_ms)
    in_body    = incoming_seg[in_fade_ms:]
    return out_body + out_tension + in_head + in_body


def render_loop_roll(mix, incoming_seg, slot_out, slot_in, strategy, chaos):
    """
    Classic DJ loop roll — captures the last N bars of the outgoing track in a
    repeating loop, then fades the incoming track up underneath it over successive
    repetitions. The loop and incoming trade volume smoothly so the combined energy
    stays constant throughout the handoff.

    Timeline:
      [out_body] [loop_cell × n_reps  ↕ incoming_fade_in] [incoming_body]

    The loop cell is phrase-aligned so the repeat seam lands on a clean downbeat.
    A tiny crossfade (50 ms) at each loop join prevents clicks.
    In the second half of the overlap, the outgoing loop loses its bass via a
    gentle highpass ramp, creating headroom for the incoming kick/sub.

    Select for: smooth energy handoffs between genre-compatible tracks. The loop
    rewards the crowd with a familiar hook before revealing the new track. Works
    best at similar BPM (< 6% delta). Camelot rules 1–3.
    """
    bpm        = slot_out.get("actual_bpm") or 120.0
    beat_times = slot_out.get("beat_times") or []
    loop_bars  = strategy["loop_bars"]
    n_reps     = strategy["n_reps"]

    loop_ms = bars_to_ms(loop_bars, bpm)

    # Need at least 1.5× the loop length to have a safe out_body before it
    if len(mix) < loop_ms * 1.5 or len(incoming_seg) < 2000:
        fade_ms = min(bars_to_ms(4, bpm), len(mix) - 1000, len(incoming_seg) - 1000)
        fade_ms = max(0, fade_ms)
        if fade_ms <= 0:
            return mix + incoming_seg
        return mix[:-fade_ms] + mix[-fade_ms:].fade_out(fade_ms).overlay(
            incoming_seg[:fade_ms].fade_in(fade_ms)
        ) + incoming_seg[fade_ms:]

    # Phrase-align the loop start point
    raw_start = len(mix) - loop_ms
    if strategy.get("do_phrase_align") and beat_times:
        loop_start = find_phrase_boundary(beat_times, raw_start)
    else:
        loop_start = raw_start
    loop_start = max(0, min(loop_start, len(mix) - loop_ms))

    loop_cell = mix[loop_start:]  # from the loop boundary to end of mix
    out_body  = mix[:loop_start]

    # Build n_reps repetitions with tiny crossfade joins
    xfade_ms   = min(50, len(loop_cell) // 4)
    loop_audio = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    for _ in range(n_reps):
        if len(loop_audio) == 0:
            loop_audio = loop_cell
        elif xfade_ms > 0 and len(loop_audio) > xfade_ms and len(loop_cell) > xfade_ms:
            loop_audio = loop_audio.append(loop_cell, crossfade=xfade_ms)
        else:
            loop_audio = loop_audio + loop_cell

    total_ms  = len(loop_audio)
    inc_head  = incoming_seg[:total_ms]
    inc_body  = incoming_seg[total_ms:]

    # In the second half of the loop, apply a gentle highpass to the loop to
    # clear bass headroom for the incoming kick — mirrors how a real DJ EQs
    half = total_ms // 2
    if PEDALBOARD_AVAILABLE:
        centroid   = slot_out.get("spectral_centroid_mean")
        bass_hz    = compute_crossover_hz(centroid)
        loop_tail  = apply_highpass_adaptive(loop_audio[half:], bass_hz, 0.0)
        loop_audio = loop_audio[:half] + loop_tail

    loop_out = loop_audio.fade_out(total_ms)
    inc_in   = inc_head.fade_in(total_ms)
    overlap  = loop_out.overlay(inc_in)
    return out_body + overlap + inc_body


def render_loop_stutter(mix, incoming_seg, slot_out, slot_in, strategy, chaos):
    """
    Buffer stutter roll — the outgoing track is caught in a progressively
    shorter loop: starting at N bars, halving each stage (N → N/2 → N/4 → 1),
    with LOOP_STUTTER_REPS repetitions at each stage. This creates the classic
    CDJ "buffer roll" stuttering effect that builds mounting tension before the
    hard cut to the incoming drop.

    Timeline:
      [out_body] [4×2][4×2][2×2][2×2][1×2][1×2] | cut | [incoming fade-in]

    The cut IS the release — the stutter raises energy then drops it clean.
    A 1-bar fade-in on the incoming head prevents a digital click at the splice
    while preserving the punch of the drop.

    BPM delta: any — no overlap. Chaos factor: 0.4+.
    Select for: high-energy climax moments, peak-of-set tension, drops.
    Avoid on tracks that start with a long intro (needs a hard downbeat).
    """
    bpm          = slot_out.get("actual_bpm") or 120.0
    beat_times   = slot_out.get("beat_times") or []
    start_bars   = strategy["stutter_bars"]
    reps_per_len = strategy["reps_per_len"]

    loop_ms   = bars_to_ms(start_bars, bpm)
    raw_start = max(0, len(mix) - loop_ms)

    # Phrase-align: stutter starts on a clean phrase boundary
    if strategy.get("do_phrase_align") and beat_times:
        stutter_start = find_phrase_boundary(beat_times, raw_start)
    else:
        stutter_start = raw_start
    stutter_start = max(0, min(stutter_start, len(mix) - loop_ms))

    out_body      = mix[:stutter_start]
    stutter_audio = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)

    bars = start_bars
    xfade_ms = LOOP_STUTTER_XFADE_MS
    while bars >= 1:
        region_ms = bars_to_ms(bars, bpm)
        region    = mix[stutter_start : stutter_start + int(region_ms)]
        if len(region) < region_ms * 0.5:
            break
        for _ in range(reps_per_len):
            if len(stutter_audio) > xfade_ms and len(region) > xfade_ms:
                stutter_audio = stutter_audio.append(region, crossfade=xfade_ms)
            else:
                stutter_audio = stutter_audio + region
        bars = bars // 2

    # Hard drop: brief 1-bar fade-in on incoming preserves the punch
    in_fade_ms = min(bars_to_ms(1, bpm), len(incoming_seg) // 4)
    in_head    = incoming_seg[:in_fade_ms].fade_in(in_fade_ms)
    in_body    = incoming_seg[in_fade_ms:]
    return out_body + stutter_audio + in_head + in_body


# ── Transition strategy ──────────────────────────────────────────────────────

def transition_strategy(hint, chaos, preview_mode, bpm=None, quality_mode=False):
    """
    Return a strategy dict describing how to execute the transition.

    CHANGED: quality_mode now controls stretch/phrase-align/EQ independently
    of chaos. Chaos only affects musical choices (swap bar count), NOT
    pipeline quality.

    For long_blend:  staged 3-phase DJ transition (the real deal)
    For crossfade:   simple overlapping volume fade with EQ
    For quick_fade:  short volume fade, no EQ
    For hard_cut:    sequential tail-fade + head-fade, no overlap

    Returns:
        {
          "method":          "staged" | "crossfade" | "quick_fade" | "hard_cut"
          "do_stretch":      bool
          "do_phrase_align": bool
          "do_eq":           bool       (for crossfade)
          # staged only:
          "highs_ms":        int   phase 2 duration
          "swap_ms":         int   phase 3 duration
          "outro_ms":        int   phase 4 duration
          # simple only:
          "fade_ms":         int
        }
    """
    bpm = bpm or 120.0

    # ── Quality gates — chaos no longer degrades these ────────────────────────
    # In quality mode: always stretch, always phrase-align, always EQ
    # In default mode: stretch + EQ if libraries available, phrase-align yes
    # In preview mode: no stretch, no phrase-align, shorter phases
    if preview_mode:
        do_stretch      = False
        do_phrase_align = False
        do_eq           = False
    elif quality_mode:
        do_stretch      = RUBBERBAND_AVAILABLE
        do_phrase_align = True
        do_eq           = PEDALBOARD_AVAILABLE
    else:
        # Default mode — use libs if available, no chaos gate
        do_stretch      = RUBBERBAND_AVAILABLE
        do_phrase_align = True
        do_eq           = PEDALBOARD_AVAILABLE

    if hint == "long_blend":
        highs_bars = PHASE_HIGHS_BARS_FAST if bpm >= 120 else PHASE_HIGHS_BARS_SLOW
        # Chaos affects swap bar count — a musical choice:
        # low chaos = tight 2-bar swap (decisive), high chaos = loose 4-bar (messy but fun)
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
        fade_ms = CROSSFADE_MS // 2 if preview_mode else CROSSFADE_MS
        return {
            "method":          "crossfade",
            "do_stretch":      do_stretch,
            "do_phrase_align": False,
            "fade_ms":         fade_ms,
            "do_eq":           do_eq,
        }

    if hint == "quick_fade":
        return {
            "method":          "quick_fade",
            "do_stretch":      False,
            "do_phrase_align": False,
            "fade_ms":         QUICK_FADE_MS,
            "do_eq":           False,
        }

    if hint == "filter_sweep":
        sweep_bars = max(8, SWEEP_BARS // 2) if preview_mode else SWEEP_BARS
        return {
            "method":          "filter_sweep",
            "do_stretch":      do_stretch,
            "do_phrase_align": do_phrase_align,
            "sweep_ms":        bars_to_ms(sweep_bars, bpm),
        }

    if hint == "reverb_wash":
        wash_bars = max(6, REVERB_WASH_BARS // 2) if preview_mode else REVERB_WASH_BARS
        return {
            "method":          "reverb_wash",
            "do_stretch":      do_stretch,
            "do_phrase_align": do_phrase_align,
            "wash_ms":         bars_to_ms(wash_bars, bpm),
        }

    if hint == "harmonic_blend":
        highs_bars = HARMONIC_HIGHS_BARS_FAST if bpm >= 120 else HARMONIC_HIGHS_BARS_SLOW
        swap_bars  = PHASE_SWAP_BARS_TIGHT if chaos < 0.5 else PHASE_SWAP_BARS_LOOSE
        outro_bars = PHASE_OUTRO_BARS
        if preview_mode:
            highs_bars = max(4, highs_bars // 2)
            swap_bars  = 1
            outro_bars = max(4, outro_bars // 2)
        return {
            "method":          "harmonic_blend",
            "do_stretch":      do_stretch,
            "do_phrase_align": do_phrase_align,
            "highs_ms":        bars_to_ms(highs_bars, bpm),
            "swap_ms":         bars_to_ms(swap_bars,  bpm),
            "outro_ms":        bars_to_ms(outro_bars, bpm),
        }

    if hint == "tension_drop":
        build_bars = max(8, TENSION_BUILD_BARS // 2) if preview_mode else TENSION_BUILD_BARS
        return {
            "method":          "tension_drop",
            "do_stretch":      False,   # intentional — tempo clash adds tension
            "do_phrase_align": do_phrase_align,
            "build_ms":        bars_to_ms(build_bars, bpm),
        }

    if hint == "loop_roll":
        loop_bars = LOOP_ROLL_BARS
        n_reps    = max(2, LOOP_ROLL_REPS // 2) if preview_mode else LOOP_ROLL_REPS
        return {
            "method":          "loop_roll",
            "do_stretch":      False,   # loop is same-tempo content — no stretch needed
            "do_phrase_align": do_phrase_align,
            "loop_bars":       loop_bars,
            "n_reps":          n_reps,
        }

    if hint == "loop_stutter":
        start_bars   = max(2, LOOP_STUTTER_BARS // 2) if preview_mode else LOOP_STUTTER_BARS
        reps_per_len = 1 if preview_mode else LOOP_STUTTER_REPS
        return {
            "method":          "loop_stutter",
            "do_stretch":      False,   # no overlap — BPM delta irrelevant
            "do_phrase_align": do_phrase_align,
            "stutter_bars":    start_bars,
            "reps_per_len":    reps_per_len,
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

    CHANGED: EQ filters always use steep/cascaded mode. The old chaos-gated
    shelf mode was causing bass bleed and muddiness in transitions.
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
            phrase_ms = find_phrase_boundary(beat_times, breakdown_ms)
            phrase_ms = min(phrase_ms, int(len(outgoing) * 0.82))
            total_available = len(outgoing) - phrase_ms
            if total_available >= total_ms * 0.5:
                if total_available < total_ms:
                    scale    = total_available / total_ms
                    highs_ms = int(highs_ms * scale)
                    swap_ms  = max(int(swap_ms * scale), 500)
                    outro_ms = int(outro_ms * scale)
                    total_ms = highs_ms + swap_ms + outro_ms

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
    swap_ms  = max(500, min(swap_ms,  total_ms // 4))
    highs_ms = max(500, min(highs_ms, (total_ms - swap_ms) // 2))
    outro_ms = max(500, total_ms - highs_ms - swap_ms)
    # Re-clamp total to actual sum of phases
    total_ms = highs_ms + swap_ms + outro_ms
    total_ms = min(total_ms, max_out, max_in)
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

    loop_bars = LOOP_BARS_LONG if bpm < 120 else LOOP_BARS_DEFAULT
    loop_ms   = bars_to_ms(loop_bars, bpm)

    # Only loop if the tail genuinely lacks enough content for the blend.
    # The previous `< total_ms * 1.2` was always True (tail_available_ms is
    # clamped to total_ms), causing looping on every transition.
    # Also: beat_times are in track-local coordinates, not mix coordinates.
    # Passing None prevents loop_segment from snapping to a beat in the wrong
    # timeline (which previously copied 4+ minutes of the accumulated mix).
    needs_loop = tail_available_ms < total_ms

    if needs_loop and loop_ms > 0 and len(outgoing) > loop_ms * 1.5:
        loop_audio, loop_point_ms = loop_segment(
            outgoing, loop_bars, bpm, beat_times=None
        )
        if len(loop_audio) > 0:
            outgoing = outgoing[:loop_point_ms] + loop_audio

    # ── Slice regions ──────────────────────────────────────────────────────────
    p2_start = max(0, len(outgoing) - (highs_ms + swap_ms + outro_ms))
    p3_start = p2_start + highs_ms
    p4_start = p3_start + swap_ms

    out_body  = outgoing[:p2_start]
    out_p2    = outgoing[p2_start:p3_start]
    out_p3    = outgoing[p3_start:p4_start]
    out_p4    = outgoing[p4_start:]

    inc_p2    = incoming[:highs_ms]
    inc_p3    = incoming[highs_ms:highs_ms+swap_ms]
    inc_p4    = incoming[highs_ms+swap_ms:highs_ms+swap_ms+outro_ms]
    inc_body  = incoming[highs_ms+swap_ms+outro_ms:]

    # ── Phase 2: incoming highs only over outgoing full range ─────────────────
    # CHANGED: always use chaos=0.0 for steep filter — no shelf mode
    inc_p2_hi  = apply_highpass_adaptive(inc_p2, highs_hz, 0.0)
    overlap_p2 = out_p2.overlay(inc_p2_hi)
    # ── Phase 3: bass swap — decisive, tight window ───────────────────────────
    # CHANGED: always steep EQ
    out_p3_nobase = apply_highpass_adaptive(out_p3, bass_hz, 0.0)
    inc_p3_full   = inc_p3
    overlap_p3    = out_p3_nobase.overlay(inc_p3_full)
    # ── Phase 4: outgoing highs fade out, incoming full range ─────────────────
    # CHANGED: always steep EQ
    out_p4_hi  = apply_highpass_adaptive(out_p4, highs_hz, 0.0).fade_out(outro_ms)
    overlap_p4 = inc_p4.overlay(out_p4_hi)
    return out_body + overlap_p2 + overlap_p3 + overlap_p4 + inc_body


def render_transition(outgoing, incoming, prev_seg_raw, slot_out, slot_in,
                      chaos, preview_mode, quality_mode=False, stems_cache=None):
    """
    Entry point for all transitions.
    Determines strategy via energy + hint + chaos, dispatches to the
    appropriate render function.

    outgoing:     accumulated mix buffer (grows with each track)
    incoming:     the next individual track segment
    prev_seg_raw: the previous individual track segment (same audio as the
                  tail of `outgoing`, but in its own timeline — used for
                  stem-based transitions where we need track-local coordinates)
    stems_cache:  dict {track_id: {stem_name: path}} for stem_blend transitions
    """
    if stems_cache is None:
        stems_cache = {}

    hint     = slot_in.get("transition_hint", "crossfade")
    hint     = energy_aware_hint(hint, outgoing, incoming)
    bpm_out  = slot_out.get("actual_bpm")
    bpm_in   = slot_in.get("actual_bpm")
    bpm      = bpm_out or bpm_in or 120.0
    strategy = transition_strategy(hint, chaos, preview_mode, bpm=bpm,
                                   quality_mode=quality_mode)
    method   = strategy["method"]
    # ── BPM SAFETY GATE ──────────────────────────────────────────────────────
    # If BPM delta > 5% and we're NOT stretching, overlapping blends train-wreck.
    # Exceptions: reverb_wash and tension_drop don't overlap, so BPM delta is fine.
    # filter_sweep's spectral wipe also masks minor tempo clash at <8% delta.
    overlap_methods = ("staged", "crossfade", "harmonic_blend", "filter_sweep", "loop_roll")
    if method in overlap_methods and bpm_out and bpm_in:
        bpm_delta_pct = abs(bpm_out - bpm_in) / max(bpm_out, bpm_in)
        if bpm_delta_pct > BPM_BLEND_SAFETY_PCT and not strategy.get("do_stretch"):
            strategy = transition_strategy("hard_cut", chaos, preview_mode, bpm=bpm,
                                           quality_mode=quality_mode)
            method   = "hard_cut"

    # ── Stem blend — surgical stem-based transition ───────────────────────────
    if hint == "stem_blend" and prev_seg_raw is not None:
        stem_strategy = transition_strategy("long_blend", chaos, preview_mode,
                                            bpm=bpm, quality_mode=quality_mode)
        return render_stem_blend(outgoing, incoming, prev_seg_raw, slot_out, slot_in,
                                 stem_strategy, chaos, stems_cache)

    # ── New transitions ───────────────────────────────────────────────────────
    if method == "filter_sweep":
        return render_filter_sweep(outgoing, incoming, slot_out, slot_in, strategy, chaos)
    if method == "reverb_wash":
        return render_reverb_wash(outgoing, incoming, slot_out, slot_in, strategy, chaos)
    if method == "harmonic_blend":
        return render_harmonic_blend(outgoing, incoming, slot_out, slot_in, strategy, chaos)
    if method == "tension_drop":
        return render_tension_drop(outgoing, incoming, slot_out, slot_in, strategy, chaos)
    if method == "loop_roll":
        return render_loop_roll(outgoing, incoming, slot_out, slot_in, strategy, chaos)
    if method == "loop_stutter":
        return render_loop_stutter(outgoing, incoming, slot_out, slot_in, strategy, chaos)

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
        incoming = _stretch_incoming(incoming, bpm_in, bpm_out, fade_ms)
    if strategy.get("do_eq") and PEDALBOARD_AVAILABLE:
        centroid     = slot_out.get("spectral_centroid_mean")
        bass_hz      = compute_crossover_hz(centroid)
        out_tail_raw = outgoing[-fade_ms:]
        in_head_raw  = incoming[:fade_ms]
        # CHANGED: always use steep EQ (chaos=0.0) — gentle shelf causes bleed
        out_tail_eq  = apply_highpass_adaptive(out_tail_raw, bass_hz, 0.0)
        in_head_eq   = apply_highpass_adaptive(in_head_raw,  bass_hz, 0.0)
        outgoing  = outgoing[:-fade_ms] + out_tail_eq
        incoming  = in_head_eq + incoming[fade_ms:]
    out_body = outgoing[:-fade_ms]
    out_tail = outgoing[-fade_ms:].fade_out(fade_ms)
    in_head  = incoming[:fade_ms].fade_in(fade_ms)
    in_body  = incoming[fade_ms:]
    overlap  = out_tail.overlay(in_head)
    return out_body + overlap + in_body


# ── Stem blend transition ─────────────────────────────────────────────────────

def render_stem_blend(mix, incoming_seg, prev_seg_raw, slot_out, slot_in,
                      strategy, chaos, stems_cache):
    """
    Surgical stem-based DJ transition.

    Instead of EQ filter approximations (highpass/lowpass), uses actual demucs
    stems for the outgoing and incoming tracks to perform a precise mix:

      Phase 2 — incoming melody/vocals (other+vocals) enter over outgoing full
      Phase 3 — bass swap: outgoing bass exits, incoming bass enters
      Phase 4 — outgoing drums+highs fade, incoming plays full range

    Falls back to render_staged() if stems are unavailable for either track.

    prev_seg_raw: the INDIVIDUAL outgoing track (not accumulated mix) — needed
                  to load its stems and correctly time the transition window.
    """
    from modules.stems import load_stems

    out_id = slot_out.get("track_id", "")
    in_id  = slot_in.get("track_id",  "")

    out_stem_paths = stems_cache.get(out_id)
    in_stem_paths  = stems_cache.get(in_id)

    if not out_stem_paths or not in_stem_paths:
        # Stems unavailable — degrade gracefully to standard staged blend
        print(f"\n  ⚠  stem_blend: stems missing, falling back to long_blend")
        return render_staged(mix, incoming_seg, slot_out, slot_in, strategy, chaos=chaos)

    out_stems = load_stems(out_stem_paths)
    in_stems  = load_stems(in_stem_paths)

    if not out_stems or not in_stems:
        print(f"\n  ⚠  stem_blend: stem load failed, falling back to long_blend")
        return render_staged(mix, incoming_seg, slot_out, slot_in, strategy, chaos=chaos)

    # ── Phase timing (same as staged) ────────────────────────────────────────
    bpm      = slot_out.get("actual_bpm") or 120.0
    highs_ms = strategy["highs_ms"]
    swap_ms  = strategy["swap_ms"]
    outro_ms = strategy["outro_ms"]
    total_ms = highs_ms + swap_ms + outro_ms

    # Clamp to available audio
    max_out  = max(4000, len(prev_seg_raw) - 2000)
    max_in   = max(4000, len(incoming_seg) - 2000)
    total_ms = min(total_ms, max_out, max_in)
    swap_ms  = max(500, min(swap_ms,  total_ms // 4))
    highs_ms = max(500, min(highs_ms, (total_ms - swap_ms) // 2))
    outro_ms = max(500, total_ms - highs_ms - swap_ms)

    # Time-stretch if enabled
    if strategy.get("do_stretch"):
        bpm_in  = slot_in.get("actual_bpm")
        bpm_out = slot_out.get("actual_bpm")
        incoming_seg = _stretch_incoming(incoming_seg, bpm_in, bpm_out, total_ms)
        for name in ("drums", "bass", "other", "vocals"):
            if name in in_stems:
                in_stems[name] = _stretch_incoming(in_stems[name], bpm_in, bpm_out, total_ms)

    # ── Slice stems to their respective phase windows ─────────────────────────
    # Outgoing stems: take the last `total_ms` from prev_seg_raw timeline
    out_total_ms = len(prev_seg_raw)
    out_t_start  = max(0, out_total_ms - total_ms)

    out_p2_drums = out_stems["drums"][out_t_start : out_t_start + highs_ms]
    out_p2_bass  = out_stems["bass"] [out_t_start : out_t_start + highs_ms]
    out_p2_mel   = out_stems["other"][out_t_start : out_t_start + highs_ms]

    out_p3_start = out_t_start + highs_ms
    out_p3_drums = out_stems["drums"][out_p3_start : out_p3_start + swap_ms]
    out_p3_bass  = out_stems["bass"] [out_p3_start : out_p3_start + swap_ms]
    out_p3_mel   = out_stems["other"][out_p3_start : out_p3_start + swap_ms]

    out_p4_start = out_p3_start + swap_ms
    out_p4_drums = out_stems["drums"][out_p4_start : out_p4_start + outro_ms]
    out_p4_mel   = out_stems["other"][out_p4_start : out_p4_start + outro_ms]

    # Incoming stems: take the first `total_ms`
    in_p2_mel    = in_stems["other"] [:highs_ms]
    in_p3_bass   = in_stems["bass"]  [highs_ms : highs_ms + swap_ms]
    in_p3_drums  = in_stems["drums"] [highs_ms : highs_ms + swap_ms]
    in_p3_mel    = in_stems["other"] [highs_ms : highs_ms + swap_ms]
    in_p4_mel    = in_stems["other"] [highs_ms + swap_ms : highs_ms + swap_ms + outro_ms]
    in_body      = incoming_seg[total_ms:]

    # ── Mix each phase ────────────────────────────────────────────────────────
    # Phase 2: outgoing full (drums+bass+melody) + incoming melody only
    out_p2_full = out_p2_drums.overlay(out_p2_bass).overlay(out_p2_mel)
    overlap_p2  = out_p2_full.overlay(in_p2_mel)

    # Phase 3: bass swap — outgoing drums+melody, incoming bass+drums
    out_p3_no_bass = out_p3_drums.overlay(out_p3_mel)
    in_p3_full     = in_p3_drums.overlay(in_p3_bass).overlay(in_p3_mel)
    overlap_p3     = out_p3_no_bass.overlay(in_p3_full)

    # Phase 4: outgoing melody fades, incoming full plays
    out_p4_fade  = out_p4_drums.overlay(out_p4_mel).fade_out(outro_ms)
    in_p4_full   = incoming_seg[highs_ms + swap_ms : highs_ms + swap_ms + outro_ms]
    overlap_p4   = in_p4_full.overlay(out_p4_fade)

    # Mix body: everything in accumulated `mix` up to transition start
    mix_body = mix[: len(mix) - total_ms] if len(mix) > total_ms else mix[:0]

    return mix_body + overlap_p2 + overlap_p3 + overlap_p4 + in_body


def render_session(setlist, chaos, preview_mode, progress_cb=None, quality_mode=False,
                   stems_cache=None):
    """
    Render all slots into a single AudioSegment.
    progress_cb(position, total, filename) called per track if provided.

    stems_cache: optional dict {track_id: {drums/bass/other/vocals: path}}
                 passed through to render_transition for stem_blend transitions.
    """
    if stems_cache is None:
        stems_cache = {}

    mix       = None
    prev_seg  = None
    prev_slot = None
    total     = len(setlist)

    for i, slot in enumerate(setlist):
        if progress_cb:
            progress_cb(i, total, slot["filename"])
        seg = load_track(slot["filepath"], preview_mode)
        if seg is None:
            print(f"  ⚠  Skipping {slot['filename']} (could not load)")
            continue
        seg = trim_to_cues(seg, slot.get("intro_end_sec"), slot.get("outro_start_sec"))
        seg = pydub_normalize(seg)

        if mix is None:
            mix = seg
        else:
            mix = render_transition(
                mix, seg, prev_seg, prev_slot, slot,
                chaos, preview_mode,
                quality_mode=quality_mode,
                stems_cache=stems_cache,
            )
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
    session_path    = "./data/session.json",
    output_path     = None,
    preview_mode    = False,
    quality_mode    = False,
    stems_cache_dir = "./data/.stems_cache",
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

    mode_str = "PREVIEW (fast)" if preview_mode else "QUALITY (full pipeline)" if quality_mode else "DEFAULT"

    print(f"\n  Tracks:       {len(setlist)}")
    print(f"  Duration:     ~{duration} min")
    print(f"  Chaos:        {chaos}")
    print(f"  Mode:         {mode_str}")
    print(f"  Output:       {output_path}")
    print(f"\n  Libraries:    rubberband={'✓' if RUBBERBAND_AVAILABLE else '✗ (no stretch)'}  "
          f"pedalboard={'✓' if PEDALBOARD_AVAILABLE else '✗ (no EQ)'}  "
          f"librosa={'✓' if LIBROSA_AVAILABLE else '✗ (no beat-align)'}")

    # Warn about degraded pipeline
    if quality_mode:
        if not RUBBERBAND_AVAILABLE:
            print(f"\n  ⚠  pyrubberband not available — time-stretching disabled even in quality mode")
        if not PEDALBOARD_AVAILABLE:
            print(f"  ⚠  pedalboard not available — EQ swap disabled even in quality mode")
    elif not preview_mode:
        if not RUBBERBAND_AVAILABLE:
            print(f"\n  ⚠  pyrubberband not available — time-stretching disabled")
        if not PEDALBOARD_AVAILABLE:
            print(f"  ⚠  pedalboard not available — EQ swap disabled")

    print(f"\n  {'─' * 56}")

    # ── Pre-process: separate stems for stem_blend transitions ────────────────
    has_stem_blend = any(s.get("transition_hint") == "stem_blend" for s in setlist)
    stems_cache    = {}
    if has_stem_blend:
        from modules.stems import run_stems_for_setlist, demucs_available
        # stems_cache_dir passed as parameter (or defaulted above)
        if demucs_available():
            stems_cache = run_stems_for_setlist(setlist, stems_cache_dir)
        else:
            print(
                "  ⚠  stem_blend transitions requested but demucs not found in PATH.\n"
                "     Install demucs (Python 3.12): pip install demucs\n"
                "     stem_blend will degrade to long_blend."
            )

    start_time = time.time()

    def progress(pos, total, fname):
        print_progress(pos + 1, total, fname, start_time)

    mix = render_session(setlist, chaos, preview_mode, progress_cb=progress,
                         quality_mode=quality_mode, stems_cache=stems_cache)

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