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

# Crossfade durations by transition hint (milliseconds)
FADE_DURATIONS = {
    "hard_cut":   0,
    "quick_fade": 4_000,
    "crossfade":  10_000,
    "long_blend": 24_000,
}

# EQ crossover frequency (Hz) — bass cut on outgoing, bring in on incoming
EQ_CROSSOVER_HZ = 200.0


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


# ── EQ swap ───────────────────────────────────────────────────────────────────

def apply_eq_swap(outgoing_seg, incoming_seg, fade_ms):
    """
    During the overlap window:
      - Outgoing: gradually cut bass (low-pass → high-pass)
      - Incoming: gradually bring in bass (high-pass → low-pass)

    Uses pedalboard for filtering. Degrades to unfiltered if unavailable.
    Returns (outgoing_tail, incoming_head) with EQ applied.
    """
    if not PEDALBOARD_AVAILABLE or fade_ms <= 0:
        return outgoing_seg, incoming_seg

    try:
        def seg_to_float(s):
            arr = np.array(s.get_array_of_samples(), dtype=np.float32)
            arr /= (2 ** (s.sample_width * 8 - 1))
            if s.channels == 2:
                arr = arr.reshape(-1, 2).T  # pedalboard wants (channels, samples)
            else:
                arr = arr.reshape(1, -1)
            return arr

        def float_to_seg(arr, channels):
            if channels == 2:
                arr = arr.T.flatten()
            else:
                arr = arr.flatten()
            arr = np.clip(arr, -1.0, 1.0)
            pcm = (arr * (2 ** 15 - 1)).astype(np.int16)
            return AudioSegment(pcm.tobytes(), frame_rate=SAMPLE_RATE,
                                sample_width=2, channels=channels)

        # Outgoing tail: cut low end (simulates bass leaving the mix)
        out_tail  = outgoing_seg[-fade_ms:]
        out_audio = seg_to_float(out_tail)
        board_out = Pedalboard([LowpassFilter(cutoff_frequency_hz=EQ_CROSSOVER_HZ * 2)])
        out_filtered = board_out(out_audio, SAMPLE_RATE)
        out_tail_eq  = float_to_seg(out_filtered, out_tail.channels)

        # Incoming head: cut low end initially (bass builds in)
        in_head   = incoming_seg[:fade_ms]
        in_audio  = seg_to_float(in_head)
        board_in  = Pedalboard([HighpassFilter(cutoff_frequency_hz=EQ_CROSSOVER_HZ)])
        in_filtered  = board_in(in_audio, SAMPLE_RATE)
        in_head_eq   = float_to_seg(in_filtered, in_head.channels)

        # Reconstruct full segments with EQ applied to transition regions
        outgoing_eq = outgoing_seg[:-fade_ms] + out_tail_eq
        incoming_eq = in_head_eq + incoming_seg[fade_ms:]

        return outgoing_eq, incoming_eq

    except Exception as e:
        print(f"    ⚠  EQ swap failed ({e}) — skipping EQ")
        return outgoing_seg, incoming_seg


# ── Transition decider ────────────────────────────────────────────────────────

def decide_transition_ops(hint, chaos, preview_mode):
    """
    Given a transition hint and chaos factor, return a dict of which
    operations to apply. At high chaos, skip expensive/precise operations.

    Returns:
        {
          "fade_ms":    int,
          "do_stretch": bool,
          "do_eq":      bool,
          "do_beat_align": bool,
        }
    """
    base_fade = FADE_DURATIONS.get(hint, 8_000)

    if preview_mode:
        # Preview: halve fade durations, no stretching
        return {
            "fade_ms":       min(base_fade, 6_000),
            "do_stretch":    False,
            "do_eq":         False,
            "do_beat_align": False,
        }

    if chaos >= 0.8:
        # Wild: hard cut or tiny fade, nothing else
        return {
            "fade_ms":       0 if hint == "hard_cut" else min(base_fade, 2_000),
            "do_stretch":    False,
            "do_eq":         False,
            "do_beat_align": False,
        }
    elif chaos >= 0.6:
        # Volume fade only
        return {
            "fade_ms":       base_fade,
            "do_stretch":    False,
            "do_eq":         False,
            "do_beat_align": False,
        }
    elif chaos >= 0.3:
        # EQ + fade, no stretch
        return {
            "fade_ms":       base_fade,
            "do_stretch":    False,
            "do_eq":         PEDALBOARD_AVAILABLE,
            "do_beat_align": False,
        }
    else:
        # Full pipeline
        return {
            "fade_ms":       base_fade,
            "do_stretch":    RUBBERBAND_AVAILABLE,
            "do_eq":         PEDALBOARD_AVAILABLE,
            "do_beat_align": LIBROSA_AVAILABLE,
        }


# ── Core render ───────────────────────────────────────────────────────────────

def render_transition(outgoing, incoming, slot_out, slot_in, chaos, preview_mode):
    """
    Blend outgoing → incoming segment based on transition ops.

    outgoing / incoming : pydub AudioSegments (already trimmed to cues)
    slot_out / slot_in  : session slot dicts (for BPM, beat_times, hint)

    Returns the combined segment including the overlap blend.
    """
    hint = slot_in.get("transition_hint", "crossfade")
    ops  = decide_transition_ops(hint, chaos, preview_mode)
    fade_ms = ops["fade_ms"]

    # ── Beat alignment ────────────────────────────────────────────────────────
    if ops["do_beat_align"]:
        beat_times = slot_out.get("beat_times") or []
        if beat_times and fade_ms > 0:
            # Snap the splice point to the nearest beat before the outro
            splice_target_ms = len(outgoing) - fade_ms
            aligned_ms       = nearest_beat_ms(splice_target_ms, beat_times)
            # Only adjust if within 500ms of original splice point
            if abs(aligned_ms - splice_target_ms) < 500:
                fade_ms = len(outgoing) - aligned_ms

    fade_ms = max(0, min(fade_ms, len(outgoing) - 1000, len(incoming) - 1000))

    # ── BPM time-stretch ──────────────────────────────────────────────────────
    if ops["do_stretch"] and fade_ms > 0:
        bpm_out = slot_out.get("actual_bpm")
        bpm_in  = slot_in.get("actual_bpm")
        if bpm_out and bpm_in and abs(bpm_out - bpm_in) > 1:
            # Stretch only the incoming head (overlap region) to match outgoing
            head     = incoming[:fade_ms + 4000]  # small buffer past fade
            rest     = incoming[fade_ms + 4000:]
            head_str = time_stretch_segment(head, bpm_in, bpm_out)
            incoming = head_str + rest

    # ── EQ swap ───────────────────────────────────────────────────────────────
    if ops["do_eq"] and fade_ms > 0:
        outgoing, incoming = apply_eq_swap(outgoing, incoming, fade_ms)

    # ── Volume crossfade ──────────────────────────────────────────────────────
    if fade_ms <= 0:
        return outgoing + incoming

    out_body = outgoing[:-fade_ms]
    out_tail = outgoing[-fade_ms:].fade_out(fade_ms)
    in_head  = incoming[:fade_ms].fade_in(fade_ms)
    in_body  = incoming[fade_ms:]

    # Overlay the tails
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