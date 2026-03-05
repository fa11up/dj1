"""
Module: Creative Mode
─────────────────────
Full-session stem composition. Every moment of the mix is built from stems
of different tracks playing simultaneously across 4 independent channels:

  drums   | bass   | melody (other) | vocals

The brain plans which track contributes each channel at each segment of the
session. Segments are measured in bars, not seconds, keeping everything
beat-grid-aligned regardless of tempo.

Key concepts:
  - master_bpm: single tempo for the entire session — all stems are
    time-stretched to this BPM before mixing
  - Segment: a time window (N bars) where each channel names which track to pull
  - Channel offset: the renderer tracks how far into each track it's read,
    so consecutive segments from the same track continue naturally
  - Crossfade: when a channel swaps to a different track, a bar-length
    crossfade is applied at the boundary

Flow:
  1. plan_creative_session() — Claude picks master_bpm + segment schedule
  2. stems.run_stems_for_creative() — demucs separates all referenced tracks
  3. render_creative_session() — builds 4 channels, mixes down, exports
"""

import json
import logging
import math
import os
import sys
import time
import yaml

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

from modules.stems import run_stems_for_creative, demucs_available, load_stems
from modules.brain import _build_track_summary, _make_anthropic_client
from dotenv import load_dotenv

try:
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
except ImportError:
    RUBBERBAND_AVAILABLE = False

load_dotenv()
log = logging.getLogger(__name__)

MODEL         = "claude-haiku-4-5" # claude-sonnet-4-6
SAMPLE_RATE   = 44100
CHANNELS      = 2
STEM_NAMES    = ("drums", "bass", "melody", "vocals")  # 'melody' maps to demucs 'other'
DEMUCS_TO_STEM = {"other": "melody"}     # rename demucs 'other' → 'melody' at load time
STEM_TO_DEMUCS = {"melody": "other"}     # inverse for loading from cache

# Crossfade between channel swaps (bars) — keeps joins musical
CHANNEL_CROSSFADE_BARS = 4
# Minimum segment duration (bars) — prevents choppy scheduling
SEGMENT_MIN_BARS = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def bars_to_ms(bars: float, bpm: float) -> int:
    if not bpm or bpm <= 0:
        bpm = 128.0
    return int(bars * 4 * 60.0 / bpm * 1000)


# ── Brain tool schema ─────────────────────────────────────────────────────────

CREATIVE_TOOL = {
    "name": "plan_creative_session",
    "description": (
        "Plan a full-session stem composition. Each segment defines which track "
        "provides drums, bass, melody, and vocals for a block of bars. "
        "The renderer mixes them simultaneously at the master_bpm."
    ),
    "input_schema": {
        "type": "object",
        "required": ["master_bpm", "segments", "session_narrative"],
        "properties": {
            "master_bpm": {
                "type": "number",
                "description": (
                    "Single BPM for the entire session. All stems will be time-stretched "
                    "to this tempo. Choose a BPM that most tracks in the library cluster "
                    "around (within ±15%). Avoid extremes unless mood requires it."
                )
            },
            "segments": {
                "type": "array",
                "description": (
                    "Time segments covering the full session duration. "
                    "Segments must be consecutive (each start_bar = prev start_bar + duration_bars). "
                    "Each segment must be at least 8 bars. Use multiples of 8 for musical phrasing."
                ),
                "items": {
                    "type": "object",
                    "required": ["start_bar", "duration_bars", "note"],
                    "properties": {
                        "start_bar": {
                            "type": "integer",
                            "description": "First bar of this segment (0-indexed). Must equal previous segment's end bar."
                        },
                        "duration_bars": {
                            "type": "integer",
                            "description": "Length of segment in bars. Minimum 8, multiples of 8 preferred."
                        },
                        "drums": {
                            "type": ["string", "null"],
                            "description": "track_id providing drums. Null = silence on drums channel."
                        },
                        "bass": {
                            "type": ["string", "null"],
                            "description": "track_id providing bass. Null = silence on bass channel."
                        },
                        "melody": {
                            "type": ["string", "null"],
                            "description": "track_id providing melody/synths/pads. Null = silence."
                        },
                        "vocals": {
                            "type": ["string", "null"],
                            "description": "track_id providing vocals. Null = no vocals (common for instrumentals)."
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "Why these sources work together here. Mention: harmonic compatibility, "
                                "energy intention, texture being created."
                            )
                        }
                    }
                }
            },
            "session_narrative": {
                "type": "string",
                "description": (
                    "3-5 sentences describing the arc of the composition: "
                    "how the mood_prompt shaped the master_bpm choice, "
                    "the overall energy journey, and any signature moments."
                )
            }
        }
    }
}

CREATIVE_SYSTEM_PROMPT = """\
You are composing a DJ mix in Creative Mode: instead of playing full tracks sequentially,
you are building a living stem composition where 4 channels (drums, bass, melody, vocals)
each draw from different source tracks simultaneously.

## Core Concept
At any moment the listener hears:
  - Drums from Track A
  - Bass from Track B
  - Melody (synths/pads) from Track C
  - Vocals from Track D (or silence if none suit)

These can swap independently — bass might stay constant for 64 bars while melody changes
every 32 and drums swap every 48. The result is a new composition that doesn't exist
anywhere in the original tracks.

## master_bpm Selection
- ALL stems are time-stretched to master_bpm. Choose carefully.
- Select a BPM that most tracks cluster around (within ±15% is clean stretch).
- For dark/melancholic moods: 90–118 BPM
- For driving/late-night: 118–132 BPM
- For euphoric/festival: 128–145 BPM
- Avoid stretching tracks more than ±20% — artifacts become noticeable.

## Harmonic Compatibility (Camelot Wheel)
Melody sources in adjacent/simultaneous segments should be harmonically compatible:
  - Same Camelot number: 8A and 8B (relative major/minor) — very smooth
  - Adjacent number, same letter: 8A→9A — slight tension, works well
  - Distant keys: only for intentional dissonance

Bass compatibility with melody:
  - Bass and melody from the same key sound best.
  - A bass in 6A under a melody in 8A creates tension — use intentionally.

Drums are harmonically neutral — they can always coexist with any melody/bass.

## Composition Strategy
- **Slow evolution**: change only one channel at a time for smooth texture shifts
- **Signature moments**: change all 4 channels simultaneously for a dramatic reset
- **Bass as anchor**: bass often stays constant longer (32-64 bars) while melody varies
- **Vocal sparing**: vocals cut through everything — use for peaks, not throughout
- **Build and release**: start minimal (drums + bass), layer in melody, peak with vocals,
  then strip back for the outro

## Segment Sizing Guidelines
- 8 bars: quick texture swap, high energy moment, bridge between ideas
- 16 bars: standard section length — a complete musical phrase
- 32 bars: longer section, allows the combination to breathe and develop
- 64 bars: extended groove, peak set moments, or meditative passages

## Output Requirements
- Segments must be consecutive (no gaps, no overlaps)
- Every track_id must come from the provided library
- Use null for a channel when silence works better than any available source
- Sessions typically need 8–16 segments to tell a compelling story
- The session_narrative should describe the arc, not just list what happens
"""


# ── Brain call ────────────────────────────────────────────────────────────────

def _build_creative_user_message(
    track_summaries: list[dict],
    mood_prompt: str,
    duration_min: int,
    chaos_factor: float,
) -> str:
    # Estimate total bars at median BPM
    bpms = [t["bpm"] for t in track_summaries if t.get("bpm")]
    median_bpm = sorted(bpms)[len(bpms)//2] if bpms else 128.0
    approx_bars = int(duration_min * 60 / (4 * 60.0 / median_bpm))

    chaos_note = (
        "High chaos: dramatic swaps, unexpected combinations, dissonant textures welcome."
        if chaos_factor > 0.6 else
        "Medium chaos: mostly smooth harmonic combinations, occasional surprise."
        if chaos_factor > 0.3 else
        "Low chaos: harmonically conservative, smooth slow-evolving composition."
    )

    library_json = json.dumps(track_summaries, ensure_ascii=False)

    return (
        f"Compose a {duration_min}-minute stem session.\n"
        f"  Mood: {mood_prompt or 'no specific mood'}\n"
        f"  Duration: ~{duration_min} min (~{approx_bars} bars at the chosen master_bpm)\n"
        f"  Chaos factor ({chaos_factor:.2f}/1.0): {chaos_note}\n\n"
        f"Track library ({len(track_summaries)} tracks):\n"
        f"{library_json}\n\n"
        f"Call plan_creative_session with your stem composition."
    )


def plan_creative_session(
    analyses: list[dict],
    config: dict,
) -> Optional[dict]:
    """
    Call Claude to plan a creative stem composition.

    Returns:
        creative_plan dict on success, None on any failure.
        creative_plan keys: master_bpm, segments, session_narrative, tracks_used
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("creative: ANTHROPIC_API_KEY not set")
        return None

    sess  = config.get("session", {})
    dj    = config.get("dj",      {})

    mood_prompt  = sess.get("mood_prompt", "")
    duration_min = sess.get("duration_minutes", 60)
    chaos        = float(dj.get("chaos_factor", 0.3))

    pool = [a for a in analyses if not a.get("analysis_error") and a.get("bpm")]
    if not pool:
        log.warning("creative: empty track pool")
        return None

    # Compact summaries (same as brain.py but include camelot prominently)
    summaries = [_build_track_summary(t) for t in pool]

    user_msg = _build_creative_user_message(summaries, mood_prompt, duration_min, chaos)

    print(f"  Creative: calling Claude API ({MODEL})...")
    try:
        client   = _make_anthropic_client(api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=CREATIVE_SYSTEM_PROMPT,
            tools=[CREATIVE_TOOL],
            tool_choice={"type": "tool", "name": "plan_creative_session"},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        log.error(f"creative: API call failed: {e}")
        return None

    # Parse tool use
    plan = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "plan_creative_session":
            plan = block.input
            break

    if not plan:
        log.error("creative: no tool_use block in response")
        return None

    # Validate segments
    segments = plan.get("segments", [])
    if len(segments) < 2:
        log.error(f"creative: only {len(segments)} segments — too sparse")
        return None

    # Collect all referenced track_ids
    valid_ids = {a["track_id"] for a in pool}
    tracks_used = set()
    for seg in segments:
        for ch in ("drums", "bass", "melody", "vocals"):
            tid = seg.get(ch)
            if tid and tid not in valid_ids:
                log.warning(f"creative: unknown track_id {tid!r} — setting to null")
                seg[ch] = None
            elif tid:
                tracks_used.add(tid)

    plan["tracks_used"] = list(tracks_used)

    master_bpm = plan.get("master_bpm", 128.0)
    narrative  = plan.get("session_narrative", "")
    print(f"  Creative: {len(segments)} segments, master BPM: {master_bpm:.1f}")
    if narrative:
        print(f"  Creative: {narrative}")

    return plan


# ── Stem preparation ──────────────────────────────────────────────────────────

def _time_stretch_seg(seg: AudioSegment, from_bpm: float, to_bpm: float) -> AudioSegment:
    """Time-stretch a pydub segment from from_bpm to to_bpm."""
    if not RUBBERBAND_AVAILABLE or not from_bpm or not to_bpm:
        return seg
    ratio = from_bpm / to_bpm
    ratio = max(0.8, min(1.25, ratio))   # clamp to ±25%
    if abs(ratio - 1.0) < 0.01:
        return seg
    try:
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= (2 ** (seg.sample_width * 8 - 1))
        if seg.channels == 2:
            samples = samples.reshape(-1, 2)
            l = pyrb.time_stretch(samples[:, 0], SAMPLE_RATE, ratio)
            r = pyrb.time_stretch(samples[:, 1], SAMPLE_RATE, ratio)
            stretched = np.stack([l, r], axis=1).flatten()
        else:
            stretched = pyrb.time_stretch(samples, SAMPLE_RATE, ratio)
        stretched = np.clip(stretched, -1.0, 1.0)
        pcm = (stretched * (2**15 - 1)).astype(np.int16)
        return AudioSegment(pcm.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=seg.channels)
    except Exception as e:
        log.warning(f"creative: time-stretch failed ({e}) — using original")
        return seg


def prepare_stems(
    creative_plan: dict,
    stems_cache: dict[str, dict[str, str]],
    analyses_by_id: dict[str, dict],
    preview_mode: bool = False,
) -> dict[str, dict[str, AudioSegment]]:
    """
    Load stems for all tracks used in the creative plan and time-stretch
    each to master_bpm.

    Returns:
        {track_id: {channel_name: AudioSegment}}
        'melody' channel maps to demucs 'other' stem.
    """
    master_bpm = float(creative_plan.get("master_bpm", 128.0))
    prepared   = {}

    for track_id in creative_plan.get("tracks_used", []):
        stem_paths = stems_cache.get(track_id)
        if not stem_paths:
            log.warning(f"creative: no stems for {track_id} — channel will be silent")
            continue

        # Load raw stems (demucs names: drums, bass, other, vocals)
        raw = load_stems(stem_paths)
        if not raw:
            log.warning(f"creative: stem load failed for {track_id}")
            continue

        # Rename 'other' → 'melody'
        stems = {}
        for demucs_name, seg in raw.items():
            ch = DEMUCS_TO_STEM.get(demucs_name, demucs_name)
            stems[ch] = seg

        # Time-stretch to master_bpm
        track_bpm = (
            analyses_by_id.get(track_id, {}).get("bpm_normalised") or
            analyses_by_id.get(track_id, {}).get("bpm") or
            master_bpm
        )

        if not preview_mode and abs(track_bpm - master_bpm) > 1.0:
            print(f"  Creative: stretching {analyses_by_id.get(track_id, {}).get('filename', track_id)} "
                  f"{track_bpm:.1f} → {master_bpm:.1f} BPM")
            stretched = {}
            for ch_name, seg in stems.items():
                stretched[ch_name] = _time_stretch_seg(seg, track_bpm, master_bpm)
            prepared[track_id] = stretched
        else:
            prepared[track_id] = stems

    return prepared


# ── Channel builder ───────────────────────────────────────────────────────────

def _build_channel(
    channel_name: str,
    segments: list[dict],
    prepared_stems: dict[str, dict[str, AudioSegment]],
    master_bpm: float,
) -> AudioSegment:
    """
    Build one stem channel as a continuous AudioSegment.

    For each segment, extracts `duration_bars` of audio from the chosen track.
    When the track changes between consecutive segments, applies a crossfade.
    Tracks are read sequentially — the channel remembers its position in each
    track so consecutive segments from the same source continue naturally.
    """
    cf_ms       = bars_to_ms(CHANNEL_CROSSFADE_BARS, master_bpm)
    bar_ms      = bars_to_ms(1, master_bpm)
    silence_fn  = lambda ms: AudioSegment.silent(duration=ms, frame_rate=SAMPLE_RATE).set_channels(CHANNELS)

    channel      = silence_fn(0)
    offsets      = {}    # {track_id: current_read_position_ms}
    prev_track   = None

    for seg in segments:
        track_id    = seg.get(channel_name)
        duration_ms = int(seg["duration_bars"] * bar_ms)

        if not track_id or track_id not in prepared_stems:
            # Silence on this channel for this segment
            if len(channel) == 0:
                channel = silence_fn(duration_ms)
            else:
                channel = channel + silence_fn(duration_ms)
            prev_track = None
            continue

        stems   = prepared_stems[track_id]
        stem_ch = stems.get(channel_name)
        if stem_ch is None:
            if len(channel) == 0:
                channel = silence_fn(duration_ms)
            else:
                channel = channel + silence_fn(duration_ms)
            prev_track = None
            continue

        # Initialise offset at the track's playable start (skip silence head)
        if track_id not in offsets:
            offsets[track_id] = 0

        offset_ms = offsets[track_id]

        # Extract audio with crossfade buffer
        needed_ms = duration_ms + cf_ms
        available = len(stem_ch) - offset_ms

        if available < duration_ms:
            # Track exhausted — wrap to 20% in (avoids re-intro)
            wrap_ms = int(len(stem_ch) * 0.20)
            offsets[track_id] = wrap_ms
            offset_ms         = wrap_ms
            available         = len(stem_ch) - offset_ms

        extract_ms = min(needed_ms, available)
        audio      = stem_ch[offset_ms : offset_ms + extract_ms]

        # Pad if still short
        if len(audio) < duration_ms:
            audio = audio + silence_fn(duration_ms - len(audio))

        offsets[track_id] = offset_ms + duration_ms

        # Append with crossfade if track changed
        switching = prev_track is not None and prev_track != track_id
        if len(channel) == 0:
            channel = audio[:duration_ms]
        elif switching and len(channel) >= cf_ms and len(audio) >= cf_ms + duration_ms:
            channel = channel.append(audio, crossfade=cf_ms)
        else:
            channel = channel + audio[:duration_ms]

        prev_track = track_id

    return channel


# ── Channel mixer ─────────────────────────────────────────────────────────────

def _mix_channels(
    channels: list[AudioSegment],
    target_duration_ms: int,
) -> AudioSegment:
    """
    Mix N stem channels to a single stereo AudioSegment by averaging in float32.
    Pads/trims all channels to target_duration_ms before mixing.
    """
    active = [ch for ch in channels if ch is not None and len(ch) > 0]
    if not active:
        return AudioSegment.silent(duration=target_duration_ms, frame_rate=SAMPLE_RATE)

    # Pad or trim to target length
    normed = []
    silence = AudioSegment.silent(duration=target_duration_ms, frame_rate=SAMPLE_RATE).set_channels(CHANNELS)
    for ch in active:
        ch = ch.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
        if len(ch) < target_duration_ms:
            ch = ch + AudioSegment.silent(duration=target_duration_ms - len(ch), frame_rate=SAMPLE_RATE).set_channels(CHANNELS)
        else:
            ch = ch[:target_duration_ms]
        normed.append(ch)

    # Mix via numpy average (prevents clipping)
    # target_samples is the ground-truth length — pydub ms-slicing has ±1 frame
    # rounding so arrays can differ by 2 samples (stereo); normalise before summing.
    target_samples = int(target_duration_ms * SAMPLE_RATE * CHANNELS / 1000)
    arrays = []
    for ch in normed:
        arr = np.array(ch.get_array_of_samples(), dtype=np.float32)
        arr /= (2 ** (ch.sample_width * 8 - 1))
        if len(arr) < target_samples:
            arr = np.concatenate([arr, np.zeros(target_samples - len(arr), dtype=np.float32)])
        elif len(arr) > target_samples:
            arr = arr[:target_samples]
        arrays.append(arr)

    mixed = sum(arrays) / len(arrays)
    mixed = np.clip(mixed, -1.0, 1.0)
    pcm   = (mixed * (2**15 - 1)).astype(np.int16)

    return AudioSegment(
        pcm.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=CHANNELS,
    )


# ── Render ────────────────────────────────────────────────────────────────────

def render_creative_session(
    creative_plan: dict,
    prepared_stems: dict[str, dict[str, AudioSegment]],
    duration_min: int,
    preview_mode: bool = False,
) -> AudioSegment:
    """
    Render the creative stem composition to a single AudioSegment.

    Builds 4 independent channels (drums, bass, melody, vocals) from the
    segment schedule, then mixes them down to stereo.
    """
    master_bpm  = float(creative_plan.get("master_bpm", 128.0))
    segments    = creative_plan.get("segments", [])

    # Total target duration
    target_ms = int(duration_min * 60 * 1000)

    print(f"\n  Creative: building channels at {master_bpm:.1f} BPM...")
    t0 = time.time()

    built = {}
    for ch in ("drums", "bass", "melody", "vocals"):
        print(f"  Creative: assembling {ch} channel...")
        built[ch] = _build_channel(ch, segments, prepared_stems, master_bpm)

    print(f"  Creative: mixing 4 channels → stereo...")
    mix = _mix_channels(list(built.values()), target_ms)
    mix = pydub_normalize(mix)

    elapsed = time.time() - t0
    print(f"  Creative: rendered in {elapsed:.1f}s  ({len(mix)/1000/60:.1f} min)")

    return mix


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    analysis_path   = "./data/analysis.json",
    config_path     = "./config/session.yaml",
    stems_cache_dir = "./data/.stems_cache",
    output_path     = None,
    preview_mode    = False,
    quality_mode    = False,
):
    """
    Full creative mode pipeline: plan → separate → render → export.
    Replaces Modules 4 + 5 when session.mode == "creative".
    """
    print("─" * 60 + "\n  AutoDJ — Creative Mode\n" + "─" * 60)

    if not Path(analysis_path).exists():
        sys.exit(f"✗ {analysis_path} not found — run Module 2 first.")
    if not Path(config_path).exists():
        sys.exit(f"✗ {config_path} not found.")

    analyses_raw = json.loads(Path(analysis_path).read_text())["analyses"]
    config       = yaml.safe_load(Path(config_path).read_text())

    sess         = config.get("session", {})
    dj           = config.get("dj",      {})
    output_cfg   = config.get("output",  {})
    duration_min = sess.get("duration_minutes", 60)
    mood_prompt  = sess.get("mood_prompt", "")

    print(f"\n  Mode:     Creative (full-session stem composition)")
    print(f"  Duration: ~{duration_min} min")
    print(f"  Mood:     {mood_prompt or '(none)'}")
    print(f"  Tracks:   {len([a for a in analyses_raw if not a.get('analysis_error')])} usable")

    # ── Plan ──────────────────────────────────────────────────────────────────
    creative_plan = plan_creative_session(analyses_raw, config)
    if not creative_plan:
        sys.exit("✗ Creative planning failed — check ANTHROPIC_API_KEY and track pool.")

    # ── Stem separation ────────────────────────────────────────────────────────

    if not demucs_available():
        sys.exit(
            "✗ Creative mode requires demucs for stem separation.\n"
            "  Install: pip install demucs  (Python ≤3.12)\n"
            "  Ensure `demucs` is in your PATH."
        )

    print(f"\n  Separating stems for {len(creative_plan['tracks_used'])} tracks...")
    stems_cache = run_stems_for_creative(
        tracks_used=creative_plan["tracks_used"],
        analyses=analyses_raw,
        cache_dir=stems_cache_dir,
    )

    if not stems_cache:
        sys.exit("✗ Stem separation produced no results.")

    # ── Prepare (load + stretch) ───────────────────────────────────────────────
    analyses_by_id = {a["track_id"]: a for a in analyses_raw}
    prepared = prepare_stems(creative_plan, stems_cache, analyses_by_id, preview_mode)

    if not prepared:
        sys.exit("✗ No stems could be loaded for rendering.")

    # ── Render ────────────────────────────────────────────────────────────────
    mix = render_creative_session(creative_plan, prepared, duration_min, preview_mode)

    if mix is None or len(mix) == 0:
        sys.exit("✗ Creative render produced no audio.")

    # ── Export ────────────────────────────────────────────────────────────────
    if output_path is None:
        fmt       = output_cfg.get("format", "mp3")
        suffix    = "_preview" if preview_mode else "_creative"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"./output/mix_{timestamp}{suffix}.{fmt}"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lstrip(".").lower() or "mp3"
    export_kwargs = {"format": fmt}
    if fmt == "mp3":
        export_kwargs["bitrate"] = "128k" if preview_mode else output_cfg.get("bitrate", "320k")

    print(f"\n  Exporting → {output_path}")
    mix.export(str(out), **export_kwargs)

    size_mb = out.stat().st_size / (1024 * 1024)
    duration_actual = len(mix) / 1000 / 60
    print(f"  ✅ {output_path}  ({size_mb:.1f} MB, {duration_actual:.1f} min)")
    print("─" * 60)

    # Save plan for reference
    plan_path = Path(stems_cache_dir).parent / "creative_plan.json"
    plan_path.write_text(json.dumps(creative_plan, indent=2, ensure_ascii=False))
    print(f"  📋 Creative plan → {plan_path}")

    return str(out)
