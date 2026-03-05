"""
Tests for Creative Mode (modules/creative.py)
Run with: uv run python tests/test_creative.py

No API calls, no demucs — tests cover pure-logic helpers with synthetic audio:
  bars_to_ms, _mix_channels, _build_channel, prepare_stems, schema validation.
"""

import sys, traceback
from pathlib import Path

import numpy as np
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.creative import (
    bars_to_ms,
    CREATIVE_TOOL,
    CREATIVE_SYSTEM_PROMPT,
    CHANNEL_CROSSFADE_BARS,
    SEGMENT_MIN_BARS,
    STEM_NAMES,
    DEMUCS_TO_STEM,
    _mix_channels,
    _build_channel,
    prepare_stems,
    RUBBERBAND_AVAILABLE,
)

_results = {"passed": 0, "failed": 0, "errors": []}

SAMPLE_RATE = 44100
CHANNELS    = 2

def ok(cond, msg=""):
    if not cond: raise AssertionError(msg or f"Expected truthy, got {cond!r}")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got: {a!r}")
def approx(a, b, tol, msg=""):
    if abs(a - b) > tol: raise AssertionError(msg or f"Expected ~{b}±{tol}, got {a}")

def run_test(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        _results["passed"] += 1
    except Exception as e:
        print(f"  ❌ {name}  →  {type(e).__name__}: {e}")
        _results["failed"] += 1
        _results["errors"].append((name, traceback.format_exc()))

def section(title):
    print(f"\n{'─'*52}\n  {title}\n{'─'*52}")


# ── Fixture helpers ───────────────────────────────────────────────────────────

def make_seg(duration_sec=10.0, freq=440):
    n      = int(SAMPLE_RATE * duration_sec)
    t      = np.linspace(0, duration_sec, n, endpoint=False)
    wave   = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
    stereo = np.stack([wave, wave], axis=1).flatten()
    return AudioSegment(stereo.tobytes(), frame_rate=SAMPLE_RATE,
                        sample_width=2, channels=CHANNELS)

def make_stems(duration_sec=30.0):
    """Fake stems dict with 4 channels (creative uses 'melody', not 'other')."""
    return {
        ch: make_seg(duration_sec, freq=200 + 50 * i)
        for i, ch in enumerate(("drums", "bass", "melody", "vocals"))
    }

def make_segment(start_bar=0, dur_bars=16,
                 drums="t1", bass="t1", melody="t1", vocals=None):
    return {
        "start_bar":     start_bar,
        "duration_bars": dur_bars,
        "drums":         drums,
        "bass":          bass,
        "melody":        melody,
        "vocals":        vocals,
        "note":          "test segment",
    }


# ── bars_to_ms ────────────────────────────────────────────────────────────────

def test_bars_to_ms():
    section("bars_to_ms")

    run_test("4 bars at 120 BPM = 8000ms",
        lambda: eq(bars_to_ms(4, 120.0), 8000))
    run_test("1 bar at 128 BPM ≈ 1875ms",
        lambda: approx(bars_to_ms(1, 128.0), 1875, 5))
    run_test("zero BPM → fallback, no crash",
        lambda: ok(bars_to_ms(4, 0) > 0))
    run_test("None BPM → fallback, no crash",
        lambda: ok(bars_to_ms(4, None) > 0))
    run_test("more bars → more ms",
        lambda: ok(bars_to_ms(8, 128) > bars_to_ms(4, 128)))
    run_test("faster BPM → less ms per bar",
        lambda: ok(bars_to_ms(4, 160) < bars_to_ms(4, 80)))
    run_test("fractional bars supported",
        lambda: ok(bars_to_ms(0.5, 128) > 0))


# ── CREATIVE_TOOL schema ──────────────────────────────────────────────────────

def test_creative_tool_schema():
    section("CREATIVE_TOOL schema")

    run_test("has correct name",
        lambda: eq(CREATIVE_TOOL["name"], "plan_creative_session"))
    run_test("has description",
        lambda: ok(len(CREATIVE_TOOL.get("description", "")) > 10))
    run_test("has input_schema",
        lambda: ok("input_schema" in CREATIVE_TOOL))

    def schema_has_required_properties():
        props = CREATIVE_TOOL["input_schema"].get("properties", {})
        for key in ("master_bpm", "segments", "session_narrative"):
            ok(key in props, f"schema missing property: {key}")

    def segments_have_channel_fields():
        seg_props = (
            CREATIVE_TOOL["input_schema"]["properties"]["segments"]
            .get("items", {})
            .get("properties", {})
        )
        for ch in ("drums", "bass", "melody", "vocals"):
            ok(ch in seg_props, f"segment schema missing channel: {ch}")

    def segments_have_required_fields():
        required = (
            CREATIVE_TOOL["input_schema"]["properties"]["segments"]
            .get("items", {})
            .get("required", [])
        )
        ok("start_bar"     in required, "start_bar must be required")
        ok("duration_bars" in required, "duration_bars must be required")

    def master_bpm_is_number_type():
        bpm_schema = CREATIVE_TOOL["input_schema"]["properties"]["master_bpm"]
        eq(bpm_schema["type"], "number")

    run_test("schema_has_required_properties",  schema_has_required_properties)
    run_test("segments_have_channel_fields",    segments_have_channel_fields)
    run_test("segments_have_required_fields",   segments_have_required_fields)
    run_test("master_bpm_is_number_type",       master_bpm_is_number_type)


# ── _mix_channels ─────────────────────────────────────────────────────────────

def test_mix_channels():
    section("_mix_channels")

    def returns_audio_segment():
        ch1 = make_seg(10.0)
        ch2 = make_seg(10.0)
        result = _mix_channels([ch1, ch2], len(ch1))
        ok(isinstance(result, AudioSegment))

    def correct_length_equal_inputs():
        target = 10_000
        ch1 = make_seg(10.0)
        ch2 = make_seg(10.0)
        result = _mix_channels([ch1, ch2], target)
        approx(len(result), target, 50)

    def pads_short_channel():
        short  = make_seg(5.0)
        target = 10_000
        result = _mix_channels([short], target)
        approx(len(result), target, 50)

    def trims_long_channel():
        long_  = make_seg(15.0)
        target = 10_000
        result = _mix_channels([long_], target)
        approx(len(result), target, 50)

    def empty_list_returns_silence():
        result = _mix_channels([], 5_000)
        ok(isinstance(result, AudioSegment))
        approx(len(result), 5_000, 50)

    def all_none_returns_silence():
        result = _mix_channels([None, None], 5_000)
        ok(isinstance(result, AudioSegment))

    def single_channel_no_crash():
        ch = make_seg(5.0)
        result = _mix_channels([ch], len(ch))
        ok(isinstance(result, AudioSegment))

    def four_channels_no_crash():
        channels = [make_seg(10.0, freq=200 + i * 100) for i in range(4)]
        result = _mix_channels(channels, 10_000)
        ok(isinstance(result, AudioSegment))

    def shape_mismatch_tolerance():
        """Channels that differ by a few samples (pydub ms rounding) should not crash."""
        ch1 = make_seg(10.0)
        # Simulate ±2 sample rounding: strip 4 bytes (1 stereo frame) from raw
        raw = ch1.raw_data[:-4]
        ch2 = AudioSegment(raw, frame_rate=SAMPLE_RATE, sample_width=2, channels=CHANNELS)
        result = _mix_channels([ch1, ch2], len(ch1))
        ok(isinstance(result, AudioSegment))

    def mixed_output_quieter_than_single():
        """Averaging N channels should be ≤ volume of loudest single channel."""
        ch1 = make_seg(5.0, freq=440)
        ch2 = make_seg(5.0, freq=880)
        target = min(len(ch1), len(ch2))
        mixed  = _mix_channels([ch1, ch2], target)
        single = _mix_channels([ch1], target)
        # Max amplitude of mixed should be ≤ single (averaging, not summing)
        mixed_max  = max(abs(s) for s in mixed.get_array_of_samples())
        single_max = max(abs(s) for s in single.get_array_of_samples())
        ok(mixed_max <= single_max + 100,  # small tolerance for float rounding
           "Mixed output should not be louder than single channel")

    for fn in [returns_audio_segment, correct_length_equal_inputs, pads_short_channel,
               trims_long_channel, empty_list_returns_silence, all_none_returns_silence,
               single_channel_no_crash, four_channels_no_crash,
               shape_mismatch_tolerance, mixed_output_quieter_than_single]:
        run_test(fn.__name__, fn)


# ── _build_channel ────────────────────────────────────────────────────────────

def test_build_channel():
    section("_build_channel")

    master_bpm = 128.0
    prepared   = {"t1": make_stems(60.0), "t2": make_stems(60.0)}

    def returns_audio_segment():
        segs   = [make_segment(0, 16)]
        result = _build_channel("drums", segs, prepared, master_bpm)
        ok(isinstance(result, AudioSegment))

    def silence_when_track_is_none():
        segs = [{"start_bar": 0, "duration_bars": 8,
                 "drums": None, "bass": None, "melody": None, "vocals": None,
                 "note": "silence"}]
        result = _build_channel("drums", segs, prepared, master_bpm)
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    def single_segment_approximate_length():
        dur_bars = 16
        segs = [make_segment(0, dur_bars, drums="t1")]
        result = _build_channel("drums", segs, prepared, master_bpm)
        expected_ms = bars_to_ms(dur_bars, master_bpm)
        approx(len(result), expected_ms, expected_ms * 0.2)

    def two_segments_longer_than_one():
        segs1 = [make_segment(0, 16)]
        segs2 = [make_segment(0, 16), make_segment(16, 16)]
        r1 = _build_channel("drums", segs1, prepared, master_bpm)
        r2 = _build_channel("drums", segs2, prepared, master_bpm)
        ok(len(r2) > len(r1), "Two segments should produce more audio than one")

    def track_switch_no_crash():
        segs = [make_segment(0,  16, drums="t1"),
                make_segment(16, 16, drums="t2")]
        result = _build_channel("drums", segs, prepared, master_bpm)
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    def missing_track_falls_back_to_silence():
        segs = [make_segment(0,  16, drums="t1"),
                make_segment(16, 16, drums="MISSING_TRACK")]
        result = _build_channel("drums", segs, prepared, master_bpm)
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    def empty_segments_returns_silence():
        result = _build_channel("drums", [], prepared, master_bpm)
        ok(isinstance(result, AudioSegment))

    for fn in [returns_audio_segment, silence_when_track_is_none,
               single_segment_approximate_length, two_segments_longer_than_one,
               track_switch_no_crash, missing_track_falls_back_to_silence,
               empty_segments_returns_silence]:
        run_test(fn.__name__, fn)


# ── prepare_stems ─────────────────────────────────────────────────────────────

def test_prepare_stems():
    section("prepare_stems")

    def returns_empty_when_cache_is_empty():
        plan = {"master_bpm": 128.0, "tracks_used": ["t1"]}
        result = prepare_stems(plan, {}, {"t1": {"bpm": 128.0, "bpm_normalised": 128.0}})
        eq(result, {})

    def handles_empty_tracks_used():
        plan = {"master_bpm": 128.0, "tracks_used": []}
        result = prepare_stems(plan, {}, {})
        eq(result, {})

    def skips_tracks_absent_from_cache():
        plan  = {"master_bpm": 128.0, "tracks_used": ["t1", "t2"]}
        cache = {}  # t1 and t2 not in cache
        result = prepare_stems(plan, cache, {})
        eq(result, {})

    def returns_dict_type():
        plan = {"master_bpm": 128.0, "tracks_used": []}
        result = prepare_stems(plan, {}, {})
        ok(isinstance(result, dict))

    for fn in [returns_empty_when_cache_is_empty, handles_empty_tracks_used,
               skips_tracks_absent_from_cache, returns_dict_type]:
        run_test(fn.__name__, fn)


# ── module constants ──────────────────────────────────────────────────────────

def test_constants():
    section("module constants")

    run_test("STEM_NAMES has 4 entries",
        lambda: eq(len(STEM_NAMES), 4))
    run_test("STEM_NAMES uses 'melody' not 'other'",
        lambda: ok("melody" in STEM_NAMES and "other" not in STEM_NAMES,
                   "creative mode uses 'melody' (demucs 'other' is renamed)"))
    run_test("DEMUCS_TO_STEM maps 'other' → 'melody'",
        lambda: eq(DEMUCS_TO_STEM.get("other"), "melody"))
    run_test("CHANNEL_CROSSFADE_BARS >= 1",
        lambda: ok(CHANNEL_CROSSFADE_BARS >= 1))
    run_test("SEGMENT_MIN_BARS >= 4",
        lambda: ok(SEGMENT_MIN_BARS >= 4))
    run_test("CREATIVE_SYSTEM_PROMPT is non-empty string",
        lambda: ok(isinstance(CREATIVE_SYSTEM_PROMPT, str) and len(CREATIVE_SYSTEM_PROMPT) > 100))
    run_test("CREATIVE_SYSTEM_PROMPT mentions master_bpm",
        lambda: ok("master_bpm" in CREATIVE_SYSTEM_PROMPT.lower() or
                   "master bpm" in CREATIVE_SYSTEM_PROMPT.lower()))


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Creative Mode Tests\n" + "═"*52)
    print(f"\n  rubberband={'✓' if RUBBERBAND_AVAILABLE else '✗'}\n")

    test_bars_to_ms()
    test_creative_tool_schema()
    test_mix_channels()
    test_build_channel()
    test_prepare_stems()
    test_constants()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'═'*52}")
    if _results["failed"]:
        print(f"  {_results['passed']}/{total} passed | {_results['failed']} FAILED")
        for name, tb in _results["errors"]:
            print(f"\n  ── {name} ──\n{tb}")
        sys.exit(1)
    else:
        print(f"  {total}/{total} passed ✅")
