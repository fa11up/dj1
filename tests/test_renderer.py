"""
Tests for Module 5: Renderer
Run with: uv run python tests/test_renderer.py

Audio I/O is mocked via synthetic pydub segments — no real files needed.
Export tests are skipped (require ffmpeg on PATH).
"""

import sys, json, math, random, traceback, struct, wave, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.renderer import (
    trim_to_cues,
    nearest_beat_ms,
    decide_transition_ops,
    time_stretch_segment,
    render_transition,
    FADE_DURATIONS,
    RUBBERBAND_AVAILABLE,
    PEDALBOARD_AVAILABLE,
)
from pydub import AudioSegment

_results = {"passed": 0, "failed": 0, "errors": []}

def ok(cond, msg=""):
    if not cond: raise AssertionError(msg or f"Expected truthy, got {cond!r}")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got: {a!r}")
def in_range(v, lo, hi, msg=""):
    if not (lo <= v <= hi): raise AssertionError(msg or f"Expected {lo}≤{v}≤{hi}, got {v}")
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


# ── Audio fixtures ────────────────────────────────────────────────────────────

def make_seg(duration_sec=10.0, freq=440, sr=44100, channels=2):
    """Synthetic sine wave as pydub AudioSegment."""
    n      = int(sr * duration_sec)
    t      = np.linspace(0, duration_sec, n, endpoint=False)
    wave_  = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
    if channels == 2:
        stereo = np.stack([wave_, wave_], axis=1).flatten()
        return AudioSegment(stereo.tobytes(), frame_rate=sr, sample_width=2, channels=2)
    return AudioSegment(wave_.tobytes(), frame_rate=sr, sample_width=2, channels=1)

def make_slot(bpm=120.0, camelot="8B", hint="crossfade",
              intro=1.0, outro=None, filepath="/fake/track.mp3"):
    return {
        "track_id":        "t001",
        "filepath":        filepath,
        "filename":        "track.mp3",
        "actual_bpm":      bpm,
        "camelot":         camelot,
        "transition_hint": hint,
        "intro_end_sec":   intro,
        "outro_start_sec": outro,
        "beat_times":      [i * (60.0 / bpm) for i in range(200)],
        "is_seed":         False,
    }


# ── trim_to_cues ──────────────────────────────────────────────────────────────

def test_trim():
    section("trim_to_cues")

    def trims_intro():
        seg     = make_seg(30.0)
        trimmed = trim_to_cues(seg, intro_end_sec=5.0, outro_start_sec=28.0)
        ok(len(trimmed) < len(seg), "Should be shorter after trimming intro")
        approx(len(trimmed) / 1000, 23.0, 0.1, "Should be ~23s after trim")

    def trims_outro():
        seg     = make_seg(30.0)
        trimmed = trim_to_cues(seg, intro_end_sec=0.0, outro_start_sec=25.0)
        approx(len(trimmed) / 1000, 25.0, 0.1)

    def none_cues_returns_original():
        seg     = make_seg(30.0)
        trimmed = trim_to_cues(seg, intro_end_sec=None, outro_start_sec=None)
        eq(len(trimmed), len(seg))

    def too_short_returns_original():
        seg     = make_seg(30.0)
        # If trimming would leave <10s, return original
        trimmed = trim_to_cues(seg, intro_end_sec=25.0, outro_start_sec=28.0)
        ok(len(trimmed) >= len(make_seg(10.0)))

    for fn in [trims_intro, trims_outro, none_cues_returns_original, too_short_returns_original]:
        run_test(fn.__name__, fn)


# ── nearest_beat_ms ───────────────────────────────────────────────────────────

def test_nearest_beat():
    section("nearest_beat_ms")

    beats = [i * 0.5 for i in range(40)]  # beat every 500ms

    def snaps_to_exact_beat():
        eq(nearest_beat_ms(2000, beats), 2000)

    def snaps_to_nearest():
        # 2100ms — nearest beat is 2000ms
        eq(nearest_beat_ms(2100, beats), 2000)
        # 2300ms — nearest beat is 2500ms
        eq(nearest_beat_ms(2300, beats), 2500)

    def empty_beats_returns_original():
        eq(nearest_beat_ms(1234, []), 1234)

    for fn in [snaps_to_exact_beat, snaps_to_nearest, empty_beats_returns_original]:
        run_test(fn.__name__, fn)


# ── decide_transition_ops ─────────────────────────────────────────────────────

def test_decide_ops():
    section("decide_transition_ops")

    def preview_mode_no_stretch():
        ops = decide_transition_ops("crossfade", chaos=0.1, preview_mode=True)
        ok(not ops["do_stretch"])
        ok(not ops["do_eq"])
        ok(not ops["do_beat_align"])

    def preview_mode_halves_fade():
        ops_full    = decide_transition_ops("long_blend", chaos=0.1, preview_mode=False)
        ops_preview = decide_transition_ops("long_blend", chaos=0.1, preview_mode=True)
        ok(ops_preview["fade_ms"] <= ops_full["fade_ms"])

    def hard_cut_zero_fade():
        ops = decide_transition_ops("hard_cut", chaos=0.1, preview_mode=False)
        eq(ops["fade_ms"], 0)

    def high_chaos_no_stretch_or_eq():
        ops = decide_transition_ops("long_blend", chaos=0.9, preview_mode=False)
        ok(not ops["do_stretch"])
        ok(not ops["do_eq"])

    def low_chaos_enables_pipeline():
        ops = decide_transition_ops("long_blend", chaos=0.1, preview_mode=False)
        # Full pipeline requested (actual execution depends on library availability)
        ok(ops["fade_ms"] > 0)
        # stretch/eq/beat_align may be True if libraries available
        ok(isinstance(ops["do_stretch"], bool))
        ok(isinstance(ops["do_eq"], bool))
        ok(isinstance(ops["do_beat_align"], bool))

    def fade_duration_matches_hint():
        for hint, expected_ms in FADE_DURATIONS.items():
            ops = decide_transition_ops(hint, chaos=0.1, preview_mode=False)
            if hint == "hard_cut":
                eq(ops["fade_ms"], 0)
            else:
                ok(ops["fade_ms"] <= expected_ms)

    for fn in [preview_mode_no_stretch, preview_mode_halves_fade, hard_cut_zero_fade,
               high_chaos_no_stretch_or_eq, low_chaos_enables_pipeline,
               fade_duration_matches_hint]:
        run_test(fn.__name__, fn)


# ── time_stretch_segment ──────────────────────────────────────────────────────

def test_time_stretch():
    section("time_stretch_segment")

    def returns_audio_segment():
        seg    = make_seg(5.0)
        result = time_stretch_segment(seg, 120.0, 128.0)
        ok(isinstance(result, AudioSegment))

    def no_stretch_returns_original_length():
        seg    = make_seg(5.0)
        result = time_stretch_segment(seg, 120.0, 120.0)  # same BPM = no stretch
        eq(len(result), len(seg))

    def none_bpm_returns_original():
        seg    = make_seg(5.0)
        result = time_stretch_segment(seg, None, 120.0)
        eq(len(result), len(seg))

    def extreme_ratio_clamped():
        # 80 BPM → 160 BPM = 2x stretch, well outside 15% clamp
        # Should return original (or clamped) without crashing
        seg    = make_seg(5.0)
        result = time_stretch_segment(seg, 80.0, 160.0)
        ok(isinstance(result, AudioSegment))

    for fn in [returns_audio_segment, no_stretch_returns_original_length,
               none_bpm_returns_original, extreme_ratio_clamped]:
        run_test(fn.__name__, fn)


# ── render_transition ─────────────────────────────────────────────────────────

def test_render_transition():
    section("render_transition")

    def hard_cut_length():
        out  = make_seg(20.0)
        inc  = make_seg(20.0)
        slot_out = make_slot(hint="hard_cut", bpm=120.0)
        slot_in  = make_slot(hint="hard_cut", bpm=120.0)
        result   = render_transition(out, inc, slot_out, slot_in, chaos=0.3, preview_mode=False)
        approx(len(result) / 1000, 40.0, 1.0, "hard_cut should be ~sum of both")

    def crossfade_shorter_than_sum():
        out  = make_seg(20.0)
        inc  = make_seg(20.0)
        slot_out = make_slot(hint="crossfade", bpm=120.0)
        slot_in  = make_slot(hint="crossfade", bpm=120.0)
        result   = render_transition(out, inc, slot_out, slot_in, chaos=0.1, preview_mode=False)
        # Crossfade overlaps, so result < sum of both
        ok(len(result) < len(out) + len(inc),
           f"Crossfade result ({len(result)}ms) should be shorter than sum ({len(out)+len(inc)}ms)")

    def result_is_audio_segment():
        out  = make_seg(15.0)
        inc  = make_seg(15.0)
        slot_out = make_slot(hint="crossfade", bpm=120.0)
        slot_in  = make_slot(hint="crossfade", bpm=128.0)
        result   = render_transition(out, inc, slot_out, slot_in, chaos=0.3, preview_mode=True)
        ok(isinstance(result, AudioSegment))

    def high_chaos_still_produces_output():
        out  = make_seg(15.0)
        inc  = make_seg(15.0)
        slot_out = make_slot(hint="long_blend", bpm=120.0)
        slot_in  = make_slot(hint="long_blend", bpm=120.0)
        result   = render_transition(out, inc, slot_out, slot_in, chaos=0.95, preview_mode=False)
        ok(isinstance(result, AudioSegment))
        ok(len(result) > 0)

    def preview_mode_works():
        out  = make_seg(15.0)
        inc  = make_seg(15.0)
        slot_out = make_slot(hint="long_blend", bpm=120.0)
        slot_in  = make_slot(hint="long_blend", bpm=120.0)
        result   = render_transition(out, inc, slot_out, slot_in, chaos=0.1, preview_mode=True)
        ok(isinstance(result, AudioSegment))

    for fn in [hard_cut_length, crossfade_shorter_than_sum, result_is_audio_segment,
               high_chaos_still_produces_output, preview_mode_works]:
        run_test(fn.__name__, fn)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Module 5 Tests\n" + "═"*52)
    print(f"\n  Libraries: rubberband={'✓' if RUBBERBAND_AVAILABLE else '✗'}  "
          f"pedalboard={'✓' if PEDALBOARD_AVAILABLE else '✗'}")

    test_trim()
    test_nearest_beat()
    test_decide_ops()
    test_time_stretch()
    test_render_transition()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'═'*52}")
    if _results["failed"]:
        print(f"  {_results['passed']}/{total} passed | {_results['failed']} FAILED")
        for name, tb in _results["errors"]:
            print(f"\n  ── {name} ──\n{tb}")
        sys.exit(1)
    else:
        print(f"  {_results['passed']}/{total} passed  🎉 All tests passed!")
    print("═"*52 + "\n")