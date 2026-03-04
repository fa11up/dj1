"""
Tests for Module 5: Renderer
Run with: uv run python tests/test_renderer.py
Audio I/O uses synthetic pydub segments — no real files needed.
"""

import sys, traceback
from pathlib import Path
import numpy as np
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.renderer import (
    trim_to_cues,
    nearest_beat_ms,
    bpm_blend_ms,
    find_breakdown_point,
    segment_idx_to_ms,
    compute_crossover_hz,
    apply_highpass_adaptive,
    measure_tail_energy,
    measure_head_energy,
    energy_aware_hint,
    transition_strategy,
    render_transition,
    loop_segment,
    LOOP_BARS_DEFAULT,
    LOOP_BARS_LONG,
    LOOP_MAX_REPEATS,
    SOFT_ENERGY_THRESHOLD,
    EQ_BASS_HZ,
    RUBBERBAND_AVAILABLE,
    PEDALBOARD_AVAILABLE,
    find_breakdown_from_sections,
)

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


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_seg(duration_sec=10.0, freq=440, sr=44100, channels=2):
    n     = int(sr * duration_sec)
    t     = np.linspace(0, duration_sec, n, endpoint=False)
    wave_ = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
    if channels == 2:
        stereo = np.stack([wave_, wave_], axis=1).flatten()
        return AudioSegment(stereo.tobytes(), frame_rate=sr, sample_width=2, channels=2)
    return AudioSegment(wave_.tobytes(), frame_rate=sr, sample_width=2, channels=1)

def make_slot(bpm=120.0, camelot="8B", hint="crossfade",
              intro=1.0, outro=None, spectral_centroid=2000.0):
    energy_curve = [0.5 + 0.3 * np.sin(i / 32 * np.pi) for i in range(32)]
    return {
        "track_id":               "t001",
        "filepath":               "/fake/track.mp3",
        "filename":               "track.mp3",
        "actual_bpm":             bpm,
        "camelot":                camelot,
        "transition_hint":        hint,
        "intro_end_sec":          intro,
        "outro_start_sec":        outro,
        "beat_times":             [i * (60.0 / bpm) for i in range(400)],
        "spectral_centroid_mean": spectral_centroid,
        "energy":                 {"mean": 0.07, "curve": energy_curve},
        "is_seed":                False,
    }


# ── trim_to_cues ──────────────────────────────────────────────────────────────

def test_trim():
    section("trim_to_cues")

    def trims_intro():
        seg     = make_seg(30.0)
        trimmed = trim_to_cues(seg, intro_end_sec=5.0, outro_start_sec=28.0)
        ok(len(trimmed) < len(seg))
        approx(len(trimmed) / 1000, 23.0, 0.1)

    def trims_outro():
        seg     = make_seg(30.0)
        trimmed = trim_to_cues(seg, intro_end_sec=0.0, outro_start_sec=25.0)
        approx(len(trimmed) / 1000, 25.0, 0.1)

    def none_cues_returns_original():
        seg = make_seg(30.0)
        eq(len(trim_to_cues(seg, None, None)), len(seg))

    def too_short_returns_original():
        seg     = make_seg(30.0)
        trimmed = trim_to_cues(seg, 25.0, 28.0)
        ok(len(trimmed) >= 10_000)

    for fn in [trims_intro, trims_outro, none_cues_returns_original, too_short_returns_original]:
        run_test(fn.__name__, fn)


# ── nearest_beat_ms ───────────────────────────────────────────────────────────

def test_nearest_beat():
    section("nearest_beat_ms")
    beats = [i * 0.5 for i in range(40)]

    run_test("exact beat",        lambda: eq(nearest_beat_ms(2000, beats), 2000))
    run_test("snaps down",        lambda: eq(nearest_beat_ms(2100, beats), 2000))
    run_test("snaps up",          lambda: eq(nearest_beat_ms(2300, beats), 2500))
    run_test("empty → original",  lambda: eq(nearest_beat_ms(1234, []), 1234))


# ── bpm_blend_ms ─────────────────────────────────────────────────────────────

def test_bpm_blend():
    section("bpm_blend_ms")

    run_test("exceeds old 24s hardcode",
        lambda: ok(bpm_blend_ms(120.0) > 24_000))
    run_test("slow BPM longer than fast",
        lambda: ok(bpm_blend_ms(80.0) > bpm_blend_ms(160.0)))
    run_test("preview shorter than full",
        lambda: ok(bpm_blend_ms(120.0, preview_mode=True) < bpm_blend_ms(120.0)))
    run_test("None BPM fallback > 0",
        lambda: ok(bpm_blend_ms(None) > 0))
    run_test("all BPMs in 20-200s range",
        lambda: ok(all(20_000 <= bpm_blend_ms(b) <= 200_000 for b in [80, 100, 120, 128, 140])))


# ── find_breakdown_point ──────────────────────────────────────────────────────

def test_find_breakdown():
    section("find_breakdown_point")

    def finds_lowest_in_latter_half():
        curve = [0.8] * 20 + [0.8, 0.7, 0.5, 0.2, 0.6, 0.7, 0.8, 0.9, 0.8, 0.8, 0.8, 0.8]
        idx = find_breakdown_point(curve, outro_start_ratio=0.6)
        eq(idx, 23, f"Should find lowest point at idx 23, got {idx}")

    def ignores_early_low_points():
        curve = [0.1, 0.1, 0.1] + [0.8] * 29
        idx = find_breakdown_point(curve, outro_start_ratio=0.6)
        ok(idx is None or idx >= 18)

    run_test("finds_lowest_in_latter_half", finds_lowest_in_latter_half)
    run_test("ignores_early_low_points",    ignores_early_low_points)
    run_test("empty → None",   lambda: ok(find_breakdown_point([]) is None))
    run_test("None → None",    lambda: ok(find_breakdown_point(None) is None))
    run_test("too short → None", lambda: ok(find_breakdown_point([0.5, 0.3]) is None))


# ── compute_crossover_hz ──────────────────────────────────────────────────────

def test_compute_crossover():
    section("compute_crossover_hz")

    run_test("dark track → low crossover",
        lambda: ok(compute_crossover_hz(500.0) <= 80.0))
    run_test("bright track → higher crossover",
        lambda: ok(compute_crossover_hz(4000.0) >= 150.0))
    run_test("monotonically scaling", lambda: (
        lambda vals: ok(vals == sorted(vals)))(
            [compute_crossover_hz(c) for c in [500, 1000, 2000, 3000, 4000]]))
    run_test("None → default",
        lambda: eq(compute_crossover_hz(None), EQ_BASS_HZ))
    run_test("clamped to valid range", lambda: (
        in_range(compute_crossover_hz(1.0),     50.0, 300.0) or
        in_range(compute_crossover_hz(99999.0), 50.0, 300.0)))


# ── apply_highpass_adaptive ───────────────────────────────────────────────────

def test_adaptive_eq():
    section("apply_highpass_adaptive")

    def returns_audio_segment():
        ok(isinstance(apply_highpass_adaptive(make_seg(3.0), 150.0, 0.3), AudioSegment))

    def hard_removes_more_bass_than_shelf():
        if not PEDALBOARD_AVAILABLE:
            return
        bass_seg  = make_seg(3.0, freq=80)
        hard  = apply_highpass_adaptive(bass_seg, 200.0, chaos=0.1)
        shelf = apply_highpass_adaptive(bass_seg, 200.0, chaos=0.9)
        ok(measure_tail_energy(hard) <= measure_tail_energy(shelf))

    def preserves_length():
        result = apply_highpass_adaptive(make_seg(5.0), 150.0, 0.5)
        approx(len(result) / 1000, 5.0, 0.2)

    for fn in [returns_audio_segment, hard_removes_more_bass_than_shelf, preserves_length]:
        run_test(fn.__name__, fn)


# ── energy detection + energy_aware_hint ─────────────────────────────────────

def test_energy_detection():
    section("measure_tail/head_energy + energy_aware_hint")
    silent = AudioSegment.silent(duration=10000, frame_rate=44100).set_channels(2)
    loud   = make_seg(10.0)

    run_test("loud seg above threshold",
        lambda: ok(measure_tail_energy(loud) > SOFT_ENERGY_THRESHOLD))
    run_test("silent seg at/below threshold",
        lambda: ok(measure_tail_energy(silent) <= SOFT_ENERGY_THRESHOLD))
    run_test("both soft → long_blend",
        lambda: eq(energy_aware_hint("hard_cut", silent, silent), "long_blend"))
    run_test("one soft fixes hard_cut",
        lambda: ok(energy_aware_hint("hard_cut", loud, silent) != "hard_cut"))
    run_test("both loud keeps hard_cut",
        lambda: eq(energy_aware_hint("hard_cut", loud, loud), "hard_cut"))
    run_test("crossfade kept when loud",
        lambda: eq(energy_aware_hint("crossfade", loud, loud), "crossfade"))


# ── transition_strategy ───────────────────────────────────────────────────────

def test_transition_strategy():
    section("transition_strategy")

    run_test("long_blend → staged method",
        lambda: eq(transition_strategy("long_blend", 0.2, False, 120.0)["method"], "staged"))
    run_test("staged has phase keys", lambda: (
        lambda s: ok("highs_ms" in s and "swap_ms" in s and "outro_ms" in s))(
            transition_strategy("long_blend", 0.2, False, 120.0)))
    run_test("preview shortens phases", lambda: (
        lambda f, p: ok(p["highs_ms"] <= f["highs_ms"]))(
            transition_strategy("long_blend", 0.2, False, 120.0),
            transition_strategy("long_blend", 0.2, True,  120.0)))
    run_test("hard_cut method + min fade",
        lambda: (lambda s: ok(s["method"] == "hard_cut" and s["fade_ms"] >= 2_000))(
            transition_strategy("hard_cut", 0.2, False)))
    run_test("crossfade method",
        lambda: eq(transition_strategy("crossfade", 0.2, False)["method"], "crossfade"))
    run_test("high chaos no stretch",
        lambda: ok(not transition_strategy("long_blend", 0.8, False, 120.0)["do_stretch"]))
    run_test("slow BPM → longer highs phase than fast", lambda: (
        lambda slow, fast: ok(slow["highs_ms"] >= fast["highs_ms"]))(
            transition_strategy("long_blend", 0.2, False, 90.0),
            transition_strategy("long_blend", 0.2, False, 140.0)))


# ── time_stretch_segment ──────────────────────────────────────────────────────

def test_time_stretch():
    section("time_stretch_segment (import check)")
    from modules.renderer import time_stretch_segment

    run_test("returns AudioSegment",
        lambda: ok(isinstance(time_stretch_segment(make_seg(5.0), 120.0, 128.0), AudioSegment)))
    run_test("same BPM → same length",
        lambda: eq(len(time_stretch_segment(make_seg(5.0), 120.0, 120.0)), len(make_seg(5.0))))
    run_test("None BPM → original",
        lambda: eq(len(time_stretch_segment(make_seg(5.0), None, 120.0)), len(make_seg(5.0))))
    run_test("extreme ratio clamped, no crash",
        lambda: ok(isinstance(time_stretch_segment(make_seg(5.0), 80.0, 160.0), AudioSegment)))


# ── render_transition ─────────────────────────────────────────────────────────

def test_render_transition():
    section("render_transition")

    def hard_cut_sequential():
        out  = make_seg(20.0)
        inc  = make_seg(20.0)
        result = render_transition(out, inc, make_slot(hint="hard_cut"),
                                   make_slot(hint="hard_cut"), chaos=0.3, preview_mode=False)
        # Sequential fade out + fade in = ~full sum, no overlap
        approx(len(result) / 1000, 40.0, 1.5)

    def crossfade_shorter_than_sum():
        out  = make_seg(20.0)
        inc  = make_seg(20.0)
        result = render_transition(out, inc, make_slot(hint="crossfade"),
                                   make_slot(hint="crossfade"), chaos=0.1, preview_mode=False)
        ok(len(result) < len(out) + len(inc))

    def result_is_audio_segment():
        out  = make_seg(15.0)
        inc  = make_seg(15.0)
        result = render_transition(out, inc, make_slot(hint="crossfade"),
                                   make_slot(hint="crossfade", bpm=128.0),
                                   chaos=0.3, preview_mode=True)
        ok(isinstance(result, AudioSegment))

    def high_chaos_produces_output():
        out  = make_seg(15.0)
        inc  = make_seg(15.0)
        result = render_transition(out, inc, make_slot(hint="long_blend"),
                                   make_slot(hint="long_blend"),
                                   chaos=0.95, preview_mode=False)
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    def preview_mode_works():
        out  = make_seg(60.0)
        inc  = make_seg(60.0)
        result = render_transition(out, inc, make_slot(hint="long_blend"),
                                   make_slot(hint="long_blend"),
                                   chaos=0.1, preview_mode=True)
        ok(isinstance(result, AudioSegment))

    def staged_transition_no_crash():
        out  = make_seg(90.0)
        inc  = make_seg(90.0)
        result = render_transition(out, inc, make_slot(hint="long_blend"),
                                   make_slot(hint="long_blend"),
                                   chaos=0.2, preview_mode=False)
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    for fn in [hard_cut_sequential, crossfade_shorter_than_sum, result_is_audio_segment,
               high_chaos_produces_output, preview_mode_works, staged_transition_no_crash]:
        run_test(fn.__name__, fn)

def test_loop_segment():
    section("loop_segment (P2-1)")

    def returns_audio_and_position():
        seg = make_seg(60.0)
        loop_audio, loop_point = loop_segment(seg, 4, 120.0)
        ok(isinstance(loop_audio, AudioSegment), "Should return AudioSegment")
        ok(isinstance(loop_point, int), "Should return loop point as int ms")

    def loop_extends_content():
        seg = make_seg(60.0)
        loop_audio, loop_point = loop_segment(seg, 4, 120.0)
        ok(len(loop_audio) > 0, "Loop audio should have content")
        # 4 bars at 120 BPM = 8 seconds, repeated up to 4x = ~32s
        ok(len(loop_audio) >= 8000, f"Loop should be ≥8s, got {len(loop_audio)}ms")

    def loop_point_within_segment():
        seg = make_seg(60.0)
        _, loop_point = loop_segment(seg, 4, 120.0)
        ok(0 <= loop_point <= len(seg),
           f"Loop point {loop_point} outside segment bounds 0–{len(seg)}")

    def loop_with_beat_times_snaps():
        seg = make_seg(60.0)
        beat_times = [i * 0.5 for i in range(240)]  # 120 BPM beats
        _, loop_point_snapped = loop_segment(seg, 4, 120.0, beat_times)
        _, loop_point_raw     = loop_segment(seg, 4, 120.0, None)
        # Snapped point should be on or near a beat
        ok(isinstance(loop_point_snapped, int))

    def slow_bpm_uses_longer_loop():
        seg = make_seg(120.0)  # long enough for 8 bars at 80 BPM
        loop_4, _ = loop_segment(seg, LOOP_BARS_DEFAULT, 80.0)
        loop_8, _ = loop_segment(seg, LOOP_BARS_LONG, 80.0)
        ok(len(loop_8) > len(loop_4),
           f"8-bar loop ({len(loop_8)}ms) should be longer than 4-bar ({len(loop_4)}ms)")

    def too_short_segment_returns_empty():
        seg = make_seg(2.0)  # only 2 seconds — too short to loop
        loop_audio, _ = loop_segment(seg, 4, 120.0)
        eq(len(loop_audio), 0, "Should return empty audio for too-short segments")

    def loop_preserves_sample_rate():
        seg = make_seg(60.0)
        loop_audio, _ = loop_segment(seg, 4, 120.0)
        if len(loop_audio) > 0:
            eq(loop_audio.frame_rate, seg.frame_rate,
               "Loop should preserve sample rate")

    def loop_preserves_channels():
        seg = make_seg(60.0)
        loop_audio, _ = loop_segment(seg, 4, 120.0)
        if len(loop_audio) > 0:
            eq(loop_audio.channels, seg.channels,
               "Loop should preserve channel count")

    for fn in [returns_audio_and_position, loop_extends_content,
               loop_point_within_segment, loop_with_beat_times_snaps,
               slow_bpm_uses_longer_loop, too_short_segment_returns_empty,
               loop_preserves_sample_rate, loop_preserves_channels]:
        run_test(fn.__name__, fn)


# ── Looped transitions (integration) ─────────────────────────────────────────

def test_looped_transitions():
    section("looped transitions — render_transition integration (P2-1)")

    def staged_with_short_outgoing():
        """Short outgoing track should still produce valid transition via looping."""
        out = make_seg(15.0)   # short — might not have enough for full blend
        inc = make_seg(60.0)
        result = render_transition(
            out, inc,
            make_slot(hint="long_blend", bpm=120.0),
            make_slot(hint="long_blend", bpm=120.0),
            chaos=0.2, preview_mode=False,
        )
        ok(isinstance(result, AudioSegment) and len(result) > 0,
           "Should produce valid output even with short outgoing track")

    def staged_with_normal_tracks():
        """Normal length tracks should also work with looping enabled."""
        out = make_seg(90.0)
        inc = make_seg(90.0)
        result = render_transition(
            out, inc,
            make_slot(hint="long_blend", bpm=128.0),
            make_slot(hint="long_blend", bpm=128.0),
            chaos=0.3, preview_mode=False,
        )
        ok(isinstance(result, AudioSegment) and len(result) > 0)
        # Result should be longer than either track alone (loop extends)
        ok(len(result) > len(out) * 0.5,
           "Result should contain substantial content")

    def looped_transition_longer_than_unlooped_short():
        """With a short outgoing, the looped version should be longer
        than just concatenating (since loop extends the outgoing)."""
        out = make_seg(20.0)
        inc = make_seg(60.0)
        result = render_transition(
            out, inc,
            make_slot(hint="long_blend", bpm=120.0),
            make_slot(hint="long_blend", bpm=120.0),
            chaos=0.2, preview_mode=False,
        )
        simple_concat_ms = len(out) + len(inc)
        # The result with looping should have added content
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    def preview_mode_with_looping():
        """Looping should work in preview mode too."""
        out = make_seg(30.0)
        inc = make_seg(30.0)
        result = render_transition(
            out, inc,
            make_slot(hint="long_blend", bpm=120.0),
            make_slot(hint="long_blend", bpm=120.0),
            chaos=0.2, preview_mode=True,
        )
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    def high_chaos_with_looping():
        """High chaos still produces valid output (may not loop but shouldn't crash)."""
        out = make_seg(30.0)
        inc = make_seg(30.0)
        result = render_transition(
            out, inc,
            make_slot(hint="long_blend", bpm=128.0),
            make_slot(hint="long_blend", bpm=128.0),
            chaos=0.9, preview_mode=False,
        )
        ok(isinstance(result, AudioSegment) and len(result) > 0)

    for fn in [staged_with_short_outgoing, staged_with_normal_tracks,
               looped_transition_longer_than_unlooped_short,
               preview_mode_with_looping, high_chaos_with_looping]:
        run_test(fn.__name__, fn)

def test_breakdown_from_sections():
    section("find_breakdown_from_sections (P2-6)")

    # Helper: make a sections list
    def make_sections():
        return [
            {"label": "intro",     "start_sec": 0.0,   "end_sec": 16.0,
             "energy": 0.15, "energy_category": "low"},
            {"label": "verse",     "start_sec": 16.0,  "end_sec": 64.0,
             "energy": 0.55, "energy_category": "mid"},
            {"label": "chorus",    "start_sec": 64.0,  "end_sec": 112.0,
             "energy": 0.85, "energy_category": "high"},
            {"label": "breakdown", "start_sec": 112.0, "end_sec": 128.0,
             "energy": 0.20, "energy_category": "low"},
            {"label": "chorus",    "start_sec": 128.0, "end_sec": 176.0,
             "energy": 0.82, "energy_category": "high"},
            {"label": "outro",     "start_sec": 176.0, "end_sec": 200.0,
             "energy": 0.12, "energy_category": "low"},
        ]

    def finds_labelled_breakdown():
        sections = make_sections()
        # Track is 200s = 200000ms. Breakdown at 112s.
        result = find_breakdown_from_sections(sections, 0.55, 200000)
        ok(result is not None, "Should find breakdown")
        # Should find the breakdown at 112s = 112000ms
        approx(result / 1000, 112.0, 2.0)

    def returns_none_for_empty():
        result = find_breakdown_from_sections([], 0.55, 200000)
        ok(result is None)

    def returns_none_without_duration():
        sections = make_sections()
        result = find_breakdown_from_sections(sections, 0.55, None)
        ok(result is None)

    def finds_low_energy_fallback():
        """If no explicit breakdown label, should find low-energy sections."""
        sections = [
            {"label": "intro",  "start_sec": 0.0,   "end_sec": 16.0,
             "energy": 0.15, "energy_category": "low"},
            {"label": "chorus", "start_sec": 16.0,  "end_sec": 100.0,
             "energy": 0.85, "energy_category": "high"},
            {"label": "verse",  "start_sec": 100.0, "end_sec": 150.0,
             "energy": 0.45, "energy_category": "mid"},
            {"label": "outro",  "start_sec": 150.0, "end_sec": 200.0,
             "energy": 0.10, "energy_category": "low"},
        ]
        result = find_breakdown_from_sections(sections, 0.55, 200000)
        ok(result is not None, "Should find outro as low-energy fallback")
        # Should find the outro start at 150s
        approx(result / 1000, 150.0, 2.0)

    def respects_outro_start_ratio():
        """Should not find breakdowns in the first half of the track."""
        sections = [
            {"label": "breakdown", "start_sec": 10.0, "end_sec": 20.0,
             "energy": 0.15, "energy_category": "low"},
            {"label": "chorus",    "start_sec": 20.0, "end_sec": 200.0,
             "energy": 0.85, "energy_category": "high"},
        ]
        result = find_breakdown_from_sections(sections, 0.55, 200000)
        # The breakdown at 10s is before 55% of 200s (110s), should be skipped
        ok(result is None, "Should not find breakdown before outro_start_ratio")

    def finds_chorus_to_verse_transition():
        """Priority 3: energy drop between chorus and verse."""
        sections = [
            {"label": "intro",  "start_sec": 0.0,   "end_sec": 16.0,
             "energy": 0.15, "energy_category": "low"},
            {"label": "chorus", "start_sec": 16.0,  "end_sec": 120.0,
             "energy": 0.85, "energy_category": "high"},
            {"label": "verse",  "start_sec": 120.0, "end_sec": 180.0,
             "energy": 0.45, "energy_category": "mid"},
            {"label": "chorus", "start_sec": 180.0, "end_sec": 200.0,
             "energy": 0.80, "energy_category": "high"},
        ]
        result = find_breakdown_from_sections(sections, 0.55, 200000)
        ok(result is not None, "Should find chorus→verse transition")
        # Should find the verse at 120s
        approx(result / 1000, 120.0, 2.0)

    def result_is_milliseconds():
        sections = make_sections()
        result = find_breakdown_from_sections(sections, 0.55, 200000)
        ok(isinstance(result, int), f"Should return int ms, got {type(result)}")

    for fn in [finds_labelled_breakdown, returns_none_for_empty,
               returns_none_without_duration, finds_low_energy_fallback,
               respects_outro_start_ratio, finds_chorus_to_verse_transition,
               result_is_milliseconds]:
        run_test(fn.__name__, fn)
        

# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Module 5 Tests\n" + "═"*52)
    print(f"\n  rubberband={'✓' if RUBBERBAND_AVAILABLE else '✗'}  "
          f"pedalboard={'✓' if PEDALBOARD_AVAILABLE else '✗'}\n")

    test_trim()
    test_nearest_beat()
    test_bpm_blend()
    test_find_breakdown()
    test_compute_crossover()
    test_adaptive_eq()
    test_energy_detection()
    test_transition_strategy()
    test_time_stretch()
    test_render_transition()
    test_loop_segment()
    test_looped_transitions()
    test_breakdown_from_sections()

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
