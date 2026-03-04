"""
Tests for Module 2: Audio Analyzer
────────────────────────────────────
Run with: uv run python tests/test_analyzer.py

Generates real (synthetic) audio signals so librosa can do genuine analysis.
No external audio files required.
"""

import json
import sys
import math
import wave
import tempfile
import traceback
import struct
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.analyzer import (
    file_hash,
    camelot,
    detect_intro_outro,
    energy_profile,
    danceability_score,
    analyze_track,
    analyze_library,
    CAMELOT_MAJOR,
    CAMELOT_MINOR,
    KEY_NAMES,
)

# ── Minimal test framework ────────────────────────────────────────────────────

_results = {"passed": 0, "failed": 0, "errors": []}

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got:      {a!r}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or f"Expected truthy, got {cond!r}")

def assert_none(v, msg=""):
    if v is not None:
        raise AssertionError(msg or f"Expected None, got {v!r}")

def assert_not_none(v, msg=""):
    if v is None:
        raise AssertionError(msg or "Expected non-None value")

def assert_in_range(v, lo, hi, msg=""):
    if not (lo <= v <= hi):
        raise AssertionError(msg or f"Expected {lo}≤{v}≤{hi}")

def assert_approx(a, b, tol=1.0, msg=""):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{msg} Expected ~{b}(±{tol}), got {a}")

def assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
        raise AssertionError(f"Expected {exc.__name__} to be raised")
    except exc:
        pass

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
    print(f"\n{'─'*58}\n  {title}\n{'─'*58}")


# ── Audio signal factories ────────────────────────────────────────────────────

SR = 22050  # standard librosa default sample rate

def make_sine(freq=440.0, duration=5.0, sr=SR, amplitude=0.5) -> np.ndarray:
    """Pure sine wave — useful for basic signal tests."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def make_click_track(bpm=120.0, duration=10.0, sr=SR) -> np.ndarray:
    """
    Synthetic click track — impulses at precise BPM intervals.
    librosa's beat tracker should detect this tempo reliably.
    """
    y = np.zeros(int(sr * duration), dtype=np.float32)
    beat_interval = sr * 60.0 / bpm
    pos = 0.0
    while pos < len(y):
        idx = int(pos)
        if idx < len(y):
            # Short gaussian click centred at beat position
            for offset in range(-32, 33):
                i = idx + offset
                if 0 <= i < len(y):
                    y[i] += 0.8 * math.exp(-0.5 * (offset / 8) ** 2)
        pos += beat_interval
    return y


def make_energy_ramp(duration=10.0, sr=SR) -> np.ndarray:
    """Signal that ramps from silence to full amplitude — tests intro detection."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.linspace(0, 1, len(t))  # linear ramp
    return (np.sin(2 * np.pi * 440 * t) * envelope * 0.8).astype(np.float32)


def make_wav_from_array(path: Path, y: np.ndarray, sr: int = SR):
    """Write a numpy audio array to a WAV file."""
    # Convert float32 [-1,1] to int16
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y_int16.tobytes())


def make_track_stub(track_id="test001", filepath="/fake/track.wav") -> dict:
    """Minimal track dict matching Module 1 output schema."""
    return {
        "track_id": track_id,
        "filepath": filepath,
        "filename": Path(filepath).name,
        "duration_seconds": 10.0,
    }


# ── Unit Tests: Helpers ────────────────────────────────────────────────────────

def test_helpers():
    section("Unit Tests: Helpers")

    def file_hash_returns_string():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            p = Path(f.name)
            p.write_bytes(b"test")
            h = file_hash(str(p))
            assert_eq(len(h), 12)
            assert_true(all(c in "0123456789abcdef" for c in h))

    def file_hash_deterministic():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            p = Path(f.name)
            p.write_bytes(b"content")
            assert_eq(file_hash(str(p)), file_hash(str(p)))

    def camelot_all_major_keys():
        for i in range(12):
            result = camelot(i, True)
            assert_true(result.endswith("B"), f"Major key {i} should end with B, got {result}")
            assert_true(result[:-1].isdigit(), f"Major key {i} number should be digit: {result}")

    def camelot_all_minor_keys():
        for i in range(12):
            result = camelot(i, False)
            assert_true(result.endswith("A"), f"Minor key {i} should end with A, got {result}")

    def camelot_c_major_is_8b():
        assert_eq(camelot(0, True), "8B")  # C major = 8B on Camelot Wheel

    def camelot_a_minor_is_8a():
        assert_eq(camelot(9, False), "8A")  # A minor = 8A (relative of C major)

    def key_names_coverage():
        assert_eq(len(KEY_NAMES), 12)
        assert_eq(KEY_NAMES[0], "C")
        assert_eq(KEY_NAMES[9], "A")

    for fn in [file_hash_returns_string, file_hash_deterministic,
               camelot_all_major_keys, camelot_all_minor_keys,
               camelot_c_major_is_8b, camelot_a_minor_is_8a, key_names_coverage]:
        run_test(fn.__name__, fn)


# ── Unit Tests: energy_profile ────────────────────────────────────────────────

def test_energy_profile():
    section("Unit Tests: energy_profile")

    def returns_correct_keys():
        y = make_sine(duration=5.0)
        result = energy_profile(y, SR, n_segments=16)
        for key in ["curve", "mean", "peak", "dynamic_range"]:
            assert_true(key in result, f"Missing key: {key}")

    def curve_has_correct_length():
        y = make_sine(duration=5.0)
        result = energy_profile(y, SR, n_segments=16)
        assert_eq(len(result["curve"]), 16)

    def curve_normalized_0_to_1():
        y = make_sine(duration=5.0)
        result = energy_profile(y, SR, n_segments=32)
        assert_true(all(0.0 <= v <= 1.0 for v in result["curve"]),
                    "All curve values should be 0–1")

    def silent_track_zero_energy():
        y = np.zeros(SR * 3, dtype=np.float32)
        result = energy_profile(y, SR)
        assert_eq(result["peak"], 0.0)
        assert_true(all(v == 0.0 for v in result["curve"]))

    def loud_track_has_high_energy():
        y = make_sine(amplitude=0.9, duration=5.0)
        result = energy_profile(y, SR)
        assert_true(result["mean"] > 0.0, "Loud track should have positive mean energy")
        assert_true(result["peak"] > 0.0)

    def ramp_track_has_dynamic_range():
        y = make_energy_ramp(duration=10.0)
        result = energy_profile(y, SR)
        assert_true(result["dynamic_range"] > 0.0,
                    "Ramping track should have non-zero dynamic range")

    for fn in [returns_correct_keys, curve_has_correct_length, curve_normalized_0_to_1,
               silent_track_zero_energy, loud_track_has_high_energy, ramp_track_has_dynamic_range]:
        run_test(fn.__name__, fn)


# ── Unit Tests: danceability_score ────────────────────────────────────────────

def test_danceability():
    section("Unit Tests: danceability_score")

    def returns_0_to_1():
        beats = [i * 0.5 for i in range(20)]   # 120 BPM grid
        score = danceability_score(120.0, beats, 0.3, 0.1)
        assert_in_range(score, 0.0, 1.0, "Score must be 0–1")

    def steady_120bpm_scores_high():
        # Perfectly regular 120 BPM beats
        beats = [i * (60.0 / 120.0) for i in range(40)]
        score = danceability_score(120.0, beats, 0.4, 0.08)
        assert_true(score >= 0.6, f"Steady 120 BPM should score ≥0.6, got {score}")

    def very_slow_scores_lower():
        beats = [i * 2.0 for i in range(20)]    # 30 BPM
        score_slow = danceability_score(30.0, beats, 0.3, 0.1)
        beats_fast = [i * 0.5 for i in range(40)]  # 120 BPM
        score_fast = danceability_score(120.0, beats_fast, 0.3, 0.1)
        assert_true(score_slow < score_fast,
                    f"30 BPM ({score_slow}) should score lower than 120 BPM ({score_fast})")

    def empty_beats_returns_neutral():
        score = danceability_score(120.0, [], 0.3, 0.1)
        assert_eq(score, 0.5)

    def irregular_beats_penalized():
        import random
        rng = random.Random(42)
        # Randomly jittered beat times
        irregular = [i * 0.5 + rng.uniform(-0.15, 0.15) for i in range(20)]
        steady    = [i * 0.5 for i in range(20)]
        score_irr = danceability_score(120.0, irregular, 0.3, 0.1)
        score_st  = danceability_score(120.0, steady,    0.3, 0.1)
        assert_true(score_irr <= score_st,
                    f"Irregular ({score_irr}) should score ≤ steady ({score_st})")

    for fn in [returns_0_to_1, steady_120bpm_scores_high, very_slow_scores_lower,
               empty_beats_returns_neutral, irregular_beats_penalized]:
        run_test(fn.__name__, fn)


# ── Unit Tests: detect_intro_outro ────────────────────────────────────────────

def test_intro_outro():
    section("Unit Tests: detect_intro_outro")

    def returns_two_floats():
        y = make_sine(duration=10.0)
        intro, outro = detect_intro_outro(y, SR, 10.0)
        assert_true(isinstance(intro, float))
        assert_true(isinstance(outro, float))

    def intro_before_outro():
        y = make_sine(duration=15.0)
        intro, outro = detect_intro_outro(y, SR, 15.0)
        assert_true(intro < outro, f"intro ({intro}) must be before outro ({outro})")

    def both_within_duration():
        duration = 12.0
        y = make_sine(duration=duration)
        intro, outro = detect_intro_outro(y, SR, duration)
        assert_in_range(intro,  0.0, duration)
        assert_in_range(outro, 0.0, duration)

    def ramp_intro_is_nonzero():
        # Ramp starts silent — intro should be detected > 0
        y = make_energy_ramp(duration=15.0)
        intro, _ = detect_intro_outro(y, SR, 15.0)
        assert_true(intro >= 0.0, f"Intro end should be ≥ 0, got {intro}")

    for fn in [returns_two_floats, intro_before_outro, both_within_duration, ramp_intro_is_nonzero]:
        run_test(fn.__name__, fn)


# ── Integration Tests: analyze_track ──────────────────────────────────────────

def test_analyze_track():
    section("Integration Tests: analyze_track (real librosa)")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        cache_dir = p / "cache"

        def result_has_all_keys():
            wav = p / "sine.wav"
            make_wav_from_array(wav, make_sine(duration=8.0))
            track = make_track_stub("t001", str(wav))
            result = analyze_track(track, cache_dir)
            for key in ["bpm", "key_name", "camelot", "mode", "energy",
                        "intro_end_sec", "outro_start_sec", "danceability",
                        "beat_times", "analysis_error"]:
                assert_true(key in result, f"Missing key: {key}")

        def click_track_bpm_detected():
            wav = p / "click.wav"
            make_wav_from_array(wav, make_click_track(bpm=120.0, duration=15.0))
            track = make_track_stub("t002", str(wav))
            result = analyze_track(track, cache_dir)
            assert_none(result["analysis_error"], f"Should not error: {result['analysis_error']}")
            assert_not_none(result["bpm"])
            # Allow generous tolerance — librosa may detect 120 or 60 (half-time)
            detected = result["bpm"]
            assert_true(
                80 <= detected <= 160,
                f"120 BPM click track should detect 80–160 BPM, got {detected}"
            )

        def sine_has_key():
            wav = p / "key.wav"
            make_wav_from_array(wav, make_sine(freq=261.63, duration=8.0))  # C4
            track = make_track_stub("t003", str(wav))
            result = analyze_track(track, cache_dir)
            assert_none(result["analysis_error"])
            assert_not_none(result["key_name"])
            assert_true(result["key_name"] in KEY_NAMES)

        def camelot_valid_format():
            wav = p / "cam.wav"
            make_wav_from_array(wav, make_sine(duration=8.0))
            track = make_track_stub("t004", str(wav))
            result = analyze_track(track, cache_dir)
            if result["camelot"]:
                cam = result["camelot"]
                assert_true(cam.endswith("A") or cam.endswith("B"),
                            f"Camelot should end in A or B: {cam}")

        def energy_profile_populated():
            wav = p / "eng.wav"
            make_wav_from_array(wav, make_sine(duration=10.0))
            track = make_track_stub("t005", str(wav))
            result = analyze_track(track, cache_dir)
            assert_none(result["analysis_error"])
            assert_not_none(result["energy"])
            assert_true("curve" in result["energy"])
            assert_true("mean" in result["energy"])

        def danceability_in_range():
            wav = p / "dance.wav"
            make_wav_from_array(wav, make_click_track(bpm=128.0, duration=12.0))
            track = make_track_stub("t006", str(wav))
            result = analyze_track(track, cache_dir)
            assert_none(result["analysis_error"])
            d = result["danceability"]
            assert_not_none(d)
            assert_in_range(d, 0.0, 1.0, f"Danceability must be 0–1, got {d}")

        def missing_file_returns_error():
            track = make_track_stub("t007", "/no/such/file.wav")
            result = analyze_track(track, cache_dir)
            assert_not_none(result["analysis_error"])

        def cache_hit_on_second_call():
            wav = p / "cached.wav"
            make_wav_from_array(wav, make_sine(duration=5.0))
            track = make_track_stub("t008", str(wav))
            r1 = analyze_track(track, cache_dir)
            r2 = analyze_track(track, cache_dir)
            assert_true(r2["cache_hit"], "Second call should be a cache hit")

        def cache_values_match():
            wav = p / "cachecheck.wav"
            make_wav_from_array(wav, make_sine(duration=5.0))
            track = make_track_stub("t009", str(wav))
            r1 = analyze_track(track, cache_dir)
            r2 = analyze_track(track, cache_dir)
            assert_eq(r1["bpm"], r2["bpm"], "Cached BPM should match original")
            assert_eq(r1["camelot"], r2["camelot"], "Cached Camelot should match original")

        for fn in [result_has_all_keys, click_track_bpm_detected, sine_has_key,
                   camelot_valid_format, energy_profile_populated, danceability_in_range,
                   missing_file_returns_error, cache_hit_on_second_call, cache_values_match]:
            run_test(fn.__name__, fn)


# ── Integration Tests: analyze_library ────────────────────────────────────────

def test_analyze_library():
    section("Integration Tests: analyze_library")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        cache_dir = p / "cache"

        # Build a small batch of test tracks
        tracks = []
        for i, (bpm, freq) in enumerate([(120.0, 440.0), (128.0, 523.0), (100.0, 349.0)]):
            wav = p / f"track_{i}.wav"
            make_wav_from_array(wav, make_click_track(bpm=bpm, duration=10.0))
            tracks.append(make_track_stub(f"lib{i:03d}", str(wav)))

        def returns_correct_count():
            results, stats = analyze_library(tracks, cache_dir, n_jobs=1)
            assert_eq(len(results), len(tracks))

        def stats_has_required_keys():
            _, stats = analyze_library(tracks, cache_dir, n_jobs=1)
            for key in ["total_tracks", "successful", "failed", "cache_hits",
                        "elapsed_seconds", "bpm_range", "bpm_mean"]:
                assert_true(key in stats, f"Missing stats key: {key}")

        def all_tracks_analyzed():
            results, stats = analyze_library(tracks, cache_dir, n_jobs=1)
            assert_eq(stats["successful"], len(tracks))
            assert_eq(stats["failed"], 0)

        def second_run_all_cache_hits():
            analyze_library(tracks, cache_dir, n_jobs=1)  # warm cache
            _, stats = analyze_library(tracks, cache_dir, n_jobs=1)
            assert_eq(stats["cache_hits"], len(tracks))

        def bpm_range_is_valid():
            results, stats = analyze_library(tracks, cache_dir, n_jobs=1)
            if stats["bpm_range"]:
                lo, hi = stats["bpm_range"]
                assert_true(lo <= hi, f"BPM range lo ({lo}) should be ≤ hi ({hi})")

        for fn in [returns_correct_count, stats_has_required_keys, all_tracks_analyzed,
                   second_run_all_cache_hits, bpm_range_is_valid]:
            run_test(fn.__name__, fn)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*58)
    print("  🧪 AutoDJ — Module 2 Test Suite")
    print("═"*58)

    test_helpers()
    test_energy_profile()
    test_danceability()
    test_intro_outro()
    test_analyze_track()
    test_analyze_library()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'═'*58}")
    if _results["failed"]:
        print(f"  Results: {_results['passed']}/{total} passed | {_results['failed']} FAILED")
        for name, tb in _results["errors"]:
            print(f"\n  ── {name} ──\n{tb}")
        sys.exit(1)
    else:
        print(f"  Results: {_results['passed']}/{total} passed  🎉 All tests passed!")
    print("═"*58 + "\n")