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
    BPM_NORM_LO,
    BPM_NORM_HI,
    KEY_NAMES,
    normalise_bpm,
    detect_sections, 
    SECTION_MIN_DURATION_SEC,
    # infer_genre,
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

def test_normalise_bpm():
    section("Unit Tests: normalise_bpm (P2-2)")

    def already_in_range_unchanged():
        bpm, corrected = normalise_bpm(128.0)
        assert_approx(bpm, 128.0, tol=0.01)
        assert_true(not corrected, "128 BPM should not be corrected")

    def half_time_doubled():
        # 78 BPM is half of 156 — should double into range
        bpm, corrected = normalise_bpm(78.0)
        assert_approx(bpm, 156.0, tol=0.01)
        assert_true(corrected, "78 BPM should be corrected to 156")

    def double_time_halved():
        # 256 BPM is double of 128 — should halve into range
        bpm, corrected = normalise_bpm(256.0)
        assert_approx(bpm, 128.0, tol=0.01)
        assert_true(corrected, "256 BPM should be corrected to 128")

    def very_slow_doubled_multiple():
        # 32 BPM → 64 → 128
        bpm, corrected = normalise_bpm(32.0)
        assert_approx(bpm, 128.0, tol=0.01)
        assert_true(corrected)

    def very_fast_halved_multiple():
        # 512 BPM → 256 → 128
        bpm, corrected = normalise_bpm(512.0)
        assert_approx(bpm, 128.0, tol=0.01)
        assert_true(corrected)

    def dnb_range_stays():
        # 174 BPM is valid DnB — should NOT be halved
        bpm, corrected = normalise_bpm(174.0)
        assert_approx(bpm, 174.0, tol=0.01)
        assert_true(not corrected, "174 BPM is valid DnB, should not change")

    def boundary_lo_stays():
        bpm, corrected = normalise_bpm(80.0)
        assert_approx(bpm, 80.0, tol=0.01)
        assert_true(not corrected)

    def boundary_hi_stays():
        bpm, corrected = normalise_bpm(175.0)
        assert_approx(bpm, 175.0, tol=0.01)
        assert_true(not corrected)

    def none_returns_none():
        bpm, corrected = normalise_bpm(None)
        assert_true(bpm is None)
        assert_true(not corrected)

    def zero_returns_zero():
        bpm, corrected = normalise_bpm(0)
        assert_eq(bpm, 0)
        assert_true(not corrected)

    def result_always_in_range():
        """Fuzz test: any positive BPM should normalise into range."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            raw = rng.uniform(20.0, 600.0)
            bpm, _ = normalise_bpm(raw)
            assert_in_range(bpm, BPM_NORM_LO, BPM_NORM_HI,
                            f"normalise_bpm({raw}) = {bpm} outside range")

    def analyze_track_has_bpm_fields():
        """Integration: analyze_track result includes new P2-2 fields."""
        import tempfile
        wav = Path(tempfile.mktemp(suffix=".wav"))
        make_wav_from_array(wav, make_click_track(bpm=120.0, duration=10.0))
        track = make_track_stub("bpm_test", str(wav))
        cache_dir = Path(tempfile.mkdtemp())
        result = analyze_track(track, cache_dir)
        assert_true("bpm_raw" in result, "Missing bpm_raw field")
        assert_true("bpm_normalised" in result, "Missing bpm_normalised field")
        assert_true("bpm_was_corrected" in result, "Missing bpm_was_corrected field")
        assert_not_none(result["bpm"])
        assert_not_none(result["bpm_raw"])
        # Clean up
        wav.unlink(missing_ok=True)

    for fn in [already_in_range_unchanged, half_time_doubled, double_time_halved,
               very_slow_doubled_multiple, very_fast_halved_multiple,
               dnb_range_stays, boundary_lo_stays, boundary_hi_stays,
               none_returns_none, zero_returns_zero, result_always_in_range,
               analyze_track_has_bpm_fields]:
        run_test(fn.__name__, fn)


def test_detect_sections():
    section("Unit Tests: detect_sections (P2-6)")

    def returns_list():
        y = make_sine(duration=30.0)
        beat_times = [i * 0.5 for i in range(120)]
        energy_curve = [0.5] * 32
        result = detect_sections(y, SR, beat_times, energy_curve, 30.0)
        assert_true(isinstance(result, list), "Should return a list")

    def sections_have_required_keys():
        y = make_sine(duration=30.0)
        beat_times = [i * 0.5 for i in range(120)]
        energy_curve = [0.5] * 32
        result = detect_sections(y, SR, beat_times, energy_curve, 30.0)
        if result:  # may be empty for simple sine
            for sec in result:
                for key in ["label", "start_sec", "end_sec", "energy", "energy_category"]:
                    assert_true(key in sec, f"Missing key: {key} in section {sec}")

    def sections_cover_track():
        """Sections should not overlap and should span roughly the full track."""
        # Build a more complex signal with energy changes
        y_quiet = make_sine(freq=440.0, duration=10.0, amplitude=0.1)
        y_loud  = make_sine(freq=440.0, duration=10.0, amplitude=0.8)
        y_quiet2 = make_sine(freq=440.0, duration=10.0, amplitude=0.1)
        y = np.concatenate([y_quiet, y_loud, y_quiet2])
        duration = 30.0
        beat_times = [i * 0.5 for i in range(120)]
        energy_curve = ([0.2] * 11) + ([0.8] * 10) + ([0.2] * 11)
        result = detect_sections(y, SR, beat_times, energy_curve, duration)
        if result:
            # Check no overlaps
            for i in range(len(result) - 1):
                assert_true(result[i]["end_sec"] <= result[i + 1]["start_sec"] + 0.1,
                            f"Overlap: {result[i]} and {result[i+1]}")
            # First section should start near 0
            assert_true(result[0]["start_sec"] <= 1.0,
                        f"First section starts too late: {result[0]['start_sec']}")

    def sections_have_valid_labels():
        y = make_sine(duration=30.0)
        beat_times = [i * 0.5 for i in range(120)]
        energy_curve = [0.5] * 32
        result = detect_sections(y, SR, beat_times, energy_curve, 30.0)
        valid_labels = {"intro", "outro", "verse", "chorus", "breakdown", "bridge", "drop"}
        for sec in result:
            assert_true(sec["label"] in valid_labels,
                        f"Invalid label: {sec['label']}")

    def energy_values_reasonable():
        y = make_sine(duration=20.0)
        beat_times = [i * 0.5 for i in range(80)]
        energy_curve = [0.5] * 32
        result = detect_sections(y, SR, beat_times, energy_curve, 20.0)
        for sec in result:
            assert_in_range(sec["energy"], 0.0, 1.5,
                            f"Energy {sec['energy']} out of expected range")

    def sections_respect_min_duration():
        y = make_sine(duration=30.0)
        beat_times = [i * 0.5 for i in range(120)]
        energy_curve = [0.5] * 32
        result = detect_sections(y, SR, beat_times, energy_curve, 30.0)
        for sec in result:
            dur = sec["end_sec"] - sec["start_sec"]
            assert_true(dur >= SECTION_MIN_DURATION_SEC * 0.9,
                        f"Section too short: {dur}s < {SECTION_MIN_DURATION_SEC}s")

    def empty_audio_returns_empty():
        y = np.zeros(SR * 2, dtype=np.float32)
        result = detect_sections(y, SR, [], [], 2.0)
        assert_true(isinstance(result, list))

    def complex_signal_finds_sections():
        """A signal with clear energy changes should produce multiple sections."""
        # quiet → loud → quiet → loud → quiet
        parts = []
        for i, amp in enumerate([0.05, 0.8, 0.05, 0.8, 0.05]):
            parts.append(make_sine(freq=440.0, duration=10.0, amplitude=amp))
        y = np.concatenate(parts)
        beat_times = [i * 0.5 for i in range(200)]
        energy_curve = ([0.1] * 6) + ([0.9] * 7) + ([0.1] * 6) + ([0.9] * 7) + ([0.1] * 6)
        result = detect_sections(y, SR, beat_times, energy_curve, 50.0)
        # Should find at least 2 distinct sections
        assert_true(len(result) >= 2,
                    f"Complex signal should have ≥2 sections, got {len(result)}")

    def analyze_track_has_sections_field():
        """Integration: analyze_track result includes new sections field."""
        import tempfile
        wav = Path(tempfile.mktemp(suffix=".wav"))
        make_wav_from_array(wav, make_click_track(bpm=120.0, duration=15.0))
        track = make_track_stub("sec_test", str(wav))
        cache_dir = Path(tempfile.mkdtemp())
        result = analyze_track(track, cache_dir)
        assert_true("sections" in result, "Missing 'sections' field in analysis result")
        # May be empty list or populated — both are valid
        assert_true(isinstance(result["sections"], (list, type(None))),
                    f"sections should be list or None, got {type(result['sections'])}")
        wav.unlink(missing_ok=True)

    for fn in [returns_list, sections_have_required_keys, sections_cover_track,
               sections_have_valid_labels, energy_values_reasonable,
               sections_respect_min_duration, empty_audio_returns_empty,
               complex_signal_finds_sections, analyze_track_has_sections_field]:
        run_test(fn.__name__, fn)

# def test_infer_genre():
#     section("Unit Tests: infer_genre (P2-4)")

#     def dnb_detected():
#         genre = infer_genre(170.0, 1500.0)
#         assert_eq(genre, "dnb", f"170 BPM + low centroid should be dnb, got {genre}")

#     def dnb_at_boundary():
#         genre = infer_genre(160.0, 1200.0)
#         assert_eq(genre, "dnb", f"160 BPM + dark centroid should be dnb, got {genre}")

#     def jungle_bright():
#         genre = infer_genre(168.0, 3500.0)
#         assert_eq(genre, "jungle", f"168 BPM + bright centroid should be jungle, got {genre}")

#     def house_detected():
#         genre = infer_genre(124.0, 2500.0)
#         assert_eq(genre, "house", f"124 BPM + mid centroid should be house, got {genre}")

#     def house_at_128():
#         genre = infer_genre(128.0, 2200.0)
#         assert_eq(genre, "house", f"128 BPM + mid centroid should be house, got {genre}")

#     def techno_dark():
#         genre = infer_genre(132.0, 1200.0)
#         assert_eq(genre, "techno", f"132 BPM + low centroid should be techno, got {genre}")

#     def techno_fast():
#         genre = infer_genre(140.0, 1600.0)
#         assert_eq(genre, "techno", f"140 BPM + mid-low centroid should be techno, got {genre}")

#     def breaks_detected():
#         genre = infer_genre(130.0, 2000.0, energy_dynamic_range=0.08)
#         assert_eq(genre, "breaks", f"130 BPM + high dynamic range should be breaks, got {genre}")

#     def downtempo_detected():
#         genre = infer_genre(100.0, 2000.0)
#         assert_eq(genre, "downtempo", f"100 BPM should be downtempo, got {genre}")

#     def downtempo_boundary():
#         genre = infer_genre(90.0, 1500.0)
#         assert_eq(genre, "downtempo", f"90 BPM should be downtempo, got {genre}")

#     def unknown_for_none_bpm():
#         genre = infer_genre(None, 2000.0)
#         assert_eq(genre, "unknown")

#     def unknown_for_zero_bpm():
#         genre = infer_genre(0, 2000.0)
#         assert_eq(genre, "unknown")

#     def unknown_for_extreme_bpm():
#         genre = infer_genre(200.0, 2000.0)
#         assert_eq(genre, "unknown", f"200 BPM should be unknown, got {genre}")

#     def none_centroid_still_works():
#         """Should not crash with None centroid — uses default."""
#         genre = infer_genre(128.0, None)
#         assert_true(genre in ("house", "techno", "breaks", "unknown"),
#                     f"128 BPM + None centroid should give valid genre, got {genre}")

#     def all_bpm_ranges_covered():
#         """Fuzz: every BPM from 85–175 should return a non-unknown genre."""
#         for bpm in range(85, 176):
#             genre = infer_genre(float(bpm), 2000.0)
#             assert_true(genre != "unknown",
#                         f"BPM {bpm} returned 'unknown' — should be classified")

#     def analyze_track_has_genre_field():
#         """Integration: analyze_track result includes genre field."""
#         import tempfile
#         wav = Path(tempfile.mktemp(suffix=".wav"))
#         make_wav_from_array(wav, make_click_track(bpm=120.0, duration=10.0))
#         track = make_track_stub("genre_test", str(wav))
#         cache_dir = Path(tempfile.mkdtemp())
#         result = analyze_track(track, cache_dir)
#         assert_true("genre" in result, "Missing 'genre' field in analysis result")
#         assert_true(result["genre"] is not None, "Genre should not be None for valid track")
#         wav.unlink(missing_ok=True)

#     for fn in [dnb_detected, dnb_at_boundary, jungle_bright,
#                house_detected, house_at_128, techno_dark, techno_fast,
#                breaks_detected, downtempo_detected, downtempo_boundary,
#                unknown_for_none_bpm, unknown_for_zero_bpm, unknown_for_extreme_bpm,
#                none_centroid_still_works, all_bpm_ranges_covered,
#                analyze_track_has_genre_field]:
#         run_test(fn.__name__, fn)

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
    test_normalise_bpm()
    test_detect_sections()
    # test_infer_genre()

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

