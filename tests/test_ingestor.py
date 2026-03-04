"""
Tests for Module 1: Library Ingestor
──────────────────────────────────────
Run with:  python tests/test_ingestor.py
Works standalone (no pytest needed).
All tests use stdlib WAV files — no external audio files required.
"""

import json
import sys
import wave
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ingestor import (
    make_track_id, format_duration, clean_tag,
    extract_metadata, scan_library, run,
)

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

def assert_approx(a, b, tol=0.2, msg=""):
    if abs(a - b) > tol:
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

def make_wav(path, secs=3.0, rate=44100):
    n = int(rate * secs)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n)

# ── Helpers ────────────────────────────────────────────────────────────────────
def test_helpers():
    section("Unit Tests: Helpers")

    run_test("track_id is 12 chars",        lambda: assert_eq(len(make_track_id("/a.mp3")), 12))
    run_test("track_id is hex",             lambda: assert_true(all(c in "0123456789abcdef" for c in make_track_id("/a.mp3"))))
    run_test("track_id deterministic",      lambda: assert_eq(make_track_id("/x"), make_track_id("/x")))
    run_test("track_id unique per path",    lambda: assert_true(make_track_id("/a") != make_track_id("/b")))
    run_test("format 0s",                   lambda: assert_eq(format_duration(0), "0:00"))
    run_test("format 45s",                  lambda: assert_eq(format_duration(45), "0:45"))
    run_test("format 60s",                  lambda: assert_eq(format_duration(60), "1:00"))
    run_test("format 3:27",                 lambda: assert_eq(format_duration(207), "3:27"))
    run_test("format None",                 lambda: assert_eq(format_duration(None), "0:00"))
    run_test("clean_tag string",            lambda: assert_eq(clean_tag("Hi"), "Hi"))
    run_test("clean_tag list",              lambda: assert_eq(clean_tag(["A"]), "A"))
    run_test("clean_tag empty list",        lambda: assert_none(clean_tag([])))
    run_test("clean_tag None",              lambda: assert_none(clean_tag(None)))
    run_test("clean_tag strips spaces",     lambda: assert_eq(clean_tag("  x  "), "x"))
    run_test("clean_tag empty str → None",  lambda: assert_none(clean_tag("")))

# ── extract_metadata ───────────────────────────────────────────────────────────
def test_extract_metadata():
    section("Unit Tests: extract_metadata")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)

        def all_keys():
            wav = p / "k.wav"; make_wav(wav)
            r = extract_metadata(wav)
            for k in ["track_id","filepath","filename","extension","title","artist",
                      "duration_seconds","duration_formatted","file_size_mb",
                      "has_missing_tags","parse_error"]:
                assert_true(k in r, f"Missing key: {k}")

        def wav_duration():
            wav = p / "d.wav"; make_wav(wav, secs=3.0)
            r = extract_metadata(wav)
            assert_not_none(r["duration_seconds"])
            assert_approx(r["duration_seconds"], 3.0)

        def id_stable():
            wav = p / "s.wav"; make_wav(wav)
            assert_eq(extract_metadata(wav)["track_id"], extract_metadata(wav)["track_id"])

        def ext_correct():
            wav = p / "e.wav"; make_wav(wav)
            assert_eq(extract_metadata(wav)["extension"], ".wav")

        def filepath_absolute():
            wav = p / "abs.wav"; make_wav(wav)
            assert_true(Path(extract_metadata(wav)["filepath"]).is_absolute())

        def missing_tags_flagged():
            wav = p / "ut.wav"; make_wav(wav)
            assert_true(extract_metadata(wav)["has_missing_tags"])

        def missing_file_no_crash():
            r = extract_metadata(p / "ghost.mp3")
            assert_true(isinstance(r, dict))

        def corrupt_no_crash():
            bad = p / "bad.wav"; bad.write_bytes(b"garbage")
            assert_true(isinstance(extract_metadata(bad), dict))

        def file_size():
            wav = p / "sz.wav"; make_wav(wav, secs=5.0)
            r = extract_metadata(wav)
            assert_not_none(r["file_size_mb"])
            assert_true(r["file_size_mb"] > 0)

        def sample_rate():
            wav = p / "sr.wav"; make_wav(wav, rate=44100)
            assert_eq(extract_metadata(wav)["sample_rate_hz"], 44100)

        for fn in [all_keys, wav_duration, id_stable, ext_correct, filepath_absolute,
                   missing_tags_flagged, missing_file_no_crash, corrupt_no_crash,
                   file_size, sample_rate]:
            run_test(fn.__name__, fn)

# ── scan_library ───────────────────────────────────────────────────────────────
def test_scan_library():
    section("Integration Tests: scan_library")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        make_wav(p / "t1.wav", secs=3.0)
        make_wav(p / "t2.wav", secs=2.5)
        sub = p / "artist" / "album"; sub.mkdir(parents=True)
        make_wav(sub / "deep.wav", secs=4.0)
        empty = p / "empty"; empty.mkdir()

        run_test("finds 3+ wav files",      lambda: assert_true(len(scan_library(p)[0]) >= 3))
        run_test("recurses subdirectories", lambda: assert_true(any("deep.wav" in t["filepath"] for t in scan_library(p)[0])))
        run_test("stats total matches",     lambda: assert_eq(len(scan_library(p)[0]), scan_library(p)[1]["total_tracks"]))
        run_test("duration > 0",            lambda: assert_true(scan_library(p)[1]["total_duration_seconds"] > 0))
        run_test("format breakdown exists", lambda: assert_true(".wav" in scan_library(p)[1]["formats"]))
        run_test("empty dir → []",          lambda: assert_eq(scan_library(empty)[0], []))
        run_test("bad path raises",         lambda: assert_raises(FileNotFoundError, scan_library, "/no/such/path"))
        run_test("file path raises",        lambda: assert_raises(NotADirectoryError, scan_library, p / "t1.wav"))
        run_test("all have 12-char IDs",    lambda: [assert_eq(len(t["track_id"]), 12) for t in scan_library(p)[0]])
        run_test("no duplicate IDs",        lambda: (lambda ids: assert_eq(len(ids), len(set(ids))))([t["track_id"] for t in scan_library(p)[0]]))
        run_test("scanned_at in stats",     lambda: assert_true("scanned_at" in scan_library(p)[1]))

# ── run() ──────────────────────────────────────────────────────────────────────
def test_run():
    section("Integration Tests: run()")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        music = p / "music"; music.mkdir()
        make_wav(music / "a.wav"); make_wav(music / "b.wav")

        def writes_json():
            out = p / "o1" / "tracks.json"
            run(music_dir=str(music), output_path=str(out))
            assert_true(out.exists())

        def valid_json():
            out = p / "o2" / "tracks.json"
            run(music_dir=str(music), output_path=str(out))
            data = json.loads(out.read_text())
            assert_true("tracks" in data and "stats" in data)

        def count_matches():
            out = p / "o3" / "tracks.json"
            run(music_dir=str(music), output_path=str(out))
            data = json.loads(out.read_text())
            assert_eq(len(data["tracks"]), data["stats"]["total_tracks"])

        def creates_parent_dirs():
            out = p / "a" / "b" / "c" / "tracks.json"
            run(music_dir=str(music), output_path=str(out))
            assert_true(out.exists())

        def returns_dict():
            out = p / "o4" / "tracks.json"
            r = run(music_dir=str(music), output_path=str(out))
            assert_true(isinstance(r, dict) and "tracks" in r)

        for fn in [writes_json, valid_json, count_matches, creates_parent_dirs, returns_dict]:
            run_test(fn.__name__, fn)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*58)
    print("  🧪 AutoDJ — Module 1 Test Suite")
    print("═"*58)

    test_helpers()
    test_extract_metadata()
    test_scan_library()
    test_run()

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
