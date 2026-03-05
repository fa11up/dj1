"""
Tests for Module 5b: Stems (stem separation + cache helpers)
Run with: uv run python tests/test_stems.py

No demucs required — tests cover cache helpers, path logic, and graceful
fallback behaviour when demucs is absent or files are missing.
"""

import sys, tempfile, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.stems import (
    demucs_available,
    _stem_dir,
    _stems_cached,
    _stem_paths,
    run_stems_for_setlist,
    run_stems_for_creative,
    load_stems,
    STEM_NAMES,
    DEMUCS_MODEL,
)

_results = {"passed": 0, "failed": 0, "errors": []}

def ok(cond, msg=""):
    if not cond: raise AssertionError(msg or f"Expected truthy, got {cond!r}")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got: {a!r}")

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


# ── demucs_available ──────────────────────────────────────────────────────────

def test_demucs_available():
    section("demucs_available")
    run_test("returns bool", lambda: ok(isinstance(demucs_available(), bool)))


# ── module constants ──────────────────────────────────────────────────────────

def test_constants():
    section("module constants")
    run_test("STEM_NAMES has 4 entries",
        lambda: eq(len(STEM_NAMES), 4))
    run_test("STEM_NAMES includes drums/bass/other/vocals",
        lambda: ok(all(n in STEM_NAMES for n in ("drums", "bass", "other", "vocals"))))
    run_test("DEMUCS_MODEL is a non-empty string",
        lambda: ok(isinstance(DEMUCS_MODEL, str) and len(DEMUCS_MODEL) > 0))


# ── cache helpers ─────────────────────────────────────────────────────────────

def test_cache_helpers():
    section("cache helpers (_stem_dir, _stems_cached, _stem_paths)")

    with tempfile.TemporaryDirectory() as tmp:

        def stem_dir_correct_path():
            d = _stem_dir(tmp, "track_abc")
            eq(str(d), str(Path(tmp) / "track_abc"))

        def stems_cached_false_when_dir_missing():
            ok(not _stems_cached(tmp, "nonexistent_track"))

        def stems_cached_false_when_partial():
            # Create dir but only some stem files
            d = Path(tmp) / "partial_track"
            d.mkdir()
            (d / "drums.wav").write_bytes(b"placeholder")
            ok(not _stems_cached(tmp, "partial_track"),
               "Should be False when not all stems present")

        def stems_cached_true_when_complete():
            d = Path(tmp) / "complete_track"
            d.mkdir()
            for name in STEM_NAMES:
                (d / f"{name}.wav").write_bytes(b"placeholder")
            ok(_stems_cached(tmp, "complete_track"),
               "Should be True when all stem files exist")

        def stem_paths_returns_correct_keys():
            paths = _stem_paths(tmp, "mytrack")
            eq(set(paths.keys()), set(STEM_NAMES),
               "Should return exactly the 4 stem names")

        def stem_paths_correct_filenames():
            paths = _stem_paths(tmp, "mytrack")
            for name, path in paths.items():
                ok(path.endswith(f"{name}.wav"),
                   f"Path for '{name}' should end with '{name}.wav'")

        def stem_paths_returns_strings():
            paths = _stem_paths(tmp, "mytrack")
            ok(all(isinstance(v, str) for v in paths.values()))

        for fn in [stem_dir_correct_path, stems_cached_false_when_dir_missing,
                   stems_cached_false_when_partial, stems_cached_true_when_complete,
                   stem_paths_returns_correct_keys, stem_paths_correct_filenames,
                   stem_paths_returns_strings]:
            run_test(fn.__name__, fn)


# ── run_stems_for_setlist ─────────────────────────────────────────────────────

def test_run_stems_for_setlist():
    section("run_stems_for_setlist")

    def returns_empty_when_no_stem_blend():
        setlist = [
            {"track_id": "t1", "filepath": "/fake.mp3", "transition_hint": "crossfade"},
            {"track_id": "t2", "filepath": "/fake.mp3", "transition_hint": "long_blend"},
            {"track_id": "t3", "filepath": "/fake.mp3", "transition_hint": "loop_roll"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stems_for_setlist(setlist, tmp)
            eq(result, {}, "Should return empty dict when no stem_blend slots")

    def returns_empty_for_empty_setlist():
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stems_for_setlist([], tmp)
            eq(result, {})

    def returns_empty_when_demucs_unavailable():
        """When demucs is missing, always returns empty dict."""
        if demucs_available():
            return  # skip — demucs is present in this environment
        setlist = [
            {"track_id": "t1", "filepath": "/fake.mp3", "transition_hint": "stem_blend"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stems_for_setlist(setlist, tmp)
            eq(result, {}, "Should return empty when demucs unavailable")

    def creates_cache_dir():
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "stems_cache"
            setlist = [
                {"track_id": "t1", "filepath": "/fake.mp3", "transition_hint": "stem_blend"},
            ]
            run_stems_for_setlist(setlist, str(cache))
            ok(cache.exists(), "Cache dir should be created")

    for fn in [returns_empty_when_no_stem_blend, returns_empty_for_empty_setlist,
               returns_empty_when_demucs_unavailable, creates_cache_dir]:
        run_test(fn.__name__, fn)


# ── run_stems_for_creative ────────────────────────────────────────────────────

def test_run_stems_for_creative():
    section("run_stems_for_creative")

    def returns_empty_when_no_tracks():
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stems_for_creative([], [], tmp)
            eq(result, {})

    def returns_empty_when_demucs_unavailable():
        if demucs_available():
            return
        analyses = [{"track_id": "t1", "filepath": "/fake.mp3"}]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stems_for_creative(["t1"], analyses, tmp)
            eq(result, {})

    def skips_tracks_with_missing_filepath():
        if demucs_available():
            return
        analyses = [{"track_id": "t1", "filepath": "/nonexistent/path.mp3"}]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stems_for_creative(["t1"], analyses, tmp)
            eq(result, {})

    for fn in [returns_empty_when_no_tracks, returns_empty_when_demucs_unavailable,
               skips_tracks_with_missing_filepath]:
        run_test(fn.__name__, fn)


# ── load_stems ────────────────────────────────────────────────────────────────

def test_load_stems():
    section("load_stems — graceful failure on missing/bad files")

    def returns_none_for_all_missing_files():
        result = load_stems({
            "drums":  "/nonexistent/drums.wav",
            "bass":   "/nonexistent/bass.wav",
            "other":  "/nonexistent/other.wav",
            "vocals": "/nonexistent/vocals.wav",
        })
        ok(result is None or isinstance(result, dict),
           "Should return None or dict — not raise")

    def empty_dict_returns_none_or_empty():
        result = load_stems({})
        ok(result is None or result == {},
           "Empty input should return None or empty dict")

    def partial_missing_returns_none_or_partial():
        with tempfile.TemporaryDirectory() as tmp:
            # Only drums.wav exists (and is invalid audio)
            drums = Path(tmp) / "drums.wav"
            drums.write_bytes(b"not valid audio")
            result = load_stems({
                "drums":  str(drums),
                "bass":   "/nonexistent/bass.wav",
                "other":  "/nonexistent/other.wav",
                "vocals": "/nonexistent/vocals.wav",
            })
            # Should return None (partial stems) or dict — not crash
            ok(result is None or isinstance(result, dict))

    for fn in [returns_none_for_all_missing_files, empty_dict_returns_none_or_empty,
               partial_missing_returns_none_or_partial]:
        run_test(fn.__name__, fn)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Stems Tests\n" + "═"*52)
    print(f"\n  demucs={'✓' if demucs_available() else '✗ (not installed)'}\n")

    test_constants()
    test_demucs_available()
    test_cache_helpers()
    test_run_stems_for_setlist()
    test_run_stems_for_creative()
    test_load_stems()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'═'*52}")
    if _results["failed"]:
        print(f"  {_results['passed']}/{total} passed | {_results['failed']} FAILED")
        for name, tb in _results["errors"]:
            print(f"\n  ── {name} ──\n{tb}")
        sys.exit(1)
    else:
        print(f"  {total}/{total} passed ✅")
