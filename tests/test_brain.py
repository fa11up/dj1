"""
Tests for Module 4b: Brain (AI setlist planner)
Run with: uv run python tests/test_brain.py

No API calls made — tests cover pure-logic helpers only:
  _build_track_summary, _build_user_message, _rehydrate_setlist,
  tool schema validation, plan_session fallback behaviour.
"""

import os, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.brain import (
    SYSTEM_PROMPT,
    _TOOL_INPUT_SCHEMA,
    _build_track_summary,
    _build_user_message,
    _rehydrate_setlist,
    plan_session,
    ANTHROPIC_AVAILABLE,
    OPENAI_AVAILABLE,
)

_results = {"passed": 0, "failed": 0, "errors": []}

def ok(cond, msg=""):
    if not cond: raise AssertionError(msg or f"Expected truthy, got {cond!r}")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got: {a!r}")
def has(d, key, msg=""):
    if key not in d: raise AssertionError(msg or f"Expected key {key!r} in dict")

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

def make_track(tid="t1", bpm=128.0, camelot="8B", energy_mean=0.65):
    return {
        "track_id":               tid,
        "filename":               f"{tid}.mp3",
        "filepath":               f"/music/{tid}.mp3",
        "bpm":                    bpm,
        "bpm_normalised":         bpm,
        "camelot":                camelot,
        "key_name":               "G",
        "mode":                   "major",
        "energy":                 {"mean": energy_mean, "curve": [0.5] * 32},
        "danceability":           0.8,
        "intro_end_sec":          16.0,
        "outro_start_sec":        220.0,
        "beat_times":             [i * 0.47 for i in range(300)],
        "spectral_centroid_mean": 2000.0,
        "sections":               [],
    }

def make_config(**overrides):
    base = {
        "brain":   {"enabled": True, "provider": "anthropic", "model": "", "base_url": ""},
        "session": {"mood_prompt": "dark underground", "duration_minutes": 30,
                    "energy_curve": "festival"},
        "dj":      {"chaos_factor": 0.3, "bpm_normalize": True},
    }
    base.update(overrides)
    return base


# ── Tool schema ───────────────────────────────────────────────────────────────

EXPECTED_TRANSITIONS = {
    "hard_cut", "quick_fade", "crossfade",
    "long_blend", "stem_blend",
    "filter_sweep", "reverb_wash",
    "harmonic_blend", "tension_drop",
    "loop_roll", "loop_stutter",
}

def test_tool_schema():
    section("_TOOL_INPUT_SCHEMA")

    def schema_is_dict():
        ok(isinstance(_TOOL_INPUT_SCHEMA, dict))

    def schema_has_setlist():
        ok("setlist" in _TOOL_INPUT_SCHEMA.get("properties", {}))

    def schema_has_curve_type():
        ok("curve_type" in _TOOL_INPUT_SCHEMA.get("properties", {}))

    def setlist_items_have_transition_hint():
        items = (
            _TOOL_INPUT_SCHEMA["properties"]["setlist"]
            .get("items", {})
            .get("properties", {})
        )
        ok("transition_hint" in items, "setlist items must have transition_hint")

    def setlist_items_have_reasoning():
        items = (
            _TOOL_INPUT_SCHEMA["properties"]["setlist"]
            .get("items", {})
            .get("properties", {})
        )
        ok("reasoning" in items, "setlist items must have reasoning")

    def enum_contains_all_transitions():
        enum = (
            _TOOL_INPUT_SCHEMA["properties"]["setlist"]
            ["items"]["properties"]["transition_hint"]["enum"]
        )
        missing = EXPECTED_TRANSITIONS - set(enum)
        ok(not missing, f"Enum missing transitions: {missing}")

    def enum_has_loop_transitions():
        enum = (
            _TOOL_INPUT_SCHEMA["properties"]["setlist"]
            ["items"]["properties"]["transition_hint"]["enum"]
        )
        ok("loop_roll"    in enum, "enum must contain loop_roll")
        ok("loop_stutter" in enum, "enum must contain loop_stutter")

    def enum_count_matches_expected():
        enum = (
            _TOOL_INPUT_SCHEMA["properties"]["setlist"]
            ["items"]["properties"]["transition_hint"]["enum"]
        )
        eq(len(enum), len(EXPECTED_TRANSITIONS),
           f"Expected {len(EXPECTED_TRANSITIONS)} transitions, got {len(enum)}")

    for fn in [schema_is_dict, schema_has_setlist, schema_has_curve_type,
               setlist_items_have_transition_hint, setlist_items_have_reasoning,
               enum_contains_all_transitions, enum_has_loop_transitions,
               enum_count_matches_expected]:
        run_test(fn.__name__, fn)


# ── _build_track_summary ──────────────────────────────────────────────────────

def test_build_track_summary():
    section("_build_track_summary")

    t = make_track("t42", bpm=130.0, camelot="9A", energy_mean=0.72)

    def returns_dict():
        ok(isinstance(_build_track_summary(t), dict))

    def has_required_keys():
        s = _build_track_summary(t)
        for key in ("track_id", "filename", "bpm", "camelot", "energy_mean",
                    "danceability", "intro_end_sec", "outro_start_sec"):
            has(s, key, f"Summary missing key: {key}")

    def bpm_matches():
        eq(_build_track_summary(t)["bpm"], 130.0)

    def camelot_matches():
        eq(_build_track_summary(t)["camelot"], "9A")

    def energy_mean_matches():
        s = _build_track_summary(t)
        ok(abs(s["energy_mean"] - 0.72) < 0.01)

    def is_seed_false_by_default():
        ok(not _build_track_summary(t, is_seed=False)["is_seed"])

    def is_seed_true_when_flagged():
        ok(_build_track_summary(t, is_seed=True)["is_seed"])

    def sections_included():
        ok("sections" in _build_track_summary(t))

    for fn in [returns_dict, has_required_keys, bpm_matches, camelot_matches,
               energy_mean_matches, is_seed_false_by_default, is_seed_true_when_flagged,
               sections_included]:
        run_test(fn.__name__, fn)


# ── _build_user_message ───────────────────────────────────────────────────────

def test_build_user_message():
    section("_build_user_message")

    tracks = [_build_track_summary(make_track("t1")),
              _build_track_summary(make_track("t2"))]

    def returns_string():
        msg = _build_user_message(tracks, "dark underground", 30, 5, 0.3, "festival", set())
        ok(isinstance(msg, str))

    def contains_mood():
        msg = _build_user_message(tracks, "dark underground", 30, 5, 0.3, "festival", set())
        ok("dark underground" in msg)

    def contains_target_count():
        msg = _build_user_message(tracks, "", 30, 7, 0.3, "festival", set())
        ok("7" in msg)

    def low_chaos_safe_note():
        msg = _build_user_message(tracks, "", 30, 5, 0.1, "flat", set())
        ok("safe" in msg.lower() or "Low chaos" in msg)

    def high_chaos_adventurous_note():
        msg = _build_user_message(tracks, "", 30, 5, 0.9, "flat", set())
        ok("adventurous" in msg.lower() or "High chaos" in msg)

    def contains_track_library():
        msg = _build_user_message(tracks, "", 30, 5, 0.3, "festival", set())
        ok("t1" in msg and "t2" in msg, "Should include track IDs in library")

    def seed_tracks_highlighted():
        seed_track = _build_track_summary(make_track("seed1"), is_seed=True)
        msg = _build_user_message([seed_track] + tracks, "", 30, 5, 0.3, "flat", {"seed1"})
        ok("seed1" in msg or "Seed" in msg, "Seed track should be mentioned")

    for fn in [returns_string, contains_mood, contains_target_count,
               low_chaos_safe_note, high_chaos_adventurous_note,
               contains_track_library, seed_tracks_highlighted]:
        run_test(fn.__name__, fn)


# ── _rehydrate_setlist ────────────────────────────────────────────────────────

def test_rehydrate_setlist():
    section("_rehydrate_setlist")

    tracks   = [make_track("t1", bpm=128.0), make_track("t2", bpm=132.0)]
    by_id    = {t["track_id"]: t for t in tracks}
    config   = make_config()

    def maps_track_ids_correctly():
        brain_slots = [
            {"track_id": "t1", "transition_hint": "hard_cut",  "reasoning": "opener"},
            {"track_id": "t2", "transition_hint": "long_blend", "reasoning": "solid mix"},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        eq(len(result), 2)
        eq(result[0]["track_id"], "t1")
        eq(result[1]["track_id"], "t2")

    def skips_unknown_ids():
        brain_slots = [
            {"track_id": "t1",      "transition_hint": "hard_cut", "reasoning": ""},
            {"track_id": "UNKNOWN", "transition_hint": "crossfade", "reasoning": ""},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        eq(len(result), 1)

    def skips_duplicate_ids():
        brain_slots = [
            {"track_id": "t1", "transition_hint": "hard_cut",  "reasoning": ""},
            {"track_id": "t1", "transition_hint": "crossfade", "reasoning": ""},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        eq(len(result), 1, "Duplicate track_id should be skipped")

    def carries_transition_hint():
        brain_slots = [
            {"track_id": "t1", "transition_hint": "hard_cut",  "reasoning": ""},
            {"track_id": "t2", "transition_hint": "loop_roll",  "reasoning": "loop it"},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        eq(result[1]["transition_hint"], "loop_roll")

    def carries_loop_stutter_hint():
        brain_slots = [
            {"track_id": "t1", "transition_hint": "hard_cut",    "reasoning": ""},
            {"track_id": "t2", "transition_hint": "loop_stutter", "reasoning": "stutter in"},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        eq(result[1]["transition_hint"], "loop_stutter")

    def marks_seed_tracks():
        brain_slots = [
            {"track_id": "t1", "transition_hint": "hard_cut",  "reasoning": ""},
            {"track_id": "t2", "transition_hint": "crossfade", "reasoning": ""},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, {"t1"}, config)
        ok(result[0]["is_seed"],     "t1 should be marked as seed")
        ok(not result[1]["is_seed"], "t2 should not be marked as seed")

    def carries_reasoning():
        brain_slots = [
            {"track_id": "t1", "transition_hint": "hard_cut",  "reasoning": "kicks off well"},
            {"track_id": "t2", "transition_hint": "long_blend", "reasoning": "harmonic step"},
        ]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        eq(result[0]["brain_reasoning"], "kicks off well")

    def empty_brain_slots_returns_empty():
        result = _rehydrate_setlist([], by_id, set(), config)
        eq(result, [])

    def result_has_filepath():
        brain_slots = [{"track_id": "t1", "transition_hint": "hard_cut", "reasoning": ""}]
        result = _rehydrate_setlist(brain_slots, by_id, set(), config)
        has(result[0], "filepath", "result should include filepath")

    for fn in [maps_track_ids_correctly, skips_unknown_ids, skips_duplicate_ids,
               carries_transition_hint, carries_loop_stutter_hint, marks_seed_tracks,
               carries_reasoning, empty_brain_slots_returns_empty, result_has_filepath]:
        run_test(fn.__name__, fn)


# ── plan_session fallback ─────────────────────────────────────────────────────

def test_plan_session_fallback():
    section("plan_session — fallback paths (no API calls)")

    def returns_none_without_api_key():
        saved_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        saved_brain     = os.environ.pop("BRAIN_API_KEY",     None)
        try:
            result = plan_session(
                [make_track("t1"), make_track("t2")],
                seed_ids=set(),
                config=make_config(),
                target_n=2,
            )
            ok(result is None, "Should return None without API credentials")
        finally:
            if saved_anthropic: os.environ["ANTHROPIC_API_KEY"] = saved_anthropic
            if saved_brain:     os.environ["BRAIN_API_KEY"]     = saved_brain

    def returns_none_with_empty_pool():
        result = plan_session([], seed_ids=set(), config=make_config(), target_n=2)
        ok(result is None, "Empty pool → None")

    def returns_none_with_all_errored_tracks():
        bad = [{"track_id": "x", "analysis_error": "failed", "bpm": None}]
        result = plan_session(bad, set(), make_config(), 2)
        ok(result is None, "All errored tracks → None")

    def returns_none_with_tracks_missing_bpm():
        no_bpm = [{"track_id": "x", "filename": "x.mp3", "filepath": "/x", "bpm": None}]
        result = plan_session(no_bpm, set(), make_config(), 2)
        ok(result is None, "Tracks without BPM should be filtered → None")

    for fn in [returns_none_without_api_key, returns_none_with_empty_pool,
               returns_none_with_all_errored_tracks, returns_none_with_tracks_missing_bpm]:
        run_test(fn.__name__, fn)


# ── System prompt content ─────────────────────────────────────────────────────

def test_system_prompt():
    section("SYSTEM_PROMPT content")

    run_test("is non-empty string",
        lambda: ok(isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 100))
    run_test("mentions Camelot wheel",
        lambda: ok("Camelot" in SYSTEM_PROMPT))
    run_test("mentions BPM thresholds",
        lambda: ok("BPM" in SYSTEM_PROMPT and "delta" in SYSTEM_PROMPT))
    run_test("mentions loop_roll",
        lambda: ok("loop_roll" in SYSTEM_PROMPT))
    run_test("mentions loop_stutter",
        lambda: ok("loop_stutter" in SYSTEM_PROMPT))
    run_test("mentions tension_drop",
        lambda: ok("tension_drop" in SYSTEM_PROMPT))
    run_test("mentions harmonic_blend",
        lambda: ok("harmonic_blend" in SYSTEM_PROMPT))
    run_test("mentions filter_sweep",
        lambda: ok("filter_sweep" in SYSTEM_PROMPT))
    run_test("mentions reverb_wash",
        lambda: ok("reverb_wash" in SYSTEM_PROMPT))


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Brain Tests\n" + "═"*52)
    print(f"\n  anthropic={'✓' if ANTHROPIC_AVAILABLE else '✗'}  "
          f"openai={'✓' if OPENAI_AVAILABLE else '✗'}\n")

    test_tool_schema()
    test_build_track_summary()
    test_build_user_message()
    test_rehydrate_setlist()
    test_plan_session_fallback()
    test_system_prompt()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'═'*52}")
    if _results["failed"]:
        print(f"  {_results['passed']}/{total} passed | {_results['failed']} FAILED")
        for name, tb in _results["errors"]:
            print(f"\n  ── {name} ──\n{tb}")
        sys.exit(1)
    else:
        print(f"  {total}/{total} passed ✅")
