"""
Tests for Module 4: Session Planner
Run with: uv run python tests/test_planner.py
"""

import sys, json, random, traceback, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.planner import (
    camelot_compatible,
    make_energy_curve,
    choose_curve,
    score_candidate,
    transition_hint,
    build_session,
    run,
)

_results = {"passed": 0, "failed": 0, "errors": []}

def ok(cond, msg=""):
    if not cond: raise AssertionError(msg or f"Expected truthy, got {cond!r}")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got: {a!r}")
def in_range(v, lo, hi, msg=""):
    if not (lo <= v <= hi): raise AssertionError(msg or f"Expected {lo}≤{v}≤{hi}, got {v}")

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

def make_analysis(tid="t001", bpm=120.0, camelot="8B", energy_mean=0.07, dance=0.85):
    return {
        "track_id":        tid,
        "filepath":        f"/music/{tid}.mp3",
        "filename":        f"{tid}.mp3",
        "bpm":             bpm,
        "camelot":         camelot,
        "key_name":        "C",
        "mode":            "major",
        "energy":          {"mean": energy_mean, "peak": energy_mean * 1.5,
                            "dynamic_range": 0.02, "curve": []},
        "danceability":    dance,
        "intro_end_sec":   2.0,
        "outro_start_sec": 180.0,
        "analysis_error":  None,
    }

def make_pool(n=10):
    bpms     = [90, 100, 110, 120, 120, 128, 128, 130, 140, 150]
    camelots = ["8B","8A","9B","7B","8B","3A","12B","5A","11B","2A"]
    return [make_analysis(f"t{i:03d}", bpm=bpms[i % len(bpms)],
                          camelot=camelots[i % len(camelots)],
                          energy_mean=0.04 + i * 0.008)
            for i in range(n)]

BASE_CONFIG = {
    "session": {"duration_minutes": 20},
    "dj": {"chaos_factor": 0.3, "harmonic_strict": False,
           "tempo_tolerance_bpm": 8, "transition_style": "mixed"},
}


# ── camelot_compatible ────────────────────────────────────────────────────────

def test_camelot():
    section("camelot_compatible")
    run_test("identical keys",          lambda: ok(camelot_compatible("8B", "8B")))
    run_test("relative major/minor",    lambda: ok(camelot_compatible("8A", "8B")))
    run_test("adjacent number same let",lambda: ok(camelot_compatible("8A", "9A")))
    run_test("adjacent down",           lambda: ok(camelot_compatible("8A", "7A")))
    run_test("wrap 12→1",               lambda: ok(camelot_compatible("12A", "1A")))
    run_test("incompatible keys",       lambda: ok(not camelot_compatible("8B", "3A")))
    run_test("None inputs → False",     lambda: ok(not camelot_compatible(None, "8B")))
    run_test("both None → False",       lambda: ok(not camelot_compatible(None, None)))


# ── make_energy_curve ─────────────────────────────────────────────────────────

def test_energy_curve():
    section("make_energy_curve")
    rng = random.Random(42)

    def correct_length():
        for curve in ["flat","ramp_up","peak_valley","festival","random"]:
            c = make_energy_curve(16, curve, rng=rng)
            eq(len(c), 16, f"{curve} curve wrong length")

    def values_in_range():
        for curve in ["flat","ramp_up","peak_valley","festival"]:
            c = make_energy_curve(20, curve, rng=rng)
            ok(all(0.0 <= v <= 1.0 for v in c), f"{curve} has out-of-range values")

    def ramp_up_increases():
        c = make_energy_curve(10, "ramp_up", chaos=0.0, rng=rng)
        ok(c[-1] > c[0], f"ramp_up should end higher than it starts: {c[0]} → {c[-1]}")

    def flat_is_stable():
        c = make_energy_curve(10, "flat", chaos=0.0, rng=rng)
        ok(max(c) - min(c) < 0.1, f"flat curve should have low variance")

    def festival_peaks_in_middle():
        c = make_energy_curve(20, "festival", chaos=0.0, rng=rng)
        peak_idx = c.index(max(c))
        ok(5 <= peak_idx <= 16, f"festival peak should be in middle 25-80%, got idx {peak_idx}")

    def chaos_adds_variance():
        c_low  = make_energy_curve(30, "flat", chaos=0.0, rng=random.Random(1))
        c_high = make_energy_curve(30, "flat", chaos=0.9, rng=random.Random(1))
        ok(max(c_high) - min(c_high) > max(c_low) - min(c_low),
           "Higher chaos should add more variance")

    for fn in [correct_length, values_in_range, ramp_up_increases,
               flat_is_stable, festival_peaks_in_middle, chaos_adds_variance]:
        run_test(fn.__name__, fn)


# ── choose_curve ──────────────────────────────────────────────────────────────

def test_choose_curve():
    section("choose_curve")
    rng = random.Random(42)

    def low_chaos_gives_festival():
        results = {choose_curve(0.0, rng) for _ in range(20)}
        ok("festival" in results, "Low chaos should pick festival")
        ok(len(results) == 1, f"Low chaos should always pick festival, got {results}")

    def high_chaos_varies():
        results = {choose_curve(1.0, rng) for _ in range(40)}
        ok(len(results) > 1, f"High chaos should vary curves, got {results}")

    for fn in [low_chaos_gives_festival, high_chaos_varies]:
        run_test(fn.__name__, fn)


# ── score_candidate ───────────────────────────────────────────────────────────

def test_score_candidate():
    section("score_candidate")
    rng = random.Random(42)

    def returns_float():
        c = make_analysis("t1", bpm=120.0, camelot="8B")
        s = score_candidate(c, None, 0.7, 8, False, 0.3, rng)
        ok(isinstance(s, float))

    def harmonic_match_scores_higher():
        prev = make_analysis("prev", bpm=120.0, camelot="8B")
        compat   = make_analysis("c1", bpm=120.0, camelot="8A")  # relative key
        incompat = make_analysis("c2", bpm=120.0, camelot="3A")  # incompatible
        s_compat   = score_candidate(compat,   prev, 0.7, 8, False, 0.0, rng)
        s_incompat = score_candidate(incompat, prev, 0.7, 8, False, 0.0, rng)
        ok(s_compat > s_incompat, f"Harmonic match should score higher: {s_compat} vs {s_incompat}")

    def bpm_match_scores_higher():
        prev      = make_analysis("prev", bpm=120.0, camelot="8B")
        close_bpm = make_analysis("c1",  bpm=122.0, camelot="3A")
        far_bpm   = make_analysis("c2",  bpm=145.0, camelot="3A")
        s_close = score_candidate(close_bpm, prev, 0.7, 8, False, 0.0, rng)
        s_far   = score_candidate(far_bpm,   prev, 0.7, 8, False, 0.0, rng)
        ok(s_close > s_far, f"Close BPM should score higher: {s_close} vs {s_far}")

    for fn in [returns_float, harmonic_match_scores_higher, bpm_match_scores_higher]:
        run_test(fn.__name__, fn)


# ── transition_hint ───────────────────────────────────────────────────────────

def test_transition_hint():
    section("transition_hint")

    def none_prev_is_hard_cut():
        eq(transition_hint(None, make_analysis(), "mixed"), "hard_cut")

    def cuts_style_always_hard_cut():
        prev = make_analysis("p", bpm=120.0)
        next_ = make_analysis("n", bpm=121.0)
        eq(transition_hint(prev, next_, "cuts"), "hard_cut")

    def large_bpm_delta_is_hard_cut():
        prev  = make_analysis("p", bpm=120.0)
        next_ = make_analysis("n", bpm=160.0)  # >12% delta
        eq(transition_hint(prev, next_, "mixed"), "hard_cut")

    def small_delta_harmonic_is_long_blend():
        prev  = make_analysis("p", bpm=120.0, camelot="8B")
        next_ = make_analysis("n", bpm=121.0, camelot="8A")  # compat + tiny delta
        eq(transition_hint(prev, next_, "blends"), "long_blend")

    for fn in [none_prev_is_hard_cut, cuts_style_always_hard_cut,
               large_bpm_delta_is_hard_cut, small_delta_harmonic_is_long_blend]:
        run_test(fn.__name__, fn)


# ── build_session ─────────────────────────────────────────────────────────────

def test_build_session():
    section("build_session (integration)")
    pool = make_pool(10)
    rng  = random.Random(42)

    def returns_setlist_and_curve():
        result = build_session(pool, set(), BASE_CONFIG, rng)
        ok(result is not None)
        setlist, curve = result
        ok(isinstance(setlist, list))
        ok(isinstance(curve, str))

    def setlist_has_tracks():
        setlist, _ = build_session(pool, set(), BASE_CONFIG, random.Random(1))
        ok(len(setlist) > 0, "Setlist should have at least one track")

    def no_duplicate_tracks():
        setlist, _ = build_session(pool, set(), BASE_CONFIG, random.Random(2))
        ids = [s["track_id"] for s in setlist]
        eq(len(ids), len(set(ids)), "Setlist should not have duplicate tracks")

    def slots_have_required_keys():
        setlist, _ = build_session(pool, set(), BASE_CONFIG, random.Random(3))
        for slot in setlist:
            for key in ["position", "track_id", "filepath", "filename",
                        "actual_bpm", "camelot", "energy_target",
                        "transition_hint", "is_seed"]:
                ok(key in slot, f"Missing key in slot: {key}")

    def seed_tracks_appear_first():
        seed_ids = {pool[0]["track_id"], pool[1]["track_id"]}
        setlist, _ = build_session(pool, seed_ids, BASE_CONFIG, random.Random(4))
        seed_positions = [s["position"] for s in setlist if s["is_seed"]]
        if seed_positions:
            ok(min(seed_positions) < len(setlist) // 2,
               "Seeds should tend to appear in first half")

    def positions_are_sequential():
        setlist, _ = build_session(pool, set(), BASE_CONFIG, random.Random(5))
        for i, slot in enumerate(setlist):
            eq(slot["position"], i, f"Position {i} out of order")

    def high_chaos_still_produces_setlist():
        chaos_config = {**BASE_CONFIG, "dj": {**BASE_CONFIG["dj"], "chaos_factor": 1.0}}
        setlist, _ = build_session(pool, set(), chaos_config, random.Random(6))
        ok(len(setlist) > 0)

    def empty_pool_returns_empty():
        result = build_session([], set(), BASE_CONFIG, random.Random(7))
        ok(result == [] or (isinstance(result, tuple) and result[0] == []))

    for fn in [returns_setlist_and_curve, setlist_has_tracks, no_duplicate_tracks,
               slots_have_required_keys, seed_tracks_appear_first,
               positions_are_sequential, high_chaos_still_produces_setlist,
               empty_pool_returns_empty]:
        run_test(fn.__name__, fn)


# ── run() / JSON output ───────────────────────────────────────────────────────

def test_run():
    section("run() — full integration")
    pool = make_pool(10)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)

        # Write fake analysis.json
        analysis_path = p / "analysis.json"
        analysis_path.write_text(json.dumps({"analyses": pool}))

        # Write minimal session.yaml
        config_path = p / "session.yaml"
        config_path.write_text(
            "session:\n  duration_minutes: 20\n"
            "dj:\n  chaos_factor: 0.3\n  harmonic_strict: false\n"
            "  tempo_tolerance_bpm: 8\n  transition_style: mixed\n  random_seed: 99\n"
        )

        output_path = p / "session.json"

        def writes_json():
            run(analysis_path=str(analysis_path), matched_tracks_path="/nonexistent",
                config_path=str(config_path), output_path=str(output_path))
            ok(output_path.exists())

        def output_is_valid():
            data = json.loads(output_path.read_text())
            for key in ["generated_at", "curve_type", "chaos_factor",
                        "total_tracks", "setlist"]:
                ok(key in data, f"Missing key: {key}")

        def setlist_in_output():
            data = json.loads(output_path.read_text())
            ok(len(data["setlist"]) > 0)

        def reproducible_with_seed():
            out1 = p / "s1.json"
            out2 = p / "s2.json"
            run(analysis_path=str(analysis_path), matched_tracks_path="/nonexistent",
                config_path=str(config_path), output_path=str(out1))
            run(analysis_path=str(analysis_path), matched_tracks_path="/nonexistent",
                config_path=str(config_path), output_path=str(out2))
            d1 = json.loads(out1.read_text())
            d2 = json.loads(out2.read_text())
            ids1 = [s["track_id"] for s in d1["setlist"]]
            ids2 = [s["track_id"] for s in d2["setlist"]]
            eq(ids1, ids2, "Seeded runs should be reproducible")

        for fn in [writes_json, output_is_valid, setlist_in_output, reproducible_with_seed]:
            run_test(fn.__name__, fn)


if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Module 4 Tests\n" + "═"*52)
    test_camelot()
    test_energy_curve()
    test_choose_curve()
    test_score_candidate()
    test_transition_hint()
    test_build_session()
    test_run()
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
