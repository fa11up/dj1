"""
Tests for Module 3: Spotify Bridge
Run with: uv run python tests/test_spotify_bridge.py
"""

import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.spotify_bridge import extract_playlist_id, normalize, match_score, match_to_library

_results = {"passed": 0, "failed": 0, "errors": []}

def ok(cond, msg=""): 
    if not cond: raise AssertionError(msg or f"Expected truthy, got {cond!r}")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"{msg}\n  Expected: {b!r}\n  Got: {a!r}")
def in_range(v, lo, hi):
    if not (lo <= v <= hi): raise AssertionError(f"Expected {lo}≤{v}≤{hi}")

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

def sp(name="Track", artist="Artist"):
    return {"name": name, "artist": artist}

def loc(title="Track", artist="Artist", tid="l001"):
    return {"track_id": tid, "filepath": f"/music/{title}.mp3",
            "filename": f"{title}.mp3", "title": title, "artist": artist}


def test_extract():
    section("extract_playlist_id")
    ID = "37i9dQZF1DX4JAvHpjipBk"
    run_test("raw ID",      lambda: eq(extract_playlist_id(ID), ID))
    run_test("full URL",    lambda: eq(extract_playlist_id(f"https://open.spotify.com/playlist/{ID}?si=x"), ID))
    run_test("URI",         lambda: eq(extract_playlist_id(f"spotify:playlist:{ID}"), ID))
    run_test("whitespace",  lambda: eq(extract_playlist_id(f"  {ID}  "), ID))

def test_normalize():
    section("normalize")
    run_test("lowercases",          lambda: eq(normalize("HELLO"), "hello"))
    run_test("strips feat.",        lambda: ok("feat" not in normalize("Song (feat. X)")))
    run_test("strips Radio Edit",   lambda: ok("radio edit" not in normalize("Track (Radio Edit)")))
    run_test("strips Remastered",   lambda: ok("remaster" not in normalize("Song (Remastered)")))
    run_test("collapses spaces",    lambda: eq(normalize("too  many  spaces"), "too many spaces"))

def test_match_score():
    section("match_score")
    run_test("exact match scores high",   lambda: ok(match_score("Bohemian Rhapsody", "Queen", loc("Bohemian Rhapsody", "Queen")) >= 85))
    run_test("different tracks score low",lambda: ok(match_score("Song A", "Artist X", loc("Totally Different", "Nobody")) < 60))
    run_test("partial title match",       lambda: ok(match_score("Lose Yourself", "Eminem", loc("Lose Yourself (Radio Edit)", "Eminem")) >= 70))
    run_test("score in 0-100",            lambda: in_range(match_score("Track", "Artist", loc("Track", "Artist")), 0, 100))
    run_test("no local artist → title only", lambda: ok(match_score("Track Name", "X", {**loc("Track Name", ""), "artist": None}) >= 70))

def test_match_to_library():
    section("match_to_library")
    run_test("exact → matched",       lambda: eq(len(match_to_library([sp("A","X")], [loc("A","X")])[0]), 1))
    run_test("no match → unmatched",  lambda: eq(len(match_to_library([sp("Ghost","Nobody")], [loc("Other","Other")])[1]), 1))
    run_test("matched has filepath",  lambda: ok(match_to_library([sp("A","X")], [loc("A","X")])[0][0]["local_filepath"] is not None))
    run_test("matched has score",     lambda: ok("score" in match_to_library([sp("A","X")], [loc("A","X")])[0][0]))
    run_test("unmatched has best_score", lambda: ok("best_score" in match_to_library([sp("Ghost","X")], [loc("Other","Y")], threshold=99)[1][0]))
    run_test("total count preserved", lambda: (
        lambda m, u: eq(len(m) + len(u), 3))(
            *match_to_library([sp("A","X"), sp("B","Y"), sp("Ghost","?")],
                              [loc("A","X"), loc("B","Y")])))
    run_test("lower threshold matches more", lambda: ok(
        len(match_to_library([sp("Song A Remix","X")], [loc("Song A","X")], threshold=40)[0]) >=
        len(match_to_library([sp("Song A Remix","X")], [loc("Song A","X")], threshold=95)[0])))


if __name__ == "__main__":
    print("\n" + "═"*52 + "\n  🧪 AutoDJ — Module 3 Tests\n" + "═"*52)
    test_extract()
    test_normalize()
    test_match_score()
    test_match_to_library()
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