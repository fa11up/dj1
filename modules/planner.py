"""
Module 4: Session Planner
─────────────────────────
Reads analysis.json + optional matched_tracks.json + session.yaml and produces
session.json — an ordered setlist with transition hints for the renderer.

Key behaviours:
  - Seed tracks (from Spotify match) are placed first, remainder filled from
    the full library ranked by compatibility
  - Energy curve shapes target energy across the session; chaos_factor both
    randomises track selection AND determines the curve itself at runtime
  - Harmonic mixing via Camelot Wheel (compatible = same number ±1, or same key)
  - BPM compatibility window from session.yaml (tempo_tolerance_bpm)
  - Recency penalty prevents the same track appearing twice
  - chaos_factor 0.0 = safe/predictable, 1.0 = anything goes

Output session.json schema per slot:
  {track_id, filepath, filename, position, target_bpm, actual_bpm,
   camelot, energy_target, intro_end_sec, outro_start_sec,
   transition_hint, bpm_delta_pct}
"""

import sys, json, math, random, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

VALID_CURVES = {"flat", "ramp_up", "peak_valley", "festival", "random"}
# ── Camelot compatibility ─────────────────────────────────────────────────────

def camelot_compatible(a, b):
    """
    Two Camelot positions are compatible if:
      - They are identical (same key)
      - Same number, different letter (relative major/minor: e.g. 8A↔8B)
      - Adjacent number, same letter (e.g. 8A→9A or 8A→7A)
    """
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        a_num, a_let = int(a[:-1]), a[-1]
        b_num, b_let = int(b[:-1]), b[-1]
    except (ValueError, IndexError):
        return False
    if a_num == b_num:           # same number, A↔B = relative major/minor
        return True
    if a_let == b_let:           # same letter, adjacent number = energy shift
        return abs(a_num - b_num) == 1 or abs(a_num - b_num) == 11  # wrap 12→1
    return False


# ── Energy curves ─────────────────────────────────────────────────────────────

def make_energy_curve(n, curve_type, chaos=0.0, rng=None):
    """
    Generate a target energy value (0–1) for each of n positions.

    curve_type: ramp_up | peak_valley | festival | flat | random
    chaos adds gaussian noise scaled to chaos_factor.
    """
    if rng is None:
        rng = random.Random()

    t = np.linspace(0, 1, n)

    if curve_type == "flat":
        base = np.ones(n) * 0.65

    elif curve_type == "ramp_up":
        base = 0.3 + 0.65 * t

    elif curve_type == "peak_valley":
        # Two peaks at ~35% and ~80% through the set
        base = 0.45 + 0.4 * (np.sin(2 * np.pi * t * 1.5 - 0.3) * 0.5 + 0.5)

    elif curve_type == "festival":
        # Slow build → main peak at 70% → comedown
        base = np.where(
            t < 0.7,
            0.3 + 0.65 * (t / 0.7),          # build phase
            0.95 - 0.5 * ((t - 0.7) / 0.3),  # comedown phase
        )

    else:  # random / unknown
        base = np.array([rng.uniform(0.3, 0.95) for _ in range(n)])

    # Add chaos noise
    if chaos > 0:
        noise = np.array([rng.gauss(0, chaos * 0.2) for _ in range(n)])
        base  = np.clip(base + noise, 0.15, 1.0)

    return [round(float(v), 3) for v in base]


def choose_curve(chaos, rng, explicit_curve=None):
    """
    Decide which energy curve to use.
    CHANGED: If explicit_curve is a valid curve name from the YAML config,
    use it directly. Only fall back to chaos-based randomization when the
    curve is "auto", None, or not a recognized value.
    This fixes the bug where setting energy_curve: "peak_valley" in YAML
    was ignored and the planner picked "festival" randomly.
    """
    # If the user explicitly set a valid curve in YAML, respect it
    if explicit_curve and explicit_curve in VALID_CURVES:
        return explicit_curve
    # "auto" or missing — original chaos-based logic
    if chaos >= 0.7:
        return rng.choice(["ramp_up", "peak_valley", "festival", "flat", "random"])
    if chaos >= 0.4:
        return rng.choice(["ramp_up", "peak_valley", "festival"])
    return "festival"



# ── Track scoring ─────────────────────────────────────────────────────────────

def score_candidate(candidate, prev_track, target_energy, tempo_tolerance,
                    harmonic_strict, chaos, rng, bpm_field="bpm"):
    """
    Score a candidate track (0–100) for placement after prev_track at a given
    target_energy. Higher = better fit.

    Components:
      - Energy proximity  (40 pts) — how close actual energy is to target
      - Harmonic compat   (30 pts) — Camelot wheel compatibility with prev track
      - BPM proximity     (20 pts) — within tempo_tolerance_bpm
      - Chaos wildcard    (10 pts) — random bonus scaled to chaos_factor

    CHANGED: Uses bpm_field parameter to select which BPM value to compare.
    When bpm_normalize is on, this will be "_plan_bpm" (the corrected value).
    """
    score = 0.0

    # ── Energy proximity ──────────────────────────────────────────────────────
    cand_energy = candidate.get("energy", {}).get("mean", 0.5) or 0.5
    energy_norm = min(1.0, max(0.0, (math.log(max(cand_energy, 0.001)) - math.log(0.005)) /
                                     (math.log(0.15) - math.log(0.005))))
    energy_score = max(0.0, 1.0 - abs(energy_norm - target_energy) * 2)
    score += energy_score * 40

    # ── Harmonic compatibility ─────────────────────────────────────────────────
    if prev_track:
        compat = camelot_compatible(prev_track.get("camelot"), candidate.get("camelot"))
        if compat:
            score += 30
        elif not harmonic_strict:
            score += 10
    else:
        score += 20

    # ── BPM proximity (uses normalised BPM when available) ────────────────────
    cand_bpm = candidate.get(bpm_field) or candidate.get("bpm")
    prev_bpm = (prev_track.get(bpm_field) or prev_track.get("bpm")) if prev_track else None

    if prev_bpm and cand_bpm:
        bpm_delta = abs(prev_bpm - cand_bpm)
        if bpm_delta <= tempo_tolerance:
            score += 20 * (1 - bpm_delta / tempo_tolerance)
    else:
        score += 10

    # ── Chaos wildcard ────────────────────────────────────────────────────────
    score += rng.uniform(0, chaos * 10)

    return round(score, 2)
# ── Transition hints ──────────────────────────────────────────────────────────

def transition_hint(prev_track, next_track, style):
    """
    Suggest a transition type for the renderer based on BPM delta and style.

    hard_cut      — abrupt stop/start (no blending)
    quick_fade    — short crossfade (4–8s)
    crossfade     — standard crossfade (8–16s)
    long_blend    — slow blend (16–32s), good for harmonic matches
    """
    if not prev_track:
        return "hard_cut"

    bpm_delta_pct = 0.0
    if prev_track.get("bpm") and next_track.get("bpm"):
        bpm_delta_pct = abs(prev_track["bpm"] - next_track["bpm"]) / prev_track["bpm"]

    if style == "cuts":
        return "hard_cut"

    if bpm_delta_pct > 0.12:
        return "hard_cut"

    if style == "blends":
        if camelot_compatible(prev_track.get("camelot"), next_track.get("camelot")):
            return "long_blend"
        return "crossfade"

    # mixed — choose based on BPM delta
    if bpm_delta_pct < 0.04:
        return "long_blend" if camelot_compatible(prev_track.get("camelot"), next_track.get("camelot")) else "crossfade"
    if bpm_delta_pct < 0.08:
        return "crossfade"
    return "quick_fade"


# ── Core planner ──────────────────────────────────────────────────────────────

def build_session(analyses, seed_ids, config, rng):
    """
    Build an ordered setlist from the analyzed track pool.

    analyses:  list of analysis dicts from Module 2
    seed_ids:  set of track_ids to prioritise (from Module 3)
    config:    parsed session.yaml dict
    rng:       seeded Random instance

    Returns (setlist, curve_type) tuple.

    CHANGED:
    - Reads session.energy_curve from config → passes to choose_curve()
    - Reads dj.bpm_normalize → uses bpm_normalised field from analysis
    - Reads dj.transition_style (already worked, just documenting)
    - Passes mood_prompt through for session.json output
    """
    sess     = config.get("session", {})
    dj       = config.get("dj", {})

    duration_min      = sess.get("duration_minutes", 60)
    chaos             = float(dj.get("chaos_factor", 0.3))
    harmonic_strict   = bool(dj.get("harmonic_strict", False))
    tempo_tolerance   = float(dj.get("tempo_tolerance_bpm", 8))
    transition_style  = dj.get("transition_style", "mixed")
    bpm_normalize     = bool(dj.get("bpm_normalize", False))

    # NEW: Read the explicit energy curve from YAML
    explicit_curve    = sess.get("energy_curve")

    # Choose which BPM field to use for scoring and output
    # When bpm_normalize is on, prefer the corrected value from the analyzer
    bpm_field = "bpm_normalised" if bpm_normalize else "bpm"

    # Filter to tracks that actually have usable analysis
    pool = [a for a in analyses if not a.get("analysis_error") and a.get("bpm")]

    if not pool:
        print("  ✗ No usable tracks in analysis pool.")
        return []

    # ── Resolve BPM for each track ────────────────────────────────────────────
    # Ensure every track has the chosen BPM field available.
    # Fall back to raw bpm if bpm_normalised is missing (pre-P2 analysis cache).
    for track in pool:
        if bpm_normalize and not track.get("bpm_normalised"):
            # Analysis was run before P2-2 normalisation — fall back to raw
            track["_plan_bpm"] = track.get("bpm")
        elif bpm_normalize:
            track["_plan_bpm"] = track.get("bpm_normalised")
        else:
            track["_plan_bpm"] = track.get("bpm")

    # ── Decide energy curve — FIXED: respect YAML value ───────────────────────
    curve_type = choose_curve(chaos, rng, explicit_curve=explicit_curve)

    # Estimate how many tracks fit in the session duration
    avg_track_sec  = 240.0  # assume 4 min average
    target_n       = max(4, int((duration_min * 60) / avg_track_sec))
    target_n       = min(target_n, len(pool))

    energy_targets = make_energy_curve(target_n, curve_type, chaos=chaos, rng=rng)

    print(f"  Curve: {curve_type}  |  Target slots: {target_n}  |  Pool: {len(pool)} tracks")
    if explicit_curve and explicit_curve in VALID_CURVES:
        print(f"  (energy_curve from YAML: '{explicit_curve}')")
    if bpm_normalize:
        print(f"  BPM normalisation: ON (using corrected BPM values)")

    # ── Seed track prioritisation ──────────────────────────────────────────────
    seed_pool  = [a for a in pool if a["track_id"] in seed_ids]
    fill_pool  = [a for a in pool if a["track_id"] not in seed_ids]

    # Shuffle fill pool slightly — chaos controls how much
    fill_pool  = sorted(fill_pool, key=lambda _: rng.random() * chaos + (1 - chaos))

    used_ids   = set()
    setlist    = []
    prev_track = None

    for position, target_energy in enumerate(energy_targets):
        # Candidate pool: prefer seeds for early positions if available
        use_seeds = seed_pool and position < len(seed_pool) and rng.random() > chaos * 0.5
        candidates = [t for t in (seed_pool if use_seeds else fill_pool)
                      if t["track_id"] not in used_ids]

        # Fall back to fill pool if seeds exhausted
        if not candidates:
            candidates = [t for t in fill_pool if t["track_id"] not in used_ids]
        if not candidates:
            candidates = [t for t in pool if t["track_id"] not in used_ids]
        if not candidates:
            break

        # Score all candidates, pick from top N (top-3 at low chaos, more at high chaos)
        top_n = max(1, int(1 + chaos * (len(candidates) - 1) * 0.3))
        scored = sorted(
            candidates,
            key=lambda c: score_candidate(c, prev_track, target_energy,
                                          tempo_tolerance, harmonic_strict, chaos, rng,
                                          bpm_field="_plan_bpm"),
            reverse=True,
        )
        chosen = rng.choice(scored[:top_n])

        # Use the planning BPM for delta calculation
        plan_bpm = chosen.get("_plan_bpm") or chosen.get("bpm")
        prev_plan_bpm = (prev_track.get("_plan_bpm") or prev_track.get("bpm")) if prev_track else None

        bpm_delta_pct = 0.0
        if prev_plan_bpm and plan_bpm:
            bpm_delta_pct = round(
                abs(prev_plan_bpm - plan_bpm) / prev_plan_bpm * 100, 1
            )

        setlist.append({
            "position":        position,
            "track_id":        chosen["track_id"],
            "filepath":        chosen["filepath"],
            "filename":        chosen["filename"],
            "actual_bpm":      plan_bpm,           # CHANGED: uses normalised BPM when enabled
            "bpm_raw":         chosen.get("bpm"),   # NEW: always include raw for reference
            "camelot":         chosen.get("camelot"),
            "key_name":        chosen.get("key_name"),
            "mode":            chosen.get("mode"),
            "energy_target":   target_energy,
            "energy_actual":   round(chosen.get("energy", {}).get("mean", 0) or 0, 4),
            "danceability":    chosen.get("danceability"),
            "intro_end_sec":   chosen.get("intro_end_sec"),
            "outro_start_sec": chosen.get("outro_start_sec"),
            "beat_times":      chosen.get("beat_times"),
            "spectral_centroid_mean": chosen.get("spectral_centroid_mean"),
            "sections":        chosen.get("sections"),
            "transition_hint": transition_hint(prev_track, chosen, transition_style),
            "bpm_delta_pct":   bpm_delta_pct,
            "is_seed":         chosen["track_id"] in seed_ids,
        })

        used_ids.add(chosen["track_id"])
        # Remove from whichever pool it came from
        seed_pool = [t for t in seed_pool if t["track_id"] != chosen["track_id"]]
        fill_pool = [t for t in fill_pool if t["track_id"] != chosen["track_id"]]
        prev_track = chosen

    return setlist, curve_type


# ── Display ───────────────────────────────────────────────────────────────────

def print_setlist(setlist, curve_type, duration_min):
    total = len(setlist)
    seeds = sum(1 for s in setlist if s.get("is_seed"))

    print(f"\n  {'─'*62}")
    print(f"  Session Plan  —  {total} tracks  ~{duration_min} min  curve: {curve_type}")
    if seeds:
        print(f"  Seed tracks: {seeds}  |  Fill: {total - seeds}")
    print(f"  {'─'*62}")
    print(f"  {'#':>3}  {'Filename':<28}  {'BPM':>6}  {'Key':<7}  {'Camelot':<5}  {'Trans':<12}  E↑")
    print(f"  {'─'*62}")

    for s in setlist:
        pos      = str(s["position"] + 1).rjust(3)
        name     = s["filename"][:26]
        bpm      = f"{s['actual_bpm']:.1f}" if s["actual_bpm"] else "?"
        key      = f"{s['key_name']} {s['mode'][:3]}" if s["key_name"] else "?"
        camelot  = s["camelot"] or "?"
        trans    = s["transition_hint"]
        bar_len  = int(s["energy_target"] * 10)
        bar      = "█" * bar_len + "░" * (10 - bar_len)
        seed_tag = "🌱" if s["is_seed"] else "  "
        print(f"  {pos}  {name:<28}  {bpm:>6}  {key:<7}  {camelot:<5}  {trans:<12}  {bar} {seed_tag}")

    print(f"  {'─'*62}\n")


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    analysis_path       = "./data/analysis.json",
    matched_tracks_path = "./data/matched_tracks.json",
    config_path         = "./config/session.yaml",
    output_path         = "./data/session.json",
):
    print("─" * 60 + "\n  AutoDJ — Module 4: Session Planner\n" + "─" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    if not Path(analysis_path).exists():
        sys.exit(f"✗ {analysis_path} not found — run Module 2 first.")
    if not Path(config_path).exists():
        sys.exit(f"✗ {config_path} not found.")

    analyses = json.loads(Path(analysis_path).read_text())["analyses"]
    config   = yaml.safe_load(Path(config_path).read_text())

    # Load seed track IDs from Module 3 (optional)
    seed_ids = set()
    if Path(matched_tracks_path).exists():
        matched = json.loads(Path(matched_tracks_path).read_text())
        seed_ids = {m["local_track_id"] for m in matched.get("matched_tracks", [])}
        print(f"\n  Seed tracks loaded: {len(seed_ids)} from {matched_tracks_path}")
    else:
        print(f"\n  No matched_tracks.json found — planning from full library")

    # ── RNG setup ─────────────────────────────────────────────────────────────
    dj           = config.get("dj", {})
    sess         = config.get("session", {})
    random_seed  = dj.get("random_seed")
    chaos        = float(dj.get("chaos_factor", 0.3))
    duration_min = sess.get("duration_minutes", 60)

    # NEW: Read config values that were previously ignored
    mood_prompt      = sess.get("mood_prompt")         # pass-through for Claude integration
    transition_style = dj.get("transition_style", "mixed")
    bpm_normalize    = bool(dj.get("bpm_normalize", False))
    energy_curve_cfg = sess.get("energy_curve")        # now actually used!

    if random_seed is not None:
        rng = random.Random(random_seed)
        print(f"  Random seed: {random_seed} (reproducible run)")
    else:
        rng = random.Random()

    print(f"  Chaos factor: {chaos}  |  Duration: {duration_min} min")
    print(f"  Pool: {len([a for a in analyses if not a.get('analysis_error')])} usable tracks")
    if energy_curve_cfg:
        print(f"  Energy curve (YAML): {energy_curve_cfg}")
    if mood_prompt:
        print(f"  Mood prompt: \"{mood_prompt}\"")
    if bpm_normalize:
        print(f"  BPM normalisation: enabled")
    print()

    # ── Build setlist ─────────────────────────────────────────────────────────
    result = build_session(analyses, seed_ids, config, rng)
    if not result:
        sys.exit("✗ No session could be planned — check your track pool.")

    setlist, curve_type = result

    print_setlist(setlist, curve_type, duration_min)

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "generated_at":     datetime.now().isoformat(),
        "curve_type":       curve_type,
        "chaos_factor":     chaos,
        "random_seed":      random_seed,
        "duration_min":     duration_min,
        "total_tracks":     len(setlist),
        "seed_count":       sum(1 for s in setlist if s.get("is_seed")),
        # NEW: config values written to session.json for downstream modules
        "transition_style": transition_style,
        "bpm_normalize":    bpm_normalize,
        "mood_prompt":      mood_prompt,        # pass-through for Claude integration
        "energy_curve_cfg": energy_curve_cfg,   # what the user asked for in YAML
        "setlist":          setlist,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  ✅ session.json → {Path(output_path).resolve()}\n" + "─" * 60)
    return output



# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDJ Module 4 — Session Planner")
    parser.add_argument("--analysis",  "-a", default="./data/analysis.json")
    parser.add_argument("--matched",   "-m", default="./data/matched_tracks.json")
    parser.add_argument("--config",    "-c", default="./config/session.yaml")
    parser.add_argument("--output",    "-o", default="./data/session.json")
    args = parser.parse_args()

    run(analysis_path=args.analysis, matched_tracks_path=args.matched,
        config_path=args.config, output_path=args.output)
