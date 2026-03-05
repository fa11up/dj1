"""
Module 4b: DJ Brain
────────────────────
Plans a setlist via an AI API call (tool / function use).
Supports two providers:
  - anthropic          (default) — Claude via Anthropic SDK
  - openai_compatible  — any OpenAI-compatible endpoint:
                         Ollama, llama.cpp, LM Studio, vLLM, etc.

Config (session.yaml):
  brain:
    enabled: true
    provider: openai_compatible        # or "anthropic"
    model: qwen3:latest                # required for openai_compatible
    base_url: http://localhost:11434/v1

Falls back to None (triggering algorithmic planner) if:
  - required SDK not installed
  - required API key not set
  - API call fails (any exception)
  - Tool response is malformed or returns fewer than 2 tracks

Usage: called exclusively from modules/planner.run() via plan_session().
"""

import os
import json
import logging
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai as _openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert DJ with deep knowledge of electronic music, harmonic mixing,
and crowd dynamics. You are planning a DJ set from a provided track library.

## Camelot Wheel Rules (harmonic mixing)
Each key has a number (1–12) and letter (A=minor, B=major). Compatible moves:
1. Identical position (8A→8A): same key, always safe
2. Same number, opposite letter (8A→8B): relative major/minor, very smooth
3. Adjacent number, same letter (8A→9A or 8A→7A): energy shift, smooth
4. 12 wraps to 1 (12A→1A is adjacent)
5. Two-step (8A→10A): noticeable but usable for intentional tension
6. Avoid opposite-wheel jumps unless making a deliberate break

## BPM Mixing Thresholds
- delta < 4%:  long_blend or stem_blend (beatmatch perfectly, hold the blend)
- delta 4–8%:  crossfade (time-stretching will compensate)
- delta 8–12%: quick_fade (brief overlap, let energy shift)
- delta > 12%: hard_cut only (tempo clash will sound broken if blended)
- stem_blend:  ONLY when delta < 6% AND tracks are harmonically compatible (rules 1–3 above)

## Transition Type Guide
- stem_blend:  surgical stem-based mix — drums/bass/melody controlled independently.
               Use for 2–4 signature moments per set. Maximum impact.
- long_blend:  3-phase EQ blend over 30+ bars. Best for harmonic matches + close BPM.
               The classic DJ move — sounds professional, smooth.
- crossfade:   12s overlap. Good for moderate BPM delta or imperfect key match.
- quick_fade:  4s fade. Use for intentional genre/energy jumps.
- hard_cut:    Abrupt. Use for dramatic drops, genre breaks, or >12% BPM delta.
               First track in the set MUST use hard_cut (no previous track).

## Energy Arc Principles
- Start 10–15% below peak energy (give the set room to build)
- Avoid back-to-back energy drops (mean < 0.03 twice in a row)
- High danceability (> 0.85): effective at energy peaks
- "Sections" field shows track structure: use breakdown positions for timing blends
- Respect the requested energy curve shape

## Mood Interpretation (translate keywords to musical decisions)
- "dark / melancholic":    prefer A-suffix keys (minor), lower energy_mean, slower BPM
- "euphoric / uplifting":  B-suffix keys (major), high energy, 128+ BPM, long blends
- "driving / intense":     high danceability, consistent energy, minimal drops
- "late night / intimate": lower BPM (100–118), minor keys, very long blends
- "festival / anthemic":   build-and-release arcs, major keys, 128–145 BPM
- "underground / raw":     allow dissonant key jumps, harder transitions, lower danceability
- "coding / focus":        steady energy, avoid jarring jumps, consistent BPM

## Output Rules
- Use ONLY track_ids from the provided library — no hallucinated IDs
- Do not repeat a track_id in the setlist
- Target the requested track count (derived from duration)
- Every slot needs a transition_hint and specific reasoning
- reasoning: mention actual key, BPM values, energy numbers — be specific
- Seed tracks (is_seed: true) are Spotify-matched — prioritise in the first half
"""

# ── Tool schema — Anthropic format ────────────────────────────────────────────

_TOOL_INPUT_SCHEMA = {
    "type": "object",
    "required": ["setlist", "curve_type", "session_reasoning"],
    "properties": {
        "setlist": {
            "type": "array",
            "description": "Ordered list of tracks for the session.",
            "items": {
                "type": "object",
                "required": ["position", "track_id", "transition_hint", "reasoning"],
                "properties": {
                    "position": {
                        "type": "integer",
                        "description": "0-indexed position in the setlist."
                    },
                    "track_id": {
                        "type": "string",
                        "description": "Exact track_id from the provided library."
                    },
                    "transition_hint": {
                        "type": "string",
                        "enum": ["hard_cut", "quick_fade", "crossfade",
                                 "long_blend", "stem_blend"],
                        "description": (
                            "Transition type INTO this track from the previous. "
                            "First track must be hard_cut. "
                            "stem_blend requires BPM delta <6% and harmonic compatibility."
                        )
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "One sentence explaining this placement. "
                            "Reference specific values: key/Camelot, BPM, energy. "
                            "Example: '128→126 BPM (1.6%), 8A→8B harmonic step-up, "
                            "energy 0.62→0.71, long_blend to build pressure.'"
                        )
                    }
                }
            }
        },
        "curve_type": {
            "type": "string",
            "enum": ["flat", "ramp_up", "peak_valley", "festival", "random"],
            "description": "Energy curve shape that best describes the planned setlist arc."
        },
        "session_reasoning": {
            "type": "string",
            "description": (
                "2–3 sentences describing the overall arc of the set: "
                "how the mood_prompt was interpreted, what the energy trajectory is, "
                "and any notable structural decisions."
            )
        }
    }
}

_TOOL_DESCRIPTION = (
    "Output the planned DJ setlist as structured JSON. "
    "Each slot specifies which track to play, what transition type to use "
    "coming INTO this track, and a one-sentence musical reasoning."
)

PLAN_SESSION_TOOL_ANTHROPIC = {
    "name": "plan_dj_session",
    "description": _TOOL_DESCRIPTION,
    "input_schema": _TOOL_INPUT_SCHEMA,
}

PLAN_SESSION_TOOL_OAI = {
    "type": "function",
    "function": {
        "name": "plan_dj_session",
        "description": _TOOL_DESCRIPTION,
        "parameters": _TOOL_INPUT_SCHEMA,
    },
}


# ── Client factories ──────────────────────────────────────────────────────────

def _make_anthropic_client(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


def _make_openai_client(api_key: str, base_url: str):
    return _openai.OpenAI(api_key=api_key, base_url=base_url)


# ── Track summary builder ─────────────────────────────────────────────────────

def _build_track_summary(track: dict, is_seed: bool = False) -> dict:
    energy = track.get("energy") or {}
    sections = [
        {
            "label": s["label"],
            "start_sec": round(s["start_sec"], 1),
            "energy_category": s.get("energy_category", "mid"),
        }
        for s in (track.get("sections") or [])
    ]
    return {
        "track_id":    track["track_id"],
        "filename":    track["filename"],
        "bpm":         round(track.get("bpm_normalised") or track.get("bpm") or 0, 1),
        "camelot":     track.get("camelot"),
        "key_name":    track.get("key_name"),
        "mode":        track.get("mode"),
        "energy_mean": round(energy.get("mean", 0) or 0, 4),
        "danceability": round(track.get("danceability") or 0, 3),
        "intro_end_sec":   round(track.get("intro_end_sec") or 0, 1),
        "outro_start_sec": round(track.get("outro_start_sec") or 0, 1),
        "sections":    sections,
        "is_seed":     is_seed,
    }


# ── User message builder ──────────────────────────────────────────────────────

def _build_user_message(
    track_summaries: list[dict],
    mood_prompt: str,
    duration_min: int,
    target_n: int,
    chaos_factor: float,
    energy_curve: str,
    seed_ids: set,
) -> str:
    if chaos_factor < 0.3:
        chaos_note = "Low chaos: prefer safe harmonic transitions, gradual energy changes."
    elif chaos_factor > 0.7:
        chaos_note = "High chaos: be adventurous — allow energy jumps, unexpected key moves, genre pivots."
    else:
        chaos_note = "Medium chaos: balance predictability with surprise."

    seed_names = [t["filename"] for t in track_summaries if t.get("is_seed")]
    seed_note = ""
    if seed_names:
        seed_note = (
            f"\nSeed tracks (Spotify-matched, prioritise in first half): "
            f"{', '.join(seed_names[:8])}"
        )

    library_json = json.dumps(track_summaries, ensure_ascii=False)

    return (
        f"Plan a DJ set with these parameters:\n"
        f"  Mood: {mood_prompt or 'no specific mood requested'}\n"
        f"  Duration: ~{duration_min} minutes (~{target_n} tracks)\n"
        f"  Energy curve: {energy_curve}\n"
        f"  Chaos factor ({chaos_factor:.2f}/1.0): {chaos_note}"
        f"{seed_note}\n\n"
        f"Track library ({len(track_summaries)} tracks available):\n"
        f"{library_json}\n\n"
        f"Call plan_dj_session with your ordered setlist. "
        f"Select exactly {target_n} tracks (or as close as possible without repeating)."
    )


# ── Response parsers ──────────────────────────────────────────────────────────

def _parse_anthropic_response(response) -> Optional[dict]:
    for block in response.content:
        if block.type == "tool_use" and block.name == "plan_dj_session":
            return block.input
    log.error("brain: no tool_use block found in Anthropic response")
    return None


def _parse_openai_response(response) -> Optional[dict]:
    msg = response.choices[0].message
    if not msg.tool_calls:
        log.error("brain: no tool_calls in OpenAI-compatible response")
        return None
    for tc in msg.tool_calls:
        if tc.function.name == "plan_dj_session":
            return json.loads(tc.function.arguments)
    log.error("brain: plan_dj_session not found in tool_calls")
    return None


# ── Setlist re-hydrator ───────────────────────────────────────────────────────

def _rehydrate_setlist(
    brain_slots: list[dict],
    analyses_by_id: dict,
    seed_ids: set,
    config: dict,
) -> list[dict]:
    dj            = config.get("dj", {})
    bpm_normalize = bool(dj.get("bpm_normalize", False))

    setlist  = []
    prev_bpm = None
    seen_ids = set()

    for slot in brain_slots:
        track_id = slot.get("track_id")
        if not track_id:
            continue
        if track_id not in analyses_by_id:
            log.warning(f"brain: unknown track_id {track_id!r} — skipping")
            continue
        if track_id in seen_ids:
            log.warning(f"brain: duplicate track_id {track_id!r} — skipping")
            continue
        seen_ids.add(track_id)

        a = analyses_by_id[track_id]
        plan_bpm = (a.get("bpm_normalised") if bpm_normalize else None) or a.get("bpm")
        energy   = a.get("energy") or {}

        bpm_delta_pct = 0.0
        if prev_bpm and plan_bpm:
            bpm_delta_pct = round(abs(prev_bpm - plan_bpm) / prev_bpm * 100, 1)

        rehydrated = {
            "position":               len(setlist),
            "track_id":               track_id,
            "filepath":               a["filepath"],
            "filename":               a["filename"],
            "actual_bpm":             plan_bpm,
            "bpm_raw":                a.get("bpm"),
            "camelot":                a.get("camelot"),
            "key_name":               a.get("key_name"),
            "mode":                   a.get("mode"),
            "energy_target":          0.65,
            "energy_actual":          round(energy.get("mean", 0) or 0, 4),
            "danceability":           a.get("danceability"),
            "intro_end_sec":          a.get("intro_end_sec"),
            "outro_start_sec":        a.get("outro_start_sec"),
            "beat_times":             a.get("beat_times"),
            "spectral_centroid_mean": a.get("spectral_centroid_mean"),
            "sections":               a.get("sections"),
            "transition_hint":        slot.get("transition_hint", "crossfade"),
            "bpm_delta_pct":          bpm_delta_pct,
            "is_seed":                track_id in seed_ids,
            "brain_reasoning":        slot.get("reasoning", ""),
        }
        setlist.append(rehydrated)
        prev_bpm = plan_bpm

    return setlist


# ── Public entry point ────────────────────────────────────────────────────────

def plan_session(
    analyses: list[dict],
    seed_ids: set,
    config: dict,
    target_n: int,
) -> Optional[tuple[list[dict], str]]:
    """
    Call AI API to plan a DJ session.

    Args:
        analyses:  list of analysis dicts from analysis.json
        seed_ids:  set of track_ids from Spotify matching (prefer these)
        config:    parsed session.yaml dict
        target_n:  target number of tracks (computed from session duration)

    Returns:
        (setlist, curve_type) matching build_session() return shape,
        or None on any failure (triggers algorithmic fallback in planner).
    """
    brain_cfg = config.get("brain", {})
    provider  = brain_cfg.get("provider", "anthropic").strip().lower()
    model_cfg = (brain_cfg.get("model") or "").strip()
    base_url  = (brain_cfg.get("base_url") or "").strip()

    sess = config.get("session", {})
    dj   = config.get("dj",      {})

    mood_prompt  = sess.get("mood_prompt", "")
    duration_min = sess.get("duration_minutes", 60)
    chaos        = float(dj.get("chaos_factor", 0.3))
    energy_curve = sess.get("energy_curve", "auto")

    # ── Build pool + summaries ────────────────────────────────────────────────
    pool = [a for a in analyses if not a.get("analysis_error") and a.get("bpm")]
    if not pool:
        log.warning("brain: empty track pool — falling back")
        return None

    analyses_by_id  = {a["track_id"]: a for a in pool}
    track_summaries = [_build_track_summary(t, is_seed=t["track_id"] in seed_ids) for t in pool]

    user_msg = _build_user_message(
        track_summaries=track_summaries,
        mood_prompt=mood_prompt,
        duration_min=duration_min,
        target_n=target_n,
        chaos_factor=chaos,
        energy_curve=energy_curve,
        seed_ids=seed_ids,
    )

    # ── Dispatch to provider ──────────────────────────────────────────────────
    if provider == "openai_compatible":
        brain_output = _call_openai_compatible(user_msg, model_cfg, base_url)
    else:
        brain_output = _call_anthropic(user_msg, model_cfg)

    if brain_output is None:
        return None

    # ── Re-hydrate ────────────────────────────────────────────────────────────
    try:
        setlist = _rehydrate_setlist(
            brain_output["setlist"],
            analyses_by_id,
            seed_ids,
            config,
        )
        if len(setlist) < 2:
            log.warning(f"brain: only {len(setlist)} tracks returned — falling back")
            return None
    except Exception as e:
        log.error(f"brain: re-hydration failed: {e} — falling back")
        return None

    curve_type        = brain_output.get("curve_type", "flat")
    session_reasoning = brain_output.get("session_reasoning", "")

    print(f"  Brain: {len(setlist)} tracks planned, curve: {curve_type}")
    if session_reasoning:
        print(f"  Brain: {session_reasoning}")

    return setlist, curve_type


# ── Provider implementations ──────────────────────────────────────────────────

def _call_anthropic(user_msg: str, model_cfg: str) -> Optional[dict]:
    if not ANTHROPIC_AVAILABLE:
        log.warning("brain: anthropic SDK not installed — falling back")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("brain: ANTHROPIC_API_KEY not set — falling back")
        return None

    model = model_cfg or DEFAULT_ANTHROPIC_MODEL
    print(f"  Brain: calling Anthropic API ({model})...")

    try:
        client   = _make_anthropic_client(api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[PLAN_SESSION_TOOL_ANTHROPIC],
            tool_choice={"type": "tool", "name": "plan_dj_session"},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        log.error(f"brain: Anthropic API call failed: {e} — falling back")
        return None

    try:
        return _parse_anthropic_response(response)
    except Exception as e:
        log.error(f"brain: Anthropic response parsing failed: {e} — falling back")
        return None


def _call_openai_compatible(user_msg: str, model_cfg: str, base_url: str) -> Optional[dict]:
    if not OPENAI_AVAILABLE:
        log.warning("brain: openai SDK not installed — run: pip install openai")
        return None

    if not base_url:
        log.warning("brain: brain.base_url not set for openai_compatible provider — falling back")
        return None

    if not model_cfg:
        log.warning("brain: brain.model not set for openai_compatible provider — falling back")
        return None

    api_key = os.environ.get("BRAIN_API_KEY", "none")
    print(f"  Brain: calling OpenAI-compatible endpoint ({base_url}, model={model_cfg})...")

    try:
        client   = _make_openai_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model_cfg,
            max_tokens=4096,
            tools=[PLAN_SESSION_TOOL_OAI],
            tool_choice={"type": "function", "function": {"name": "plan_dj_session"}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
    except Exception as e:
        log.error(f"brain: OpenAI-compatible API call failed: {e} — falling back")
        return None

    try:
        return _parse_openai_response(response)
    except Exception as e:
        log.error(f"brain: OpenAI-compatible response parsing failed: {e} — falling back")
        return None
