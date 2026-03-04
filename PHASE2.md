# dj1 — Phase 2 Planning Document

**For new Claude sessions:** This document describes the current state of
dj1 after Phase 1, what works, what needs improvement, and what Phase 2
should build. Read this alongside the existing module code before making any
changes.

---

## What Phase 1 Built

A fully working 5-module pipeline:

```
ingestor → analyzer → spotify_bridge → planner → renderer
```

1. **Ingestor** — scans a local music directory, extracts metadata (title,
   artist, BPM tag, duration), writes `data/tracks.json`

2. **Analyzer** — runs librosa analysis on every track: BPM detection, key/mode,
   Camelot wheel position, RMS energy curve (32 segments), beat grid timestamps,
   spectral centroid, danceability heuristic, intro/outro cue points. Writes
   `data/analysis.json` with per-track caching.

3. **Spotify Bridge** — authenticates via OAuth, fetches a Spotify playlist,
   fuzzy-matches track names/artists to local library files. Outputs
   `data/matched_tracks.json` (local filepaths + match scores). Note: Spotify
   `/audio-features` endpoint is restricted since late 2024 — no tempo/energy
   data from Spotify is available.

4. **Planner** — builds an ordered setlist from the analyzed pool. Seeds with
   Spotify-matched tracks, fills the rest from the full library. Energy curve
   shapes target energy per position (`flat`, `ramp_up`, `peak_valley`,
   `festival`). At high `chaos_factor`, the curve type itself is randomised.
   Scores each candidate track on: energy proximity (40pts), Camelot harmonic
   compatibility (30pts), BPM proximity (20pts), chaos wildcard (10pts).
   Outputs `data/session.json`.

5. **Renderer** — reads `session.json` and renders a continuous MP3/WAV.
   Implements a real DJ transition pipeline (see below). `--preview` mode for
   fast iteration, `--quality` for final output.

### Renderer transition pipeline (as built)

The renderer has four transition types:

**`long_blend` → staged 4-phase DJ transition:**
- Phase 1: Waits for energy dip in outgoing track (breakdown detection via
  energy curve), then delays to next 8-bar phrase boundary
- Phase 2: Incoming track enters with bass cut (highs only) — 8 bars at
  fast BPM (≥120), 16 bars at slow BPM
- Phase 3: Bass swap — outgoing bass cut, incoming bass in simultaneously.
  1–2 bars tight at low chaos, 3–4 bars loose at high chaos
- Phase 4: Outgoing highs fade out over 16 bars, incoming owns full spectrum

**EQ is adaptive:**
- Crossover frequency derived from `spectral_centroid_mean` (dark/DnB tracks
  get ~60–80Hz crossover, House gets ~120–150Hz)
- Filter shape: cascaded double filter (hard/steep) at chaos < 0.4,
  single shelf (gentle rolloff) at chaos ≥ 0.4

**`crossfade`** — overlapping volume fade with EQ bass-cut at low chaos

**`quick_fade`** — short volume fade, no EQ

**`hard_cut`** — sequential 2s tail-fade on outgoing + 2s head-fade on
incoming (no overlap — a breath between tracks)

**Energy override** — regardless of planner hint, if both endpoints are soft
(RMS < threshold), renderer upgrades to `long_blend`. If one side is soft and
hint is `hard_cut`, upgrades to `crossfade`.

### Key design decisions made in Phase 1

- **Chaos factor is a single dial** controlling: energy curve randomness, track
  selection adventurousness, transition complexity, EQ filter sharpness
- **No Spotify audio features** — all musical analysis is local (librosa)
- **BPM half/double-time is untouched** — the renderer trusts the analyzer;
  correction deferred to Phase 2
- **Spotify Bridge is deliberately slim** — it only does name-matching, no
  aggregation or mood derivation
- **`session.yaml`** is the single config file a user edits before each run

### What's working well (user feedback)
- Playlist structure and setlist flow: "pretty good, grooving"
- Output renders successfully end-to-end
- Harmonic mixing noticeably coherent at low chaos
- Energy curve shapes detectable in the output

### Known issues going into Phase 2
- Transitions still need work — the staged pipeline is architecturally correct
  but more iteration needed on blend quality
- BPM detection half/double-time errors affect transition quality
- No looping support (extending a 4-bar loop before incoming enters)
- No sample/soundbite injection
- `main.py` pipeline orchestrator has Module 4 and 5 stubs but isn't fully wired

---

## Phase 2 Priorities

Listed in order of impact based on user feedback and known gaps.

---

### P2-1: Transition quality — looping

**What:** Before Phase 2 enters the incoming track, loop the last 4 or 8 bars
of the outgoing track. This gives the DJ more time to bring in the new track
cleanly without the outgoing track running out of content.

**Why:** The most common DJ technique for tricky transitions. Solves cases where
the outgoing track's outro is too short for a full staged blend. Also fixes
transitions where the outro energy drops too fast.

**How:**
- Add `loop_segment(seg, loop_bars, bpm)` to renderer — slices the last N bars
  and repeats them using `pyrubberband` (already a dependency) or simple
  pydub concatenation
- In `render_staged`, before Phase 2, check if `len(outgoing) - p2_start < min_blend_ms`.
  If short, extend outgoing with a 4-bar or 8-bar loop
- Add `loop_length_bars` to `transition_strategy` output

**Complexity:** Medium. pyrubberband is already installed. pydub concat is trivial.

---

### P2-2: BPM half-time / double-time correction

**What:** librosa sometimes detects BPM at half or double the true tempo (e.g.
a 78 BPM track detected as 156, or a 128 BPM track detected as 64). This causes
the planner to place incompatible tracks together and the renderer to
time-stretch in the wrong direction.

**Why:** Affects a meaningful percentage of the library, especially DnB (high BPM
often detected as half-time) and slow tracks (sometimes doubled).

**How:**
- Add `normalise_bpm(bpm, target_range=(80, 175))` in `analyzer.py` — if BPM is
  outside target range, halve or double until it lands inside
- Target range: 80–175 covers House, Techno, DnB (at 160–175), Jungle
- Store both `bpm_raw` and `bpm_normalised` in analysis.json
- Planner and renderer use `bpm_normalised`
- Add `bpm_normalise: true` flag to `session.yaml` (already in config, just needs
  to be wired through)

**Complexity:** Low. Pure arithmetic, no new dependencies.

---

### P2-3: Sample tags system

**What:** Short audio snippets (2–8 seconds) that can be played over the
beginning of a track, over a transition, or at high-energy moments. Examples:
air horns, vocal stabs, drum fills, idents.

**Why:** User specifically requested this. Adds character and live-DJ feel.
Biggest differentiator from a simple crossfade tool.

**How (proposed architecture):**

```
samples/
  tags/
    airhorn_01.wav
    vocal_stab_hey.wav
    drum_fill_4bar.wav
  tags.json           ← sample metadata: name, duration, energy, type, tags[]
```

New module: `modules/sampler.py`
- `ingest_samples(samples_dir)` — scans tags/ dir, writes tags.json
- `pick_sample(context)` — given energy level, transition type, genre tags,
  picks an appropriate sample
- `inject_sample(mix_seg, position_ms, sample_seg, volume_db)` — overlays
  sample onto rendered mix at correct position

Integration into renderer:
- After `render_staged` phase 2/3, optionally overlay a sample at the swap point
- Chaos factor gates sample injection probability (low chaos = rare, high chaos = frequent)
- Samples respect beat grid (position snapped to nearest beat)

**Complexity:** Medium-high. New module + integration touches renderer.

---

### P2-4: Transition style per genre pair

**What:** Different genre combinations need different transition defaults.
DnB→DnB wants hard drops. House→House wants long blends. Techno→House is a
careful crossfade. Currently all transitions go through the same logic.

**Why:** User library is DnB / House / Techno — these have meaningfully different
DJ conventions.

**How:**
- Add `genre` field to analysis.json (can be inferred from BPM range + spectral
  centroid, or read from file metadata tags)
- Add `GENRE_TRANSITION_DEFAULTS` lookup table in planner.py:
  ```python
  {
    ("dnb", "dnb"):       "hard_cut",
    ("house", "house"):   "long_blend",
    ("techno", "techno"): "long_blend",
    ("house", "techno"):  "crossfade",
    ("dnb", "house"):     "crossfade",
    # etc.
  }
  ```
- Planner uses this to set `transition_hint` more intelligently than pure
  Camelot/BPM scoring
- User can override per-pair in session.yaml

**Complexity:** Low-medium. Main work is the genre inference heuristic.

---

### P2-5: Main pipeline orchestrator completion

**What:** `main.py` has Module 4 and 5 stubs but isn't fully wired. Running the
full pipeline end-to-end via `main.py` should work without manual intervention.

**How:**
- Wire Module 4 (planner) and Module 5 (renderer) into `main.py`
- Pass config values through correctly (chaos, duration, output path)
- `--from-module N` flag already exists, just needs the stubs replaced
- Add summary output at end: tracks analysed, setlist length, output file, render time

**Complexity:** Low. Mostly plumbing.

---

### P2-6: Analyser improvements

**What:**
- Detect track structure sections (intro / verse / chorus / breakdown / outro)
  beyond just intro_end and outro_start cue points. This would make
  breakdown detection in the renderer more precise.
- Improve intro/outro cue point detection — current heuristic is energy-based
  but misses some tracks

**How:**
- Use librosa's `segment` module for structural segmentation
- Store `sections: [{label, start_sec, end_sec, energy}]` in analysis.json
- Renderer uses `sections` to find breakdown points instead of raw energy curve

**Complexity:** Medium. librosa structural analysis can be noisy; needs tuning.

---

## File Reference for New Sessions

All source files in `modules/`. Tests mirror module names in `tests/`.
Data files written to `data/` at runtime (gitignored).

| File | Purpose |
|------|---------|
| `modules/ingestor.py` | Library scan, 347 lines |
| `modules/analyzer.py` | Audio analysis, 568 lines |
| `modules/spotify_bridge.py` | Playlist matching, 171 lines |
| `modules/planner.py` | Setlist generation, 416 lines |
| `modules/renderer.py` | Mix rendering, ~865 lines |
| `config/session.yaml` | User config |
| `requirements.txt` | Python deps + system dep notes |
| `main.py` | Pipeline orchestrator (partially wired) |

## Key Data Schemas

### `data/tracks.json`
```json
{
  "tracks": [
    {
      "track_id": "abc123",
      "filepath": "/music/track.mp3",
      "filename": "track.mp3",
      "title": "Track Name",
      "artist": "Artist",
      "duration_sec": 240.0
    }
  ]
}
```

### `data/analysis.json`
```json
{
  "analyses": [
    {
      "track_id": "abc123",
      "filepath": "...",
      "bpm": 128.0,
      "bpm_confidence": 0.87,
      "beat_times": [0.47, 0.94, ...],
      "key_name": "C",
      "mode": "major",
      "camelot": "8B",
      "energy": {
        "mean": 0.072,
        "peak": 0.11,
        "dynamic_range": 0.04,
        "curve": [0.05, 0.07, ...]
      },
      "intro_end_sec": 8.0,
      "outro_start_sec": 220.0,
      "spectral_centroid_mean": 2400.0,
      "danceability": 0.78
    }
  ]
}
```

### `data/matched_tracks.json`
```json
{
  "playlist_name": "My Playlist",
  "match_stats": {"total": 20, "matched": 17, "match_rate_pct": 85.0},
  "matched_tracks": [
    {
      "name": "Track Name",
      "artist": "Artist",
      "local_track_id": "abc123",
      "local_filepath": "/music/track.mp3",
      "score": 94
    }
  ],
  "unmatched_tracks": [...]
}
```

### `data/session.json`
```json
{
  "curve_type": "festival",
  "chaos_factor": 0.3,
  "setlist": [
    {
      "position": 0,
      "track_id": "abc123",
      "filepath": "...",
      "actual_bpm": 128.0,
      "camelot": "8B",
      "energy_target": 0.45,
      "transition_hint": "long_blend",
      "is_seed": true
    }
  ]
}
```

---

## Suggested Phase 2 Starting Point

Recommended order for a new session:

1. **P2-5 first** — wire `main.py` fully. Gives a clean end-to-end run command
   before touching transition logic.
2. **P2-2** — BPM normalisation. Quick win, improves everything downstream.
3. **P2-1** — looping. Biggest audible improvement to transitions.
4. **P2-4** — genre-aware transition defaults. Completes the transition system.
5. **P2-3** — sample tags. New feature, self-contained module.
6. **P2-6** — analyser improvements. Most complex, highest payoff long-term.