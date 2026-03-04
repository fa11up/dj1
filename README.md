# 🎛️ dj1 Engine

An autonomous DJ system that ingests your local music library, analyzes every
track, seeds a session from a Spotify playlist, plans a setlist with harmonic
mixing and energy curve logic, then renders a continuous mix to a single audio
file — with real DJ-style transitions.

Phase 1 is complete and producing working mixes.

---

## System Dependencies

Install these before Python packages:

```bash
brew install ffmpeg       # required by pydub (MP3/AAC/M4A loading + export)
brew install rubberband   # required by pyrubberband (BPM time-stretching)
```

---

## Setup

```bash
# 1. Navigate to project
cd ~/dj1

# 2. Install Python dependencies (uv recommended)
uv sync
# or: pip install -r requirements.txt

# 3. Set up Spotify credentials (optional)
cp .env.example .env
# Edit .env with your Spotify Client ID + Secret
# Add http://127.0.0.1:8000/callback to your Spotify app's redirect URIs
```

---

## Running the Pipeline

Each module can be run independently or chained via `main.py`.

### Full pipeline
```bash
uv run python main.py --config config/session.yaml
```

### Individual modules
```bash
# Module 1 — scan your music library
uv run python -m modules.ingestor --music-dir /path/to/music

# Module 2 — analyze all tracks (BPM, key, energy, beat grid)
uv run python -m modules.analyzer

# Module 3 — match a Spotify playlist to your local library
uv run python -m modules.spotify_bridge --list                      # browse playlists
uv run python -m modules.spotify_bridge --playlist <ID_or_URL>     # run match

# Module 4 — plan the setlist
uv run python -m modules.planner

# Module 5 — render the mix
uv run python -m modules.renderer --preview    # fast, 128k, short fades
uv run python -m modules.renderer --quality    # full pipeline, 320k
```

### Resume from a specific module
```bash
uv run python main.py --config config/session.yaml --from-module 4
```

---

## Project Structure

```
dj1/
├── main.py                    # Pipeline orchestrator
├── config/
│   └── session.yaml           # Session config — edit before each run
├── modules/
│   ├── ingestor.py            # Module 1: Library scanner + metadata
│   ├── analyzer.py            # Module 2: BPM, key, energy, beat grid
│   ├── spotify_bridge.py      # Module 3: Spotify playlist → local match
│   ├── planner.py             # Module 4: Setlist + energy curve logic
│   └── renderer.py            # Module 5: Mix rendering + DJ transitions
├── tests/
│   ├── test_ingestor.py
│   ├── test_analyzer.py
│   ├── test_spotify_bridge.py
│   ├── test_planner.py
│   └── test_renderer.py
├── data/                      # Runtime JSON (gitignored)
│   ├── tracks.json            # Module 1 output
│   ├── analysis.json          # Module 2 output
│   ├── matched_tracks.json    # Module 3 output
│   └── session.json           # Module 4 output
├── output/                    # Rendered mixes (gitignored)
├── .env.example               # Spotify credential template
└── requirements.txt
```

---

## Configuration (`config/session.yaml`)

```yaml
session:
  duration_minutes: 60
  energy_curve: festival      # flat | ramp_up | peak_valley | festival

dj:
  chaos_factor: 0.3           # 0.0 = safe/predictable  1.0 = wild
  random_seed: null           # integer for reproducible run, null for fresh
  harmonic_strict: false      # true = Camelot wheel only, no exceptions
  tempo_tolerance_bpm: 8      # BPM window for transition scoring
  transition_style: mixed     # blends | cuts | mixed
```

**`chaos_factor`** is the main dial. It controls:
- Energy curve shape (low chaos → festival; high chaos → random)
- Track selection (low chaos → tightest harmonic/BPM fit; high chaos → adventurous)
- Transition style (low chaos → full staged DJ pipeline; high chaos → raw cuts)
- EQ filter shape (low chaos → hard/steep; high chaos → gentle shelf)

---

## How Transitions Work

The renderer implements a real DJ transition pipeline, not a simple crossfade.

### Staged blend (`long_blend`)
Four phases, lengths derived from BPM (32–64 bars at tempo):

```
Phase 1  Outgoing plays alone. Renderer waits for a breakdown/energy dip
         then delays to the next 8-bar phrase boundary ("the right moment").

Phase 2  Incoming enters with bass cut — crowd hears only highs shimmer on top.
         Outgoing still owns the low end. Runs 8 bars (fast BPM) or 16 bars (slow).

Phase 3  Bass swap. Outgoing bass cut, incoming bass brought in simultaneously.
         Tight (1–2 bars) at low chaos. Loose (3–4 bars) at high chaos.
         This is the handoff — only one bassline in the mix at a time.

Phase 4  Outgoing highs fade out over 16 bars. Incoming owns the full spectrum.
```

EQ crossover frequency is derived from each track's spectral centroid:
dark/bass-heavy tracks (Techno/DnB) get a lower crossover (~60–80Hz),
House tracks get a higher crossover (~120–150Hz).

Filter shape: cascaded (hard/steep) at low chaos, single shelf (gentle) at high chaos.

### Other transition types
- **`crossfade`** — overlapping volume fade with EQ bass-cut at low chaos
- **`quick_fade`** — short volume fade, no EQ
- **`hard_cut`** — sequential tail-fade (2s) → head-fade (2s), no overlap

### Energy override
Regardless of the planner's hint, if both track endpoints are soft (RMS below
threshold), the renderer upgrades to `long_blend`. If one side is soft and the
hint is `hard_cut`, it upgrades to `crossfade`. Two loud sections keep whatever
the planner assigned.

---

## Running Tests

```bash
uv run python tests/test_ingestor.py
uv run python tests/test_analyzer.py
uv run python tests/test_spotify_bridge.py
uv run python tests/test_planner.py
uv run python tests/test_renderer.py
```

---

## Phase 1 Status

| Module | Description | Status |
|--------|-------------|--------|
| 1. Ingestor | Library scan, metadata, tracks.json | ✅ Complete |
| 2. Analyzer | BPM, key, Camelot, energy curve, beat grid | ✅ Complete |
| 3. Spotify Bridge | Playlist fetch, fuzzy match to local library | ✅ Complete |
| 4. Session Planner | Setlist, energy curves, harmonic mixing, chaos | ✅ Complete |
| 5. Renderer | Staged transitions, EQ swap, BPM stretch, mix export | ✅ Complete |

Phase 2 planning document: `PHASE2.md`

---

## Known Limitations (Phase 1)

- BPM detection occasionally reports half-time or double-time values (e.g. 156 BPM
  that should be 78). The renderer trusts the analyzer; correction is a Phase 2 item.
- Spotify `/audio-features` endpoint is restricted by Spotify since late 2024.
  Module 3 uses local analysis only — Spotify provides track names for matching only.
- Loop-based transitions (extend a 4-bar loop before the incoming track enters)
  are stubbed for Phase 2.
- Sample/soundbite injection is a Phase 2 feature.