# 🎛️ dj1 Engine

An autonomous DJ system that ingests your local music library, analyzes every
track, seeds a session from a Spotify playlist, plans a setlist with harmonic
mixing and energy curve logic, then renders a continuous mix to a single audio
file — with real DJ-style transitions.

---

## System Dependencies

Install these before Python packages:

```bash
brew install ffmpeg       # required by pydub (MP3/AAC/M4A loading + export)
brew install rubberband   # required by pyrubberband (BPM time-stretching)
```

For stem-based transitions (`stem_blend`), demucs is also required:

```bash
pip install demucs        # needs Python 3.12 or earlier (no 3.14 wheels yet)
# or: brew install demucs
```

---

## Setup

```bash
# 1. Navigate to project
cd ~/dj1

# 2. Install Python dependencies (uv recommended)
uv sync
# or: pip install -r requirements.txt

# 3. Set up credentials
cp .env.example .env
# Edit .env — add Spotify and/or AI brain keys (see sections below)
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
│   ├── brain.py               # Module 4b: AI setlist planner (optional)
│   ├── renderer.py            # Module 5: Mix rendering + DJ transitions
│   ├── stems.py               # Module 5b: Stem separation via demucs (optional)
│   └── creative.py            # Creative mode: full-session stem composition
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
│   ├── session.json           # Module 4 output
│   └── .stems_cache/          # Demucs stem cache (per track_id)
├── output/                    # Rendered mixes (gitignored)
├── .env.example               # Credential template
└── pyproject.toml
```

---

## Configuration (`config/session.yaml`)

```yaml
paths:
  music_dir: "~/Music/library"  # your local music folder
  data_dir: "./data"
  output_dir: "./output"

spotify:
  enabled: true
  playlist_ids: ["<playlist_id_or_url>"]
  match_threshold: 55           # 0–100, lower = more permissive matching

session:
  mode: sequential              # sequential | creative
  duration_minutes: 60
  energy_curve: festival        # flat | ramp_up | peak_valley | festival
  mood_prompt: ""               # e.g. "late night, melancholic, driving"

dj:
  chaos_factor: 0.3             # 0.0 = safe/predictable  1.0 = wild
  random_seed: null             # integer for reproducible run, null for fresh
  harmonic_strict: false        # true = Camelot wheel only, no exceptions
  bpm_normalize: true           # snap double/half-time BPM outliers
  tempo_tolerance_bpm: 8        # BPM window for transition scoring
  transition_style: mixed       # blends | cuts | mixed

brain:
  enabled: false                # true = AI plans the setlist
  provider: anthropic           # anthropic | openai_compatible
  model: ""                     # model name override (see below)
  base_url: ""                  # openai_compatible only

output:
  format: mp3                   # wav | mp3
  bitrate: "320k"
```

**`chaos_factor`** is the main dial. It controls:
- Energy curve shape (low chaos → festival; high chaos → random)
- Track selection (low chaos → tightest harmonic/BPM fit; high chaos → adventurous)
- Transition style (low chaos → full staged DJ pipeline; high chaos → raw cuts)
- EQ filter shape (low chaos → hard/steep; high chaos → gentle shelf)

**`session.mode`**
- `sequential` — standard planner: ordered setlist, one track at a time
- `creative` — stem composition mode: four channels (drums / bass / melody / vocals) sourced independently from different tracks and mixed bar-by-bar. Requires the brain and demucs.

---

## DJ Brain

The brain replaces the heuristic setlist planner with a single AI API call (tool
use / function calling). It reads the full track library metadata and
`mood_prompt`, then returns a harmonically and energetically considered setlist
as structured JSON.

The planner falls back to the algorithmic path silently if the brain is
unavailable, misconfigured, or returns a bad response.

### Option A — Claude via Anthropic API

```yaml
brain:
  enabled: true
  provider: anthropic
  model: ""          # blank = claude-haiku-4-5 (fast + cheap). Try claude-sonnet-4-6 for quality.
```

`.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Option B — Local model via Ollama

Start Ollama with a model that supports tool use (Qwen3, Mistral, Llama 3.1+):

```bash
ollama pull qwen3:latest
ollama serve         # starts on http://localhost:11434 by default
```

`session.yaml`:
```yaml
brain:
  enabled: true
  provider: openai_compatible
  model: qwen3:latest
  base_url: http://localhost:11434/v1
```

`.env`:
```
BRAIN_API_KEY=ollama
```

### Option C — Local model via llama.cpp server

```bash
llama-server -m /path/to/qwen3-8b.gguf --port 8080
```

`session.yaml`:
```yaml
brain:
  enabled: true
  provider: openai_compatible
  model: qwen3-8b       # matches whatever name the server reports
  base_url: http://localhost:8080/v1
```

`.env`:
```
BRAIN_API_KEY=none
```

> The `openai` Python package is required for `openai_compatible` providers:
> `pip install openai`

### Model recommendations

| Goal | Model |
|------|-------|
| Fast, cheap cloud | `claude-haiku-4-5` (default) |
| Best quality cloud | `claude-sonnet-4-6` |
| Local, tool-use capable | `qwen3:latest`, `llama3.1:8b`, `mistral:7b-instruct` |

Smaller local models work but may occasionally hallucinate track IDs — the
re-hydrator discards any unknown IDs and the planner falls back if fewer than
2 valid tracks remain.

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

### Stem blend (`stem_blend`)

When demucs is installed, `stem_blend` separates both tracks into
drums / bass / melody / vocals and mixes channels independently.
Each stem is time-stretched to the master BPM before mixing.
Used for 2–4 signature moments per set. Requires BPM delta < 6% and harmonic
compatibility (Camelot rules 1–3).

Stems are cached under `data/.stems_cache/` so separation only runs once per track.

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

## Status

| Module | Description | Status |
|--------|-------------|--------|
| 1. Ingestor | Library scan, metadata, tracks.json | ✅ Complete |
| 2. Analyzer | BPM, key, Camelot, energy curve, beat grid | ✅ Complete |
| 3. Spotify Bridge | Playlist fetch, fuzzy match to local library | ✅ Complete |
| 4. Session Planner | Setlist, energy curves, harmonic mixing, chaos | ✅ Complete |
| 4b. Brain | AI setlist planner — Anthropic + local model support | ✅ Complete |
| 5. Renderer | Staged transitions, EQ swap, BPM stretch, mix export | ✅ Complete |
| 5b. Stems | Demucs stem separation + caching for stem_blend | ✅ Complete |
| Creative Mode | Full-session stem composition across 4 channels | 🚧 In progress |

---

## Known Limitations

- BPM detection occasionally reports half-time or double-time values (e.g. 156 BPM
  that should be 78). The renderer trusts the analyzer; correction is a future item.
- Spotify `/audio-features` endpoint is restricted by Spotify since late 2024.
  Module 3 uses local analysis only — Spotify provides track names for matching only.
- demucs has no Python 3.14 wheels as of early 2026. Use Python 3.12 for stem blends,
  or install demucs via homebrew/conda and ensure it's in PATH.
- Loop-based transitions (extend a 4-bar loop before the incoming track enters)
  are planned for a future phase.
