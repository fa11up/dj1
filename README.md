# 🎛️ DJ1 Engine

An autonomous DJ system that takes your local music library and mixes it into a unique session every time — seeded by your Spotify playlists.

---

## Prerequisites

Before installing Python dependencies, you need:

### ffmpeg (required by pydub + pyrubberband)
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

### Python 3.11+
```bash
python --version  # confirm 3.11 or higher
```

---

## Setup

```bash
# 1. Clone / navigate to project
cd ~/projects/autodj

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Spotify credentials (optional for Phase 1 Spotify features)
cp .env.example .env
# Edit .env with your Spotify app credentials
```

---

## Project Structure

```
autodj/
├── main.py                  # Pipeline orchestrator
├── config/
│   └── session.yaml         # Your session config
├── modules/
│   ├── ingestor.py          # Module 1: Library scanner
│   ├── analyzer.py          # Module 2: Audio feature extraction
│   ├── spotify_bridge.py    # Module 3: Spotify API + matching
│   ├── planner.py           # Module 4: Setlist generation
│   └── renderer.py          # Module 5: Mix rendering
├── utils/
│   ├── audio_utils.py       # Shared audio helpers
│   ├── camelot.py           # Camelot Wheel key compatibility
│   └── cache.py             # Analysis cache manager
├── data/                    # Generated at runtime (gitignored)
├── output/                  # Rendered mixes (gitignored)
├── tests/                   # Module tests
├── music_sample/            # Drop a few test tracks here
├── .env.example             # Spotify credential template
└── requirements.txt
```

---

## Running the Pipeline

```bash
# Full pipeline (once all modules are built)
uv run python main.py --config config/session.yaml

# Run Module 1 only (library scan)
uv run python -m modules.ingestor --music-dir {path to music library}

# Run tests
uv run python -m pytest tests/ -v
```

---

## Phase 1 Modules Status

| Module | Status |
|--------|--------|
| 1. Ingestor | ✅ Built |
| 2. Analyzer | ✅ Built |
| 3. Spotify Bridge | ✅ Built |
| 4. Session Planner | 🔜 Next |
| 5. Mix Renderer | 🔜 Planned |
