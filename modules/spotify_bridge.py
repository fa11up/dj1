"""
Module 3: Spotify Bridge
Fetches a Spotify playlist and fuzzy-matches tracks against the local library.
Outputs matched_tracks.json — local filepaths for the Session Planner to seed from.
"""

import os, sys, json, time, re, argparse
from pathlib import Path
from datetime import datetime

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from thefuzz import fuzz

DEFAULT_THRESHOLD = 80
SPOTIFY_SCOPES    = "playlist-read-private playlist-read-collaborative"


def get_spotify_client(cache_path="./.spotify_token_cache"):
    load_dotenv()
    cid  = os.getenv("SPOTIFY_CLIENT_ID")
    sec  = os.getenv("SPOTIFY_CLIENT_SECRET")
    ruri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
    if not cid or not sec:
        sys.exit("✗ Spotify credentials missing — copy .env.example to .env")
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=cid, client_secret=sec, redirect_uri=ruri,
        scope=SPOTIFY_SCOPES, cache_path=cache_path, open_browser=True,
    ))


def extract_playlist_id(s):
    s = s.strip()
    if s.startswith("https://"):
        return s.split("/playlist/")[-1].split("?")[0].strip()
    if s.startswith("spotify:playlist:"):
        return s.split(":")[-1].strip()
    return s


def fetch_playlist_tracks(sp, playlist_id):
    pl    = sp.playlist(playlist_id, fields="name,owner,tracks.total")
    name  = pl["name"]
    print(f"\n  {pl['name']}  (by {pl['owner']['display_name']}, {pl['tracks']['total']} tracks)\n")
    tracks, offset = [], 0
    while True:
        page = sp.playlist_tracks(playlist_id, offset=offset, limit=100,
                                   fields="items(track(name,artists)),next")
        for item in page.get("items", []):
            t = item.get("track")
            if t and t.get("name"):
                tracks.append({"name": t["name"],
                                "artist": t["artists"][0]["name"] if t["artists"] else "Unknown"})
        offset += 100
        if not page.get("next"):
            break
        time.sleep(0.1)
    return name, tracks


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r'\(feat\.?.*?\)', '', text)
    text = re.sub(r'feat\.?\s+\w+', '', text)
    text = re.sub(r'\((radio edit|original mix|remaster(ed)?)\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def match_score(sp_name, sp_artist, local):
    title_score  = fuzz.token_sort_ratio(normalize(sp_name), normalize(local.get("title") or local.get("filename", "")))
    local_artist = normalize(local.get("artist") or "")
    if not local_artist:
        return title_score
    return int(title_score * 0.6 + fuzz.token_sort_ratio(normalize(sp_artist), local_artist) * 0.4)


def match_to_library(spotify_tracks, local_tracks, threshold=DEFAULT_THRESHOLD):
    matched, unmatched = [], []
    for i, sp in enumerate(spotify_tracks):
        print(f"\r  Matching {i+1}/{len(spotify_tracks)}  {sp['name'][:40]}".ljust(70), end="", flush=True)
        best_score, best_local = 0, None
        for local in local_tracks:
            s = match_score(sp["name"], sp["artist"], local)
            if s > best_score:
                best_score, best_local = s, local
        if best_score >= threshold and best_local:
            matched.append({"name": sp["name"], "artist": sp["artist"],
                             "local_track_id": best_local["track_id"],
                             "local_filepath": best_local["filepath"],
                             "score": best_score})
        else:
            unmatched.append({"name": sp["name"], "artist": sp["artist"], "best_score": best_score})
    print("\r" + " " * 70 + "\r", end="")
    return matched, unmatched


def list_playlists(sp):
    results = sp.current_user_playlists(limit=50)
    playlists = results["items"]
    while results.get("next"):
        results = sp.next(results)
        playlists.extend(results["items"])
    print(f"\n  {'Name':<40} {'Tracks':>6}  ID")
    print("  " + "─" * 60)
    for pl in playlists:
        print(f"  {pl['name'][:38]:<40} {pl['tracks']['total']:>6}  {pl['id']}")
    print()


def run(playlist_id, tracks_path="./data/tracks.json",
        output_path="./data/matched_tracks.json", threshold=DEFAULT_THRESHOLD,
        token_cache_path="./.spotify_token_cache"):

    print("─" * 60 + "\n  AutoDJ — Module 3: Spotify Bridge\n" + "─" * 60)

    sp = get_spotify_client(cache_path=token_cache_path)
    print("  ✓ Authenticated\n")

    if not Path(tracks_path).exists():
        sys.exit(f"✗ {tracks_path} not found — run Module 1 first.")

    local_tracks = json.loads(Path(tracks_path).read_text())["tracks"]
    print(f"  Local library: {len(local_tracks)} tracks\n")

    pid = extract_playlist_id(playlist_id)
    playlist_name, spotify_tracks = fetch_playlist_tracks(sp, pid)
    print(f"  ✓ Fetched {len(spotify_tracks)} tracks\n")

    matched, unmatched = match_to_library(spotify_tracks, local_tracks, threshold)
    rate = round(len(matched) / max(len(spotify_tracks), 1) * 100, 1)
    print(f"  ✓ Matched {len(matched)}/{len(spotify_tracks)} ({rate}%)"
          f"  —  {len(unmatched)} not in local library\n")

    for m in matched[:10]:
        print(f"  {'✓':2} {m['name'][:35]:<35}  {m['artist'][:20]:<20}  {m['score']}")
    if len(matched) > 10:
        print(f"  ... and {len(matched) - 10} more")

    output = {
        "generated_at": datetime.now().isoformat(),
        "playlist_name": playlist_name, "playlist_id": pid,
        "threshold_used": threshold,
        "match_stats": {"total": len(spotify_tracks), "matched": len(matched),
                        "unmatched": len(unmatched), "match_rate_pct": rate},
        "matched_tracks": matched,
        "unmatched_tracks": unmatched,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  ✅ matched_tracks.json → {Path(output_path).resolve()}\n" + "─" * 60)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDJ Module 3 — Spotify Bridge")
    parser.add_argument("--playlist",  "-p")
    parser.add_argument("--list",      "-l", action="store_true")
    parser.add_argument("--tracks",    "-t", default="./data/tracks.json")
    parser.add_argument("--output",    "-o", default="./data/matched_tracks.json")
    parser.add_argument("--threshold", "-m", type=int, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if args.list:
        list_playlists(get_spotify_client())
        sys.exit(0)
    if not args.playlist:
        parser.error("--playlist is required (or use --list)")

    run(playlist_id=args.playlist, tracks_path=args.tracks,
        output_path=args.output, threshold=args.threshold)