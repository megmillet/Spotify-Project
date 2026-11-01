"""
Add artist names (from MusicBrainz) and artist scores (from artists.csv)
to a song CSV.

Install:
    pip install pandas requests

Run (from the folder containing song_data.csv and artists.csv):
    python add_artists_musicbrainz.py

Useful options:
    --limit 50              only process the first 50 distinct songs (test run)
    --drop-duplicates       drop exact duplicate rows from the output
    --overrides fixes.csv   manual fixes (columns: song_name, artist
                            [, song_duration_ms]); separate several artists
                            with ";"
    --max-pages 2           fetch up to 2 pages (100 hits each) per title

Matching priorities for each song (title + duration):
  1. Recording title matches (punctuation/accent/case-insensitive; the title
     is also tried with "(feat. ...)" and "- Remastered/From ..." removed).
  2. Smallest duration difference (within DURATION_TOLERANCE_MS).
  3. Earliest documented first-release date among equally close matches.
  4. Highest MusicBrainz relevance score.
  5. If different artists are still tied, flag for manual review.

Outputs:
  song_data_with_artists.csv       full dataset + artist + artist stats
  songs_to_review.csv              rows that were not matched
  musicbrainz_cache.json           search cache (safe to delete; resumable)

MusicBrainz public API limit: max one request per second. The first run
needs one request per distinct cleaned title, so expect it to take hours on
a large file; the cache means re-runs are fast and interrupted runs resume.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

# >>> PUT YOUR EMAIL HERE (MusicBrainz asks for contact info) <<<
CONTACT_EMAIL = "kiwibird0316@gmail.com"

API_URL = "https://musicbrainz.org/ws/2/recording/"
MIN_INTERVAL_SECONDS = 1.2   # wait between requests: random value
MAX_INTERVAL_SECONDS = 1.5   # between these two numbers
DURATION_TOLERANCE_MS = 5000
RESULT_LIMIT = 100
SAVE_EVERY_NEW_SEARCHES = 20
CACHE_VERSION = 2
NO_DATE = "9999-99-99"
IGNORED_ARTISTS = {"variousartists", "unknown", "noartist", "specialpurposeartist"}

# Artist names that almost always mean a cover/karaoke/kids/tribute version.
# Applies even to artists listed in artists.csv (it contains Kidz Bop Kids).
COVER_ARTIST_PATTERN = re.compile(
    r"kidz\s*bop|karaoke|tribute|\bcovers?\b|backing\s*track|in the style of|"
    r"originally performed|made famous|string quartet|vitamin string|lullaby|"
    r"rockabye|party tyme|hit crew|ameritz|sunfly|stingray|music box|8-?bit|"
    r"studio (?:musicians|group|band|singers)|piano (?:tribute|covers?)|"
    r"instrumental|alvin and the chipmunks|the chipmunks|nightcore|sped up",
    re.I,
)

# Columns in artists.csv -> names used in the output
ARTIST_STAT_COLUMNS = {
    "Streams": "artist_streams",
    "Daily": "artist_daily",
    "As lead": "artist_as_lead",
    "Solo": "artist_solo",
    "As feature": "artist_as_feature",
}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def normalize(value):
    """Case/accent/punctuation-insensitive key. Keeps non-Latin letters
    (the original a-z0-9 version turned every Chinese/Korean title into '')."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[\W_]+", "", value.casefold())


_FEAT_PAREN = re.compile(
    r"\s*[\(\[]\s*(?:feat\.?|ft\.?|featuring|with)\s[^\)\]]*[\)\]]", re.I
)
_VERSION_SUFFIX = re.compile(
    r"\s+-\s+(?:from\b|\d{4}\b.*remaster|remaster|mono\b|stereo\b|"
    r"single version|album version|radio edit|bonus track).*$",
    re.I,
)
_REMASTER_PAREN = re.compile(r"\s*[\(\[][^\)\]]*remaster[^\)\]]*[\)\]]", re.I)


def clean_title(title):
    """Remove credit/version noise that MusicBrainz titles normally omit."""
    cleaned = _FEAT_PAREN.sub("", str(title))
    cleaned = _VERSION_SUFFIX.sub("", cleaned)
    cleaned = _REMASTER_PAREN.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned or str(title).strip()


def lucene_escape(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def release_date_key(value):
    """Earlier dates sort first; missing dates sort last."""
    if not value:
        return NO_DATE
    value = str(value)
    if re.fullmatch(r"\d{4}", value):
        return value + "-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value + "-01"
    return value


# --------------------------------------------------------------------------
# MusicBrainz client
# --------------------------------------------------------------------------
def slim_recording(rec):
    """Keep only the fields we use so the cache stays small."""
    names, credit = [], ""
    for item in rec.get("artist-credit") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("artist", {}).get("name", "")
        if name:
            names.append(name)
        credit += name + item.get("joinphrase", "")
    return {
        "id": rec.get("id"),
        "title": rec.get("title", ""),
        "length": rec.get("length"),
        "date": rec.get("first-release-date"),
        "score": rec.get("score", 0),
        "artists": names,
        "credit": credit.strip(),
    }


class MusicBrainzClient:
    def __init__(self, user_agent, max_pages=1):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept": "application/json"}
        )
        self.max_pages = max_pages
        self.last_request_time = 0.0

    def _get(self, params, label):
        for attempt in range(6):
            wait = random.uniform(MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS) - (
                time.monotonic() - self.last_request_time
            )
            if wait > 0:
                time.sleep(wait)
            self.last_request_time = time.monotonic()
            try:
                resp = self.session.get(API_URL, params=params, timeout=30)
                if resp.status_code in (429, 503):
                    try:
                        delay = float(resp.headers.get("Retry-After", ""))
                    except ValueError:
                        delay = 5 * (attempt + 1)
                    print(f"MusicBrainz returned {resp.status_code}; waiting {delay:.0f}s.")
                    time.sleep(delay)
                    continue
                if resp.status_code == 400:
                    print(f"MusicBrainz rejected the query for {label!r} (HTTP 400); skipping.")
                    return None
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as error:
                if attempt == 5:
                    print(f"Request failed for {label!r}: {error}")
                    return None
                time.sleep(3 * (attempt + 1))
        return None

    @staticmethod
    def _query(title, duration_ms=None):
        query = f'recording:"{lucene_escape(title)}"'
        if duration_ms is not None:
            lo = max(0, duration_ms - DURATION_TOLERANCE_MS)
            hi = duration_ms + DURATION_TOLERANCE_MS
            query += f" AND dur:[{lo} TO {hi}]"
        return query

    def search_recordings(self, title, duration_ms=None):
        """Return a list of slimmed recordings, or None on failure
        (failures are NOT cached, so a re-run retries them)."""
        results = []
        for page in range(self.max_pages):
            data = self._get(
                {
                    "query": self._query(title, duration_ms),
                    "fmt": "json",
                    "limit": RESULT_LIMIT,
                    "offset": page * RESULT_LIMIT,
                },
                title,
            )
            if data is None:
                return None if page == 0 else results
            batch = data.get("recordings", [])
            results.extend(slim_recording(r) for r in batch)
            if len(batch) < RESULT_LIMIT or len(results) >= data.get("count", 0):
                break
        return results


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def identify_artist(recordings, title, duration_ms, known_keys=frozenset()):
    """Return (credit_string, [artist names], status)."""
    if recordings is None:
        return None, [], "API error (will retry on next run)"
    if not recordings:
        return None, [], "Not found"

    wanted = {normalize(title), normalize(clean_title(title))} - {""}
    candidates = []
    for rec in recordings:
        if normalize(rec["title"]) not in wanted or rec["length"] is None:
            continue
        try:
            diff = abs(int(rec["length"]) - duration_ms)
        except (TypeError, ValueError):
            continue
        if diff > DURATION_TOLERANCE_MS or not rec["artists"]:
            continue
        if all(normalize(a) in IGNORED_ARTISTS for a in rec["artists"]):
            continue
        is_known = any(normalize(a) in known_keys for a in rec["artists"])
        if any(COVER_ARTIST_PATTERN.search(a) for a in rec["artists"]):
            continue  # cover / karaoke / kids version (even if listed in artists.csv)
        candidates.append(
            {
                "known": is_known,
                "credit": rec["credit"],
                "artists": rec["artists"],
                "diff": diff,
                "date": release_date_key(rec["date"]),
                "score": int(rec["score"] or 0),
            }
        )

    if not candidates:
        return None, [], "No title/duration match"

    # Prefer recordings by artists that appear in artists.csv.
    if known_keys and any(c["known"] for c in candidates):
        candidates = [c for c in candidates if c["known"]]

    best_diff = min(c["diff"] for c in candidates)
    candidates = [c for c in candidates if c["diff"] == best_diff]

    known = [c["date"] for c in candidates if c["date"] != NO_DATE]
    if known:
        earliest = min(known)
        candidates = [c for c in candidates if c["date"] == earliest]

    top = max(c["score"] for c in candidates)
    candidates = [c for c in candidates if c["score"] == top]

    credits = {c["credit"] for c in candidates}
    if len(credits) != 1:
        return None, [], "Ambiguous after duration/date/score ranking"

    chosen = candidates[0]
    status = "Matched - exact duration" if best_diff == 0 else "Matched - closest duration"
    if chosen["known"]:
        status += " - known artist"
    return chosen["credit"], chosen["artists"], status


# --------------------------------------------------------------------------
# artists.csv scores
# --------------------------------------------------------------------------
def load_artist_scores(path):
    art = pd.read_csv(path)
    if "Artist" not in art.columns:
        raise ValueError("artists.csv needs an 'Artist' column")
    present = {k: v for k, v in ARTIST_STAT_COLUMNS.items() if k in art.columns}
    for col in present:
        art[col] = pd.to_numeric(
            art[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    art["_key"] = art["Artist"].map(normalize)
    # Keep the highest-streams row if an artist (or spelling variant) repeats.
    sort_col = "Streams" if "Streams" in present else art.columns[1]
    art = art.sort_values(sort_col, ascending=False).drop_duplicates("_key")
    lookup = {
        row["_key"]: {new: row[old] for old, new in present.items()} | {"_name": row["Artist"]}
        for _, row in art.iterrows()
    }
    return lookup, list(present.values())


def attach_scores(names, lookup):
    """Prefer the first-credited artist; fall back to other credited artists."""
    for position, name in enumerate(names):
        hit = lookup.get(normalize(name))
        if hit:
            return hit, ("primary artist" if position == 0 else "featured/other artist")
    return None, "artist not in artists.csv" if names else "no artist"


# --------------------------------------------------------------------------
# Cache / overrides
# --------------------------------------------------------------------------
def load_cache(path):
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                return data["searches"]
            print("Cache is from an older script version; starting a fresh cache.")
        except (json.JSONDecodeError, OSError, KeyError, AttributeError):
            print("Warning: cache unreadable; starting with an empty cache.")
    return {}


def save_cache(path, searches):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"version": CACHE_VERSION, "searches": searches}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_overrides(path):
    if not path:
        return {}, {}
    ov = pd.read_csv(path)
    if not {"song_name", "artist"} <= set(ov.columns):
        raise ValueError("overrides file needs columns: song_name, artist")
    by_title, by_title_dur = {}, {}
    for _, row in ov.dropna(subset=["song_name", "artist"]).iterrows():
        names = [a.strip() for a in str(row["artist"]).split(";") if a.strip()]
        if "song_duration_ms" in ov.columns and pd.notna(row["song_duration_ms"]):
            by_title_dur[(normalize(row["song_name"]), int(row["song_duration_ms"]))] = names
        else:
            by_title[normalize(row["song_name"])] = names
    return by_title, by_title_dur


def load_seed(path):
    """Restore trustworthy results from an earlier output CSV.

    Keeps (a) matches whose artist is in artists.csv and is not a cover-style
    artist, and (b) genuine "Not found" misses. Everything else is searched
    again, because older runs may have picked covers.
    Returns {(song_name, duration_ms): (credit, [artist names], status)}.
    """
    old = pd.read_csv(path)
    needed = {"song_name", "song_duration_ms", "artist", "artist_primary", "match_status"}
    if not needed <= set(old.columns):
        raise ValueError(f"Seed file needs columns: {sorted(needed)}")
    stats_for = old["artist_stats_for"] if "artist_stats_for" in old.columns else None
    basis = old["artist_stats_basis"] if "artist_stats_basis" in old.columns else None
    seed = {}
    for i, row in old.iterrows():
        if pd.isna(row["song_name"]) or pd.isna(row["song_duration_ms"]):
            continue
        key = (row["song_name"], int(row["song_duration_ms"]))
        status = str(row["match_status"])
        if status == "Not found":
            seed[key] = (None, [], "Not found")
        elif (status.startswith("Matched") and basis is not None
              and basis[i] in ("primary artist", "featured/other artist")
              and not COVER_ARTIST_PATTERN.search(str(row["artist"]))):
            names = [row["artist_primary"]]
            if stats_for is not None and pd.notna(stats_for[i]) and stats_for[i] not in names:
                names.append(stats_for[i])
            seed[key] = (row["artist"], names, status.split(" - known artist")[0] + " - restored")
    return seed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--songs", default="song_data.csv")
    p.add_argument("--artists", default="artists.csv")
    p.add_argument("--output", default="song_data_with_artists.csv")
    p.add_argument("--review", default="songs_to_review.csv")
    p.add_argument("--cache", default="musicbrainz_cache.json")
    p.add_argument("--overrides", default=None)
    p.add_argument("--email", default=None,
                   help="contact email (overrides CONTACT_EMAIL in the script)")
    p.add_argument("--max-searches", type=int, default=None,
                   help="stop cleanly after N new MusicBrainz searches "
                        "(run repeatedly in short batches)")
    p.add_argument("--seed", default=None,
                   help="earlier output CSV; restores its trustworthy matches so "
                        "they are not searched again")
    p.add_argument("--refine", action="store_true",
                   help="after the main pass, re-search songs whose artist is not in "
                        "artists.csv using a duration-restricted query (extra requests)")
    p.add_argument("--no-prefer-known", action="store_true",
                   help="do not prefer artists listed in artists.csv")
    p.add_argument("--max-pages", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N distinct songs (testing)")
    p.add_argument("--drop-duplicates", action="store_true")
    return p.parse_args()


def run(args, client=None):
    songs_path, artists_path = Path(args.songs), Path(args.artists)
    for path in (songs_path, artists_path):
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}")

    df = pd.read_csv(songs_path)
    missing = {"song_name", "song_duration_ms"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped {before - len(df):,} exact duplicate rows.")

    lookup, stat_cols = load_artist_scores(artists_path)
    cache_path = Path(args.cache)
    known_keys = frozenset() if args.no_prefer_known else frozenset(lookup)
    cache = load_cache(cache_path)
    ov_title, ov_title_dur = load_overrides(args.overrides)
    seed = load_seed(args.seed) if args.seed else {}
    if seed:
        print(f"Loaded {len(seed):,} restorable results from {args.seed}.")

    if client is None:
        email = args.email or CONTACT_EMAIL
        if "@" not in email or email == "your-email@example.com":
            raise ValueError(
                "Set CONTACT_EMAIL near the top of the script (or pass --email)."
            )
        client = MusicBrainzClient(f"SongArtistLookup/3.0 ({email})", args.max_pages)

    # One resolution per distinct (title, duration); duplicates share the result.
    pairs = df[["song_name", "song_duration_ms"]].drop_duplicates()
    if args.limit:
        pairs = pairs.head(args.limit)
    total = len(pairs)
    print(f"{len(df):,} rows, {total:,} distinct title+duration pairs.")

    resolved, new_searches = {}, 0
    main_pass_complete = False
    try:
        for n, (title, duration) in enumerate(pairs.itertuples(index=False, name=None), 1):
            pair_key = (title, duration)
            if pd.isna(title) or not str(title).strip():
                resolved[pair_key] = (None, [], "Missing title")
                continue
            if pd.isna(duration):
                resolved[pair_key] = (None, [], "Missing duration")
                continue
            duration = int(duration)
            tkey = normalize(title)

            # Manual overrides win over the API.
            names = ov_title_dur.get((tkey, duration)) or ov_title.get(tkey)
            if names:
                resolved[pair_key] = (" & ".join(names), names, "Manual override")
                continue

            query_title = clean_title(title)
            skey = normalize(query_title)
            if not skey:
                resolved[pair_key] = (None, [], "Title has no searchable characters")
                continue
            # Fresh cached search results win; otherwise reuse restored results.
            if skey not in cache and (title, duration) in seed:
                resolved[pair_key] = seed[(title, duration)]
                continue
            if skey not in cache:
                if args.max_searches and new_searches >= args.max_searches:
                    print(f"\nBatch finished: {new_searches:,} searches done. "
                          "Run the script again to continue.")
                    break
                if new_searches == 0:
                    print("Searching MusicBrainz (about 1 song per 1.2-1.5 seconds). "
                          "Press Ctrl+C to stop; progress is saved.")
                recs = client.search_recordings(query_title)
                new_searches += 1
                if new_searches % 10 == 0:
                    print(f"  ...{n:,}/{total:,} songs checked "
                          f"({new_searches:,} searches this run)")
                if recs is not None:  # never cache failures
                    cache[skey] = recs
                if new_searches % SAVE_EVERY_NEW_SEARCHES == 0:
                    save_cache(cache_path, cache)

            resolved[pair_key] = identify_artist(
                cache.get(skey), title, duration, known_keys
            )

            if n % 100 == 0 or n == total:
                print(f"Processed {n:,}/{total:,} (new searches this run: {new_searches:,})")
        else:
            main_pass_complete = True

        # Optional second pass: duration-restricted searches for songs whose
        # artist is still not in artists.csv (likely covers / missed originals).
        if args.refine and main_pass_complete:
            todo = []
            for pair_key, (credit, names, status) in resolved.items():
                if status in ("Manual override", "Missing title", "Missing duration",
                              "Not found") or status.startswith("Title has no"):
                    continue
                hit, _ = attach_scores(names, lookup)
                if hit is None:
                    todo.append(pair_key)
            print(f"\nRefine pass: {len(todo):,} songs have no artists.csv artist.")
            for i, (title, duration) in enumerate(todo, 1):
                query_title = clean_title(title)
                rkey = f"{normalize(query_title)}@{int(duration)}"
                if rkey not in cache:
                    if args.max_searches and new_searches >= args.max_searches:
                        print(f"\nBatch finished: {new_searches:,} searches done. "
                              "Run the script again to continue refining.")
                        break
                    recs = client.search_recordings(query_title, int(duration))
                    new_searches += 1
                    if recs is not None:
                        cache[rkey] = recs
                    if new_searches % SAVE_EVERY_NEW_SEARCHES == 0:
                        save_cache(cache_path, cache)
                    if new_searches % 10 == 0:
                        print(f"  ...refined {i:,}/{len(todo):,}")
                if rkey in cache:
                    combined = (cache.get(normalize(query_title)) or []) + cache[rkey]
                    resolved[(title, duration)] = identify_artist(
                        combined, title, int(duration), known_keys
                    )
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C (or your terminal/editor) - "
              "saving progress; re-run to resume.")
    except Exception as error:
        print(f"\nStopped by an error: {type(error).__name__}: {error}\n"
              "Saving progress; re-run to resume.")
    finally:
        save_cache(cache_path, cache)

    # Build output columns.
    out_artist, out_primary, out_status, out_basis = [], [], [], []
    stat_values = {c: [] for c in stat_cols}
    out_stats_for = []
    for title, duration in df[["song_name", "song_duration_ms"]].itertuples(index=False, name=None):
        credit, names, status = resolved.get(
            (title, duration), (None, [], "Not processed yet")
        )
        hit, basis = attach_scores(names, lookup)
        out_artist.append(credit)
        out_primary.append(names[0] if names else None)
        out_status.append(status)
        out_basis.append(basis if names else None)
        out_stats_for.append(hit["_name"] if hit else None)
        for c in stat_cols:
            stat_values[c].append(hit.get(c) if hit else None)

    df["artist"] = out_artist
    df["artist_primary"] = out_primary
    df["match_status"] = out_status
    df["artist_stats_for"] = out_stats_for
    df["artist_stats_basis"] = out_basis
    for c in stat_cols:
        df[c] = stat_values[c]

    df.to_csv(args.output, index=False, encoding="utf-8")
    status = df["match_status"].fillna("")
    unmatched = ~status.str.startswith(("Matched", "Manual", "Not processed"))
    df[unmatched].to_csv(args.review, index=False, encoding="utf-8")
    not_done = status.str.startswith("Not processed").sum()
    if not_done:
        print(f"\nNOTE: {not_done:,} rows were not processed yet (run stopped early "
              f"or --limit was used). Run the script again without --limit to continue; "
              f"finished searches are cached.")

    print("\nFinished. Match summary:")
    print(df["match_status"].value_counts(dropna=False).to_string())
    print("\nArtist-score summary:")
    print(df["artist_stats_basis"].value_counts(dropna=False).to_string())
    print(f"\nFull dataset: {args.output}\nManual review: {args.review}\nCache: {args.cache}")
    return df


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"Error: {err}")