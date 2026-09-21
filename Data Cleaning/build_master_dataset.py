#!/usr/bin/env python3
"""
build_master_dataset.py
=======================
Cleans the five Spotify CSV files and combines them into ONE master,
song-level dataset.

QUICK START
-----------
1.  pip install pandas            (pandas 2.x or newer; nothing else is needed)
2.  Put the five CSVs in a folder called `data` next to this script
    (or directly beside the script).
3.  python build_master_dataset.py

Optional flags:
    --input-dir  PATH    folder that contains the five CSVs
    --output-dir PATH    where results are written (default: `output` next to this script)

OUTPUTS (written to the output folder)
--------------------------------------
    master_spotify_songs.csv   one row per unique song (UTF-8 with BOM, so Excel shows accents)
    cleaning_report.txt        everything that was fixed, dropped or found -- read this once
    dropped_rows.csv           every duplicate row that was removed, for auditing
    recovered_text.csv         (only with --recover-text / a cache) every recovered or rejected text fix

RECOVERING DESTROYED TEXT (optional)
------------------------------------
    python build_master_dataset.py --recover-text --contact you@example.com
  Looks up each damaged 2024 row by ISRC on MusicBrainz (free, ~1 request/second) and restores
  the title / artist / album. Every lookup is checked against the damaged text (byte lengths
  must match), and results are saved in text_recovery_cache.json next to this script. Commit
  that file: teammates then get the recovered text automatically, with no internet needed.

HOW THE MASTER IS LAID OUT
--------------------------
* One row per unique song. The four song files are matched on a normalised
  title + overlapping artist name and outer-joined, so a song that appears in
  only one file still gets a row. `n_sources` says how many files each song was found in.
* The LEFTMOST columns are the harmonised, analysis-ready view (title,
  primary_artist, release_year, explicit, bpm, ...). When sources disagree, the
  value is taken from the first source listed in PRIORITY below.
* EVERY original column is also kept, prefixed with its source, so nothing is lost:
      sp2023_       spotify-2023.csv
      sp2024_       Most Streamed Spotify Songs 2024.csv
      alltime_      spotify_alltime_top100_songs.csv
      wrapped2025_  spotify_wrapped_2025_top50_songs.csv
      artist2025_   spotify_wrapped_2025_top50_artists.csv (joined on the song's primary artist)
* Blank cell = the source had no value. Nothing is imputed or guessed.
* Units: stream counts are raw counts (the "billions" columns are converted, so
  they are only precise to about +/-5 million); audio features are 0-1.

KNOWN LIMITATIONS (details in cleaning_report.txt)
--------------------------------------------------
* The 2023/2024 files contain text that was already destroyed before we got it
  (accents replaced by "ï¿½", non-Latin titles turned into "ýýý"). It cannot be
  recovered; we mark lost characters with "�" and set `text_damaged`.
* The sources disagree on some shared fields (BPM, audio features, release date,
  explicit). Both values are kept; the unified column follows PRIORITY.
* Artist attributes are joined on the FIRST-listed artist only.

ARTIST COLUMNS
--------------
    artists         full credit as written in the source that lists the most artists
    artists_list    the same artists, one clean name each, separated by "; "
    artist_count    number of individual artists on the song
    primary_artist  first-listed artist (used for the artist2025_ join)
Every source also keeps its own <source>_artists and <source>_artist_count.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG -- the things you might want to change
# =============================================================================

SOURCE_FILES = {
    "sp2023": "spotify-2023.csv",
    "sp2024": "Most Streamed Spotify Songs 2024.csv",
    "alltime": "spotify_alltime_top100_songs.csv",
    "wrapped2025": "spotify_wrapped_2025_top50_songs.csv",
    "artists2025": "spotify_wrapped_2025_top50_artists.csv",
}
SONG_SOURCES = ["sp2023", "sp2024", "alltime", "wrapped2025"]
ARTIST_PREFIX = "artist2025"

MASTER_FILENAME = "master_spotify_songs.csv"
REPORT_FILENAME = "cleaning_report.txt"
DROPPED_FILENAME = "dropped_rows.csv"
RECOVERED_FILENAME = "recovered_text.csv"
CACHE_FILENAME = "text_recovery_cache.json"  # lives next to this script; share it with your team
DEFAULT_CONTACT = "spotify-master-builder"   # MusicBrainz asks for a contact in the User-Agent (--contact)

# Which source wins for each harmonised column (first non-blank value is used).
# The 2023/2024 files carry Spotify-API style metadata, so they lead for factual
# fields; the two curated 2025 files lead for text because they are not damaged.
_TEXT_ORDER = ["alltime", "wrapped2025", "sp2023", "sp2024"]
_AUDIO_ORDER = ["sp2023", "alltime", "wrapped2025"]
PRIORITY: Dict[str, List[str]] = {
    "title": _TEXT_ORDER,
    "release_date": ["sp2023", "sp2024"],
    "release_year": ["sp2023", "sp2024", "alltime", "wrapped2025"],
    "explicit": ["sp2024", "alltime", "wrapped2025"],
    "primary_genre": ["alltime", "wrapped2025"],
    "artist_country": ["alltime", "wrapped2025"],
    "bpm": _AUDIO_ORDER,
    "danceability": _AUDIO_ORDER,
    "energy": _AUDIO_ORDER,
    "valence": _AUDIO_ORDER,
    "acousticness": _AUDIO_ORDER,
}
TEXT_FIELDS = {"title"}  # prefer undamaged text
# `artists` is chosen differently: the source that credits the MOST individual artists wins
# (ties -> undamaged text, then this order), so featured artists are never lost.
ARTIST_ORDER = _TEXT_ORDER
ARTIST_LIST_SEP = "; "  # separator used in the normalised `artists_list` column

# How to split an "artists" string into individual artists, per source.
#   comma  -> "A, B, C"                       (spotify-2023)
#   none   -> one artist per row, "&" is part of the name, e.g. "Jesse & Joy"  (spotify 2024)
#   collab -> "A & B", "A ft. B", "A feat. B"  (all-time / wrapped 2025)
ARTIST_SPLIT_STYLE = {"sp2023": "comma", "sp2024": "none", "alltime": "collab", "wrapped2025": "collab"}
# Names that contain the separator and must NOT be split. Add more here if needed.
PROTECTED_ARTISTS = ["Tyler, The Creator", "Earth, Wind & Fire"]

# When one source lists the same song twice, keep the row with the most data,
# breaking ties with the larger value in this column.
DEDUP_SIZE_COLUMN = {"sp2023": "streams", "sp2024": "spotify_streams"}

# Cross-source agreement checks written to the report: (field, tolerance)
CONFLICT_CHECKS = [
    ("bpm", 0), ("danceability", 0.011), ("energy", 0.011), ("valence", 0.011),
    ("acousticness", 0.011), ("release_year", 0), ("explicit", 0),
]

# Bookkeeping columns that are needed while building but not written to the master CSV.
# Delete a name from this list to get that column back.
OUTPUT_DROP_COLUMNS = ["in_sp2023", "in_sp2024", "in_alltime", "in_wrapped2025", "audio_features_source"]

LOST_CHAR = "\ufffd"  # "�" -- marks a character that was destroyed upstream

# Text-damage patterns (written as \u escapes so this file is safe in any editor).
_RE_DAMAGED_SEQ = r"(?:\u00ef\u00bf\u00bd)+"   # "ï¿½" repeated: a UTF-8 "�" that was re-saved as cp1252
_RE_DAMAGED_TAIL = r"\u00ef\u00bf?$"           # the same sequence cut off at the end of the string
_RE_LOST_SCRIPT = r"\u00fd{2,}"                # "ýýýý": a whole non-Latin title flattened


# =============================================================================
# Reporting helper (prints to the screen AND is saved to cleaning_report.txt)
# =============================================================================

class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def add(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def section(self, title: str) -> None:
        self.add("")
        self.add(title)
        self.add("-" * len(title))

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# =============================================================================
# Low-level cleaning helpers
# =============================================================================

def read_csv_robust(path: Path):
    """Read a CSV as text, trying several encodings. Returns (dataframe, encoding used).

    We never rely on the machine's default encoding: on Windows that is cp1252 and
    would silently mis-read the UTF-8 files (and vice-versa on Mac/Linux).
    """
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, dtype=str,
                             keep_default_na=False, na_values=[""])
            return df, encoding
        except UnicodeDecodeError as exc:  # try the next encoding
            last_error = exc
    raise RuntimeError(f"Could not decode {path.name}: {last_error}")


def fix_text(series: pd.Series) -> pd.Series:
    """Normalise whitespace and repair what can be repaired of upstream text damage.

    The accented/non-Latin characters in the 2023/2024 files were destroyed before
    we received them, so they can't be restored -- we replace each damaged run with
    a single "�" so it is visible and searchable, and never silently guess.
    """
    s = series.astype("string")
    s = s.str.replace(_RE_LOST_SCRIPT, LOST_CHAR, regex=True)
    s = s.str.replace(_RE_DAMAGED_SEQ, LOST_CHAR, regex=True)
    s = s.str.replace(_RE_DAMAGED_TAIL, LOST_CHAR, regex=True)
    s = s.str.replace(LOST_CHAR + "{2,}", LOST_CHAR, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.fillna("x").eq(""))


def to_number(series: pd.Series, rep: Optional[Report] = None, label: str = "",
              integer: bool = False, scale: float = 1.0) -> pd.Series:
    """Parse text like '1,234,567' into numbers. Unparseable values become blank (and are reported)."""
    text = series.astype("string").str.replace(",", "", regex=False).str.strip()
    parsed = pd.to_numeric(text, errors="coerce")
    values = pd.Series(parsed.to_numpy(dtype="float64", na_value=np.nan), index=series.index)
    bad = values.isna() & text.notna().to_numpy()
    if bad.any() and rep is not None:
        example = str(series[bad].iloc[0])[:60]
        rep.add(f"  ! {label}: {int(bad.sum())} unparseable value(s) set to blank, e.g. {example!r}")
    if scale != 1.0:
        values = values * scale
    return values.round().astype("Int64") if integer else values


def to_bool(series: pd.Series) -> pd.Series:
    """'True'/'False'/'1'/'0' -> nullable boolean."""
    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out = out.mask(text.isin(["true", "1", "1.0", "yes"]).to_numpy(), True)
    out = out.mask(text.isin(["false", "0", "0.0", "no"]).to_numpy(), False)
    return out


def to_snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def drop_exact_duplicates(df: pd.DataFrame, src: str, rep: Report) -> pd.DataFrame:
    n = int(df.duplicated().sum())
    if n:
        rep.add(f"  - {n} exact duplicate row(s) removed")
    return df.drop_duplicates().reset_index(drop=True)


def drop_empty_columns(df: pd.DataFrame, rep: Report) -> pd.DataFrame:
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        rep.add(f"  - dropped column(s) that are 100% blank: {', '.join(empty)}")
    return df.drop(columns=empty)


def count_damaged(*columns: pd.Series) -> int:
    flag = pd.Series(False, index=columns[0].index)
    for col in columns:
        flag = flag | col.astype("string").str.contains(LOST_CHAR, regex=False).fillna(False).astype(bool)
    return int(flag.sum())


# =============================================================================
# Artist splitting and song-matching keys
# =============================================================================

def split_artists(value, style: str) -> List[str]:
    """Split an artist string into individual artists according to `style`."""
    if value is None or pd.isna(value):
        return []
    text = str(value)
    if style == "none":
        return [text.strip()] if text.strip() else []
    for i, name in enumerate(PROTECTED_ARTISTS):
        text = text.replace(name, f"\x00{i}\x00")
    if style == "comma":
        pattern = r"\s*,\s*"
    else:  # collab
        pattern = r"\s*,\s*|\s+(?:&|ft\.?|feat\.?|featuring)\s+"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    for i, name in enumerate(PROTECTED_ARTISTS):
        parts = [p.replace(f"\x00{i}\x00", name) for p in parts]
    return [p.strip() for p in parts if p and p.strip()]


_NON_ALNUM = re.compile(r"[^a-z0-9]")
_OPEN_BRACKET_TAIL = re.compile(r"[(\[][^)\]]*$")
_LOST_DASH = re.compile(r"\s" + LOST_CHAR + r"+\s")

# A bracketed / " - " part of a title is IGNORED when matching songs only if it is "noise":
# a credit, soundtrack tag or edition label. Everything else -- Remix, Sped Up, Live,
# Taylor's Version, Interlude ... -- is a different recording and stays part of the key,
# so those are never merged into (or deleted as duplicates of) the original.
_NOISE_SEGMENT = re.compile(
    r"^(?:feat\b|ft\b|featuring\b|with\b|from\b|con\b|prod\b)"
    r"|explicit|\bclean\b|remaster|deluxe|bonus track|soundtrack|motion picture"
    r"|original (?:motion|series|score|cast)|\bost\b|\bfrom the\b|\bvol\.? ?\d|:"
    r"|\b(?:album|single) version\b")
# A bracket that was cut off by truncation is only ignored if it clearly started as noise.
_UNCLOSED_NOISE = re.compile(r"^(?:feat|ft|with|from|con|prod|explicit|clean|remaster|deluxe)")


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _drop_noise_segments(text: str) -> str:
    def bracket(match):
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        return " " if _NOISE_SEGMENT.search(inner.strip()) else " " + inner + " "

    text = re.sub(r"\(([^)]*)\)|\[([^\]]*)\]", bracket, text)
    tail = _OPEN_BRACKET_TAIL.search(text)
    if tail:
        inner = tail.group(0)[1:].strip()
        text = text[:tail.start()] + (" " if _UNCLOSED_NOISE.search(inner) else " " + inner + " ")
    base, *dash_parts = re.split(r"\s-\s", text)
    return " ".join([base] + [p for p in dash_parts if not _NOISE_SEGMENT.search(p.strip())])


def title_key(value) -> str:
    """Match key for a song title: ignores accents, case, punctuation and credit/edition noise."""
    if value is None or pd.isna(value):
        return ""
    folded = _LOST_DASH.sub(" - ", _ascii_fold(str(value)))  # a lost en-dash looks like " � "
    return _NON_ALNUM.sub("", _drop_noise_segments(folded)) or _NON_ALNUM.sub("", folded)


def artist_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return _NON_ALNUM.sub("", _ascii_fold(str(value)))


# =============================================================================
# Per-source cleaners.  Each returns a tidy frame with the columns
#   title, artists, primary_artist, <source specific columns...>
# (unprefixed; prefixes are added when the master is assembled)
# =============================================================================

def _add_primary_artist(out: pd.DataFrame, src: str) -> pd.DataFrame:
    style = ARTIST_SPLIT_STYLE[src]
    first = out["artists"].map(lambda v: (split_artists(v, style) or [pd.NA])[0])
    out.insert(2, "primary_artist", first.astype("string"))
    if "artist_count" not in out.columns:  # spotify-2023 ships its own; the others get one computed here
        counts = out["artists"].map(lambda v: len(split_artists(v, style)))
        out.insert(3, "artist_count", counts.astype("Int64").mask(out["artists"].isna()))
    return out


def clean_sp2024(raw: pd.DataFrame, rep: Report) -> pd.DataFrame:
    df = drop_exact_duplicates(raw, "sp2024", rep)
    out = pd.DataFrame(index=df.index)
    out["title"] = fix_text(df["Track"])
    out["artists"] = fix_text(df["Artist"])
    out["album_name"] = fix_text(df["Album Name"])
    out["isrc"] = df["ISRC"].astype("string").str.strip().str.upper()
    dates = pd.to_datetime(df["Release Date"], format="%m/%d/%Y", errors="coerce")
    unparsed = int((df["Release Date"].notna() & dates.isna()).sum())
    if unparsed:
        rep.add(f"  ! Release Date: {unparsed} value(s) not in M/D/YYYY format, set to blank")
    out["release_date"] = dates
    out["release_year"] = dates.dt.year.astype("Int64")
    float_cols = {"Track Score"}
    handled = {"Track", "Artist", "Album Name", "ISRC", "Release Date", "Explicit Track"}
    for col in df.columns:
        if col in handled:
            continue
        out[to_snake(col)] = to_number(df[col], rep, f"sp2024 {col}", integer=col not in float_cols)
    out["explicit"] = to_bool(df["Explicit Track"])
    out = drop_empty_columns(out, rep)
    rep.add(f"  - text still damaged (unrecoverable) in {count_damaged(out['title'], out['artists'], out['album_name'])} row(s)")
    return _add_primary_artist(out, "sp2024")


def clean_sp2023(raw: pd.DataFrame, rep: Report) -> pd.DataFrame:
    df = drop_exact_duplicates(raw, "sp2023", rep)
    out = pd.DataFrame(index=df.index)
    out["title"] = fix_text(df["track_name"])
    out["artists"] = fix_text(df["artist(s)_name"])
    out["artist_count"] = to_number(df["artist_count"], rep, "sp2023 artist_count", integer=True)
    year = to_number(df["released_year"], rep, "sp2023 released_year")
    month = to_number(df["released_month"], rep, "sp2023 released_month")
    day = to_number(df["released_day"], rep, "sp2023 released_day")
    dates = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce")
    invalid = int((year.notna() & dates.isna()).sum())
    if invalid:
        rep.add(f"  ! {invalid} row(s) had an impossible release date (e.g. Feb 30), set to blank")
    out["release_date"] = dates
    out["release_year"] = year.round().astype("Int64")
    counts = {  # source column -> clean name
        "in_spotify_playlists": "spotify_playlists", "in_spotify_charts": "spotify_charts",
        "streams": "streams", "in_apple_playlists": "apple_playlists",
        "in_apple_charts": "apple_charts", "in_deezer_playlists": "deezer_playlists",
        "in_deezer_charts": "deezer_charts", "in_shazam_charts": "shazam_charts",
    }
    for col, name in counts.items():
        out[name] = to_number(df[col], rep, f"sp2023 {col}", integer=True)
    out["bpm"] = to_number(df["bpm"], rep, "sp2023 bpm", integer=True)
    out["key"] = df["key"].astype("string").str.strip()
    out["mode"] = df["mode"].astype("string").str.strip().str.title()
    for col in [c for c in df.columns if c.endswith("_%")]:  # 0-100 -> 0-1 to match the other files
        out[col[:-2]] = to_number(df[col], rep, f"sp2023 {col}", scale=0.01).round(2)
    out = drop_empty_columns(out, rep)
    rep.add(f"  - text damaged (unrecoverable) in {count_damaged(out['title'], out['artists'])} row(s)")
    return _add_primary_artist(out, "sp2023")


def _clean_curated_songs(raw: pd.DataFrame, src: str, rep: Report, streams_col: str,
                         rank_col: str, extra_int: Sequence[str], extra_text: Sequence[str]) -> pd.DataFrame:
    df = drop_exact_duplicates(raw, src, rep)
    out = pd.DataFrame(index=df.index)
    out["title"] = fix_text(df["song_title"])
    out["artists"] = fix_text(df["artist"])
    out["rank"] = to_number(df[rank_col], rep, f"{src} {rank_col}", integer=True)
    out["total_streams" if src == "alltime" else "streams"] = to_number(
        df[streams_col], rep, f"{src} {streams_col}", integer=True, scale=1e9)  # billions -> raw count
    for col in extra_text:
        out[col] = fix_text(df[col])
    for col in extra_int:
        out[col] = to_number(df[col], rep, f"{src} {col}", integer=True)
    out["explicit"] = to_bool(df["explicit"])
    for col in ("danceability", "energy", "valence", "acousticness"):
        out[col] = to_number(df[col], rep, f"{src} {col}").round(2)
    return drop_empty_columns(out, rep)


def clean_alltime(raw: pd.DataFrame, rep: Report) -> pd.DataFrame:
    out = _clean_curated_songs(
        raw, "alltime", rep, streams_col="total_streams_billions", rank_col="alltime_rank",
        extra_int=["bpm", "release_year"], extra_text=["primary_genre", "artist_country"])
    return _add_primary_artist(out, "alltime")


def clean_wrapped2025(raw: pd.DataFrame, rep: Report) -> pd.DataFrame:
    out = _clean_curated_songs(
        raw, "wrapped2025", rep, streams_col="streams_2025_billions", rank_col="wrapped_2025_rank",
        extra_int=["bpm", "duration_seconds", "release_year", "peak_global_chart_position"],
        extra_text=["primary_genre", "artist_country"])
    return _add_primary_artist(out, "wrapped2025")


def clean_artists(raw: pd.DataFrame, rep: Report) -> pd.DataFrame:
    df = drop_exact_duplicates(raw, "artists2025", rep)
    out = pd.DataFrame(index=df.index)
    out["name"] = fix_text(df["artist_name"])
    out["rank"] = to_number(df["wrapped_2025_rank"], rep, "artists rank", integer=True)
    out["monthly_listeners_millions_mar2026"] = to_number(
        df["monthly_listeners_millions_mar2026"], rep, "artists monthly_listeners")
    out["followers_millions"] = to_number(df["followers_millions"], rep, "artists followers", integer=True)
    out["grammy_wins"] = to_number(df["grammy_wins"], rep, "artists grammy_wins", integer=True)
    out["debut_year"] = to_number(df["debut_year"], rep, "artists debut_year", integer=True)
    for col in ("primary_genre", "country", "gender"):
        out[col] = fix_text(df[col])
    out["top_song"] = fix_text(df["top_2025_song"])
    out["_join_key"] = out["name"].map(artist_key)

    # The same artist can appear twice (Laufey does). Keep the better-ranked row.
    order = out.sort_values(["_join_key", "rank"], kind="mergesort")
    dupes = order[order.duplicated("_join_key", keep="first")]
    for _, row in dupes.iterrows():
        rep.add(f"  ! artist {row['name']!r} listed twice; dropped rank-{row['rank']} row "
                f"(its numbers conflicted with the better-ranked row)")
    out = order.drop_duplicates("_join_key", keep="first").sort_values("rank").reset_index(drop=True)
    out = out.rename(columns={c: f"{ARTIST_PREFIX}_{c}" for c in out.columns if c != "_join_key"})
    return out


# =============================================================================
# Optional: recover text that was destroyed upstream (spotify-2024 file, via ISRC)
# =============================================================================
# In the 2024 file, every character that was lost became one "ý" (or one "ï¿½") PER UTF-8 BYTE.
# The text is gone, but its BYTE LENGTH survived. So before trusting a title looked up by
# ISRC we re-create the damage on it and require an exact match (see _matches). A wrong
# recording is rejected -- it is never silently written into the data.

_ISRC_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$")
_UNIT = "\x01"


def _is_damaged(text) -> bool:
    return isinstance(text, str) and ("\u00fd" in text or "\u00ef" in text)


def _norm_damaged(raw: str) -> str:
    """Damaged text with each lost byte as one placeholder; a unit cut in half at the end is dropped."""
    text = re.sub(r"\u00ef\u00bf?$", "", raw)
    text = re.sub(r"\u00ef\u00bf\u00bd|\u00fd", _UNIT, text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _simulate_damage(candidate: str) -> str:
    text = "".join(ch if ord(ch) < 128 else _UNIT * len(ch.encode("utf-8")) for ch in candidate)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _matches(raw: str, candidate: str) -> Optional[str]:
    """Return `candidate` (or its apostrophe variant) if it could have produced the damaged `raw`.
    Prefix match, because damaged fields are often cut short."""
    target = _norm_damaged(raw)
    if not target or not candidate:
        return None
    # A field may only be a PREFIX of the true text if it shows signs of truncation (a unit cut in
    # half at the end) or has readable text to anchor the match. A completely lost field with
    # neither ("ýýýýýý") must match the byte length exactly -- otherwise any longer title would fit.
    can_be_prefix = bool(re.search(r"\u00ef\u00bf?$", raw)) or bool(re.search(r"[a-z0-9]", target))
    for variant in dict.fromkeys([candidate, candidate.replace("'", "\u2019"), candidate.replace("\u2019", "'")]):
        simulated = _simulate_damage(variant)
        if simulated == target or (can_be_prefix and simulated.startswith(target)):
            return variant
    return None


def _compact_musicbrainz(data: dict) -> dict:
    recordings = []
    for rec in data.get("recordings") or []:
        credit = [{"name": c.get("name") or (c.get("artist") or {}).get("name") or "",
                   "join": c.get("joinphrase") or ""} for c in rec.get("artist-credit") or []]
        releases = sorted({r.get("title") for r in rec.get("releases") or [] if r.get("title")})
        recordings.append({"title": rec.get("title") or "", "credit": credit, "releases": releases})
    return {"recordings": recordings}


def fetch_musicbrainz(isrc: str, contact: str) -> Optional[dict]:
    """Ask MusicBrainz which recording(s) carry this ISRC. Returns None on a transient failure."""
    url = f"https://musicbrainz.org/ws/2/isrc/{isrc}?fmt=json&inc=artist-credits+releases"
    request = urllib.request.Request(url, headers={
        "User-Agent": f"SpotifyMasterBuilder/1.0 ({contact})", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return _compact_musicbrainz(json.load(response))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"recordings": []}
            if exc.code not in (429, 500, 502, 503, 504):
                return None
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (ValueError, OSError):
        return {}


def recover_sp2024_text(raw: pd.DataFrame, cache_path: Path, allow_network: bool, contact: str,
                        rep: Report):
    """Replace destroyed Track / Artist / Album text in the raw 2024 frame using ISRC lookups.
    Returns (raw with recovered text, audit dataframe)."""
    df = raw.copy()
    fields = {"Track": "title", "Artist": "artist", "Album Name": "album"}
    isrc = df["ISRC"].astype("string").str.strip().str.upper()
    needs = [i for i in df.index if isrc[i] is not pd.NA and _ISRC_RE.match(str(isrc[i]))
             and any(_is_damaged(df.at[i, c]) for c in fields)]
    rep.add(f"  text recovery: {len(needs)} damaged 2024 row(s) have a valid ISRC")
    if not needs:
        return df, pd.DataFrame()

    cache = _load_cache(cache_path)
    todo = sorted({isrc[i] for i in needs} - set(cache))
    if todo and allow_network:
        rep.add(f"  looking up {len(todo)} ISRC(s) on MusicBrainz (~1 per second) ...")
        failures = 0
        for n, code in enumerate(todo, 1):
            result = fetch_musicbrainz(code, contact)
            time.sleep(1.1)  # MusicBrainz allows 1 request/second
            if result is None:
                failures += 1
                if failures >= 3 and failures == n:  # nothing has worked yet -> offline / blocked
                    rep.add("  ! MusicBrainz is unreachable from this machine; stopping lookups")
                    break
                continue
            cache[code] = result
        if any(code in cache for code in todo):  # don't write an empty cache when everything failed
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
            rep.add(f"  cache saved: {cache_path}")
    elif todo:
        rep.add(f"  {len(todo)} ISRC(s) not in the cache; re-run with --recover-text to look them up")

    audit = []

    def log(i, field, before, after, status):
        audit.append({"isrc": isrc[i], "field": field, "damaged_text": before, "recovered_text": after,
                      "status": status})

    for i in needs:
        entry = cache.get(isrc[i])
        if entry is None:
            continue
        recs = entry.get("recordings", [])
        if not recs:
            log(i, "all", df.at[i, "Track"], "", "ISRC not found")
            continue
        pick = None
        for rec in recs:  # the recording's title must reproduce the damaged/clean title
            title = _matches(str(df.at[i, "Track"]), rec.get("title", ""))
            if title:
                pick = (rec, title)
                break
        if pick is None:
            log(i, "all", df.at[i, "Track"], recs[0].get("title", ""), "rejected: title does not fit the damage")
            continue
        rec, title = pick
        if _is_damaged(df.at[i, "Track"]):
            log(i, "title", df.at[i, "Track"], title, "recovered")
            df.at[i, "Track"] = title
        if _is_damaged(df.at[i, "Artist"]):
            names = [c.get("name", "") for c in rec.get("credit", []) if c.get("name")]
            full = "".join(c.get("name", "") + c.get("join", "") for c in rec.get("credit", []))
            options = [x for x in dict.fromkeys([names[0] if names else "", full, ", ".join(names)]) if x]
            found = next((v for v in (_matches(str(df.at[i, "Artist"]), o) for o in options) if v), None)
            if found:
                log(i, "artist", df.at[i, "Artist"], found, "recovered")
                df.at[i, "Artist"] = found
            else:
                log(i, "artist", df.at[i, "Artist"], " / ".join(options), "rejected: does not fit the damage")
        if _is_damaged(df.at[i, "Album Name"]):
            options = [t + suffix for t in rec.get("releases", []) for suffix in ("", " - Single", " - EP")]
            found = next((v for v in (_matches(str(df.at[i, "Album Name"]), o) for o in options) if v), None)
            if found:
                log(i, "album", df.at[i, "Album Name"], found, "recovered")
                df.at[i, "Album Name"] = found
    report = pd.DataFrame(audit)
    if len(report):
        counts = report.groupby(["field", "status"]).size()
        rep.add("  recovery results: " + "; ".join(f"{f} {s}: {n}" for (f, s), n in counts.items()))
    return df, report



# =============================================================================
# Matching songs across (and within) files
# =============================================================================

def assign_song_ids(frames: Dict[str, pd.DataFrame], rep: Report) -> Dict[str, pd.DataFrame]:
    """Give every row a `song_id`. Rows are the same song when their normalised titles are
    equal AND they share at least one individual artist (union-find over those links)."""
    rows = []  # (source, position, title_key, artist_keys)
    for src, df in frames.items():
        style = ARTIST_SPLIT_STYLE[src]
        for pos, (title, artists) in enumerate(zip(df["title"], df["artists"])):
            keys = frozenset(k for k in (artist_key(p) for p in split_artists(artists, style)) if k)
            rows.append((src, pos, title_key(title), keys))

    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_title = defaultdict(list)
    for i, (_, _, tkey, akeys) in enumerate(rows):
        if tkey and akeys:  # rows with no usable title/artist can never be matched -> stay separate
            by_title[tkey].append(i)
    for members in by_title.values():
        owner: Dict[str, int] = {}
        for i in members:
            for akey in rows[i][3]:
                if akey in owner:
                    union(i, owner[akey])
                else:
                    owner[akey] = i

    # Deterministic numbering: sort components by (title key, artist key, first row).
    component_sort_key: Dict[int, tuple] = {}
    for i, (_, _, tkey, akeys) in enumerate(rows):
        candidate = (tkey, min(akeys) if akeys else "", i)
        root = find(i)
        if root not in component_sort_key or candidate < component_sort_key[root]:
            component_sort_key[root] = candidate
    ranked = sorted(component_sort_key, key=lambda r: component_sort_key[r])
    song_id_of_root = {root: n + 1 for n, root in enumerate(ranked)}

    ids: Dict[str, List[int]] = defaultdict(list)
    for i, (src, _, _, _) in enumerate(rows):
        ids[src].append(song_id_of_root[find(i)])
    out = {}
    for src, df in frames.items():
        df = df.copy()
        df.insert(0, "song_id", ids[src])
        out[src] = df
    return out


def dedupe_within_sources(frames: Dict[str, pd.DataFrame], rep: Report):
    """If a source lists the same song more than once (different album/release), keep the most complete row."""
    kept_frames: Dict[str, pd.DataFrame] = {}
    dropped_parts = []
    for src, df in frames.items():
        work = df.copy()
        work["_pos"] = np.arange(len(work))
        work["_filled"] = df.drop(columns=["song_id"]).notna().sum(axis=1).to_numpy()
        sort_cols, ascending = ["_filled"], [False]
        size_col = DEDUP_SIZE_COLUMN.get(src)
        if size_col and size_col in work.columns:
            sort_cols.append(size_col)
            ascending.append(False)
        sort_cols.append("_pos")
        ascending.append(True)
        ordered = work.sort_values(sort_cols, ascending=ascending, kind="mergesort", na_position="last")
        keep_mask = ~ordered.duplicated("song_id", keep="first")
        kept, dropped = ordered[keep_mask], ordered[~keep_mask]
        if len(dropped):
            rep.add(f"  {src}: {len(dropped)} row(s) were the same song listed more than once "
                    f"(other album/release); kept the most complete row of each")
            dropped_parts.append(pd.DataFrame({
                "source": src, "song_id": dropped["song_id"].to_numpy(),
                "title": dropped["title"].to_numpy(), "artists": dropped["artists"].to_numpy(),
                "reason": "same song as a more complete/higher-streamed row in the same file"}))
        kept_frames[src] = kept.sort_values("_pos").drop(columns=["_pos", "_filled"]).reset_index(drop=True)
    dropped_all = (pd.concat(dropped_parts, ignore_index=True) if dropped_parts
                   else pd.DataFrame(columns=["source", "song_id", "title", "artists", "reason"]))
    return kept_frames, dropped_all


# =============================================================================
# Assemble the master
# =============================================================================

def _coalesce(master: pd.DataFrame, field: str, prefer_undamaged: bool = False) -> pd.Series:
    cols = [f"{s}_{field}" for s in PRIORITY[field] if f"{s}_{field}" in master.columns]
    if prefer_undamaged:
        clean = [master[c].mask(master[c].str.contains(LOST_CHAR, regex=False).fillna(False).astype(bool))
                 for c in cols]
        cols_series = clean + [master[c] for c in cols]  # undamaged first, damaged as last resort
    else:
        cols_series = [master[c] for c in cols]
    result = cols_series[0]
    for nxt in cols_series[1:]:
        result = result.combine_first(nxt)
    return result


def _choose_artists(master: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
    """Pick, per song, the artist credit listing the most individual artists, and derive
    artists / artists_list / artist_count / primary_artist from that one string."""
    chosen, lists = [], []
    for i in master.index:
        best = None  # (n_artists, undamaged, -priority_rank, text, parts)
        for rank, src in enumerate(ARTIST_ORDER):
            text = master.at[i, f"{src}_artists"]
            if pd.isna(text):
                continue
            parts = split_artists(text, ARTIST_SPLIT_STYLE[src])
            cand = (len(parts), LOST_CHAR not in text, -rank, text, parts)
            if best is None or cand[:3] > best[:3]:
                best = cand
        chosen.append(best[3] if best else pd.NA)
        lists.append(best[4] if best else [])
    unified["artists"] = pd.Series(chosen, index=master.index, dtype="string")
    unified["artists_list"] = pd.Series([ARTIST_LIST_SEP.join(x) if x else pd.NA for x in lists],
                                        index=master.index, dtype="string")
    unified["artist_count"] = pd.Series([len(x) if x else pd.NA for x in lists],
                                        index=master.index, dtype="Int64")
    unified["primary_artist"] = pd.Series([x[0] if x else pd.NA for x in lists],
                                          index=master.index, dtype="string")
    return unified


def build_master(frames: Dict[str, pd.DataFrame], artists: pd.DataFrame, rep: Report) -> pd.DataFrame:
    master: Optional[pd.DataFrame] = None
    for src in SONG_SOURCES:
        df = frames[src].drop(columns=["primary_artist"], errors="ignore")
        df = df.rename(columns={c: f"{src}_{c}" for c in df.columns if c != "song_id"})
        # keep primary artist separately so the unified column can use it
        pa = frames[src][["song_id", "primary_artist"]].rename(columns={"primary_artist": f"{src}_primary_artist"})
        df = df.merge(pa, on="song_id", how="left", validate="one_to_one")
        master = df if master is None else master.merge(df, on="song_id", how="outer", validate="one_to_one")
    assert master is not None
    master = master.sort_values("song_id").reset_index(drop=True)

    for src in SONG_SOURCES:
        master[f"in_{src}"] = master["song_id"].isin(frames[src]["song_id"]).to_numpy()
    master["n_sources"] = master[[f"in_{s}" for s in SONG_SOURCES]].sum(axis=1).astype(int)

    unified = pd.DataFrame({"song_id": master["song_id"]})
    for field in PRIORITY:
        unified[field] = _coalesce(master, field, prefer_undamaged=field in TEXT_FIELDS)
    unified = _choose_artists(master, unified)
    unified["text_damaged"] = (unified["title"].str.contains(LOST_CHAR, regex=False).fillna(False).astype(bool)
                               | unified["artists"].str.contains(LOST_CHAR, regex=False).fillna(False).astype(bool))

    audio_source = pd.Series(pd.NA, index=master.index, dtype="string")
    for src in reversed(PRIORITY["bpm"]):  # reversed so the highest-priority source is applied last
        audio_source = audio_source.mask(master[f"{src}_bpm"].notna().to_numpy(), src)
    unified["audio_features_source"] = audio_source

    # Join artist-level attributes on the song's primary artist.
    unified["_join_key"] = unified["primary_artist"].map(artist_key)
    artist_cols = [c for c in artists.columns if c != "_join_key"]
    before = len(unified)
    unified = unified.merge(artists, on="_join_key", how="left", validate="many_to_one")
    assert len(unified) == before, "artist join changed the row count"
    unified = unified.drop(columns=["_join_key"])
    matched = int(unified[f"{ARTIST_PREFIX}_rank"].notna().sum())
    rep.add(f"  {matched} of {before} songs matched to a Wrapped-2025 top-50 artist "
            f"({artists.shape[0]} artists available)")

    flag_cols = ["n_sources"] + [f"in_{s}" for s in SONG_SOURCES]
    front = ["song_id", "title", "primary_artist", "artists", "artists_list", "artist_count", "text_damaged"] + flag_cols + [
        "release_date", "release_year", "explicit", "primary_genre", "artist_country",
        "bpm", "danceability", "energy", "valence", "acousticness", "audio_features_source"]
    per_source = [c for s in SONG_SOURCES for c in master.columns if c.startswith(f"{s}_")]
    per_source = [c for c in per_source if not c.endswith("_primary_artist")]
    unified_flags = master[["song_id"] + flag_cols]
    result = unified.merge(unified_flags, on="song_id", validate="one_to_one")
    result = result.merge(master[["song_id"] + per_source], on="song_id", validate="one_to_one")
    ordered = front + artist_cols + per_source
    return result[ordered].sort_values("song_id").reset_index(drop=True)


# =============================================================================
# Validation + diagnostics
# =============================================================================

def validate(master: pd.DataFrame, frames: Dict[str, pd.DataFrame], rep: Report) -> None:
    rep.section("Validation")
    if master["song_id"].duplicated().any():
        raise RuntimeError("song_id is not unique in the master -- this is a bug")
    problems = []
    for src in SONG_SOURCES:
        if int(master[f"in_{src}"].sum()) != len(frames[src]):
            problems.append(f"row count mismatch for {src}")
    for col in ("danceability", "energy", "valence", "acousticness"):
        bad = int((~master[col].dropna().between(0, 1)).sum())
        if bad:
            problems.append(f"{col}: {bad} value(s) outside 0-1")
    years = master["release_year"].dropna()
    bad_years = int((~years.between(1900, pd.Timestamp.today().year)).sum())
    if bad_years:
        problems.append(f"release_year: {bad_years} value(s) outside 1900-today")
    if problems:
        for p in problems:
            rep.add(f"  ! {p}")
    else:
        rep.add("  OK: song_id unique, per-source row counts reconcile, audio features within 0-1, years plausible")


def report_conflicts(master: pd.DataFrame, rep: Report) -> None:
    rep.section("Do the sources agree where they overlap?")
    rep.add("(Both values are kept in the master; the unified column follows PRIORITY.)")
    for field, tol in CONFLICT_CHECKS:
        for a, b in combinations(PRIORITY[field], 2):
            ca, cb = f"{a}_{field}", f"{b}_{field}"
            if ca not in master.columns or cb not in master.columns:
                continue
            both = master[[ca, cb]].dropna()
            if len(both) < 5:
                continue
            if str(master[ca].dtype) == "boolean":
                agree = (both[ca].astype(bool) == both[cb].astype(bool)).mean()
            else:
                agree = ((both[ca].astype(float) - both[cb].astype(float)).abs() <= tol).mean()
            flag = "   <-- large disagreement" if agree < 0.9 else ""
            rep.add(f"  {field:<13} {a:>11} vs {b:<11} {agree:6.1%} agree  (n={len(both)}){flag}")


def summarise(master: pd.DataFrame, rep: Report) -> None:
    rep.section("Summary")
    rep.add(f"  Master: {len(master):,} unique songs x {master.shape[1]} columns")
    for n, count in master["n_sources"].value_counts().sort_index().items():
        rep.add(f"    found in {n} source file(s): {count:,} songs")
    rep.add(f"  Songs whose title/artist text is partly destroyed upstream: {int(master['text_damaged'].sum()):,}")
    unmatched_note = ("  Songs with damaged names are less likely to match across files, so a few songs may "
                      "appear as two rows.")
    rep.add(unmatched_note)


# =============================================================================
# Entry point
# =============================================================================

def find_input_dir(explicit: Optional[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path(explicit).expanduser()] if explicit else [script_dir / "data", script_dir, Path.cwd()]
    tried = []
    for folder in candidates:
        missing = [f for f in SOURCE_FILES.values() if not (folder / f).is_file()]
        if not missing:
            return folder
        tried.append(f"  {folder}  (missing: {', '.join(missing)})")
    raise FileNotFoundError("Could not find all five CSV files. Looked in:\n" + "\n".join(tried)
                            + "\nUse --input-dir to point at the folder that contains them.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean the Spotify CSVs and build one master dataset.")
    parser.add_argument("--input-dir", help="folder containing the five CSV files")
    parser.add_argument("--output-dir", help="folder for results (default: ./output next to this script)")
    parser.add_argument("--recover-text", action="store_true",
                        help="look up destroyed 2024 titles/artists/albums by ISRC on MusicBrainz (needs internet; "
                             "results are cached in text_recovery_cache.json and reused offline afterwards)")
    parser.add_argument("--contact", default=DEFAULT_CONTACT,
                        help="your email or project name, sent to MusicBrainz in the User-Agent (they ask for one)")
    parser.add_argument("--cache-file", help=f"recovery cache (default: {CACHE_FILENAME} next to this script)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # never crash on a console that can't show "�" or accents
        sys.stdout.reconfigure(errors="replace")

    rep = Report()
    try:
        input_dir = find_input_dir(args.input_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rep.add("Spotify master dataset builder")
    rep.add(f"Input folder : {input_dir}")
    rep.add(f"pandas {pd.__version__}")

    cleaners = {"sp2023": clean_sp2023, "sp2024": clean_sp2024, "alltime": clean_alltime,
                "wrapped2025": clean_wrapped2025, "artists2025": clean_artists}
    cleaned: Dict[str, pd.DataFrame] = {}
    recovery_audit = pd.DataFrame()
    cache_path = Path(args.cache_file).expanduser() if args.cache_file else Path(__file__).resolve().parent / CACHE_FILENAME
    rep.section("1. Reading and cleaning each file")
    for key, filename in SOURCE_FILES.items():
        raw, encoding = read_csv_robust(input_dir / filename)
        rep.add(f"{filename}: {len(raw):,} rows x {raw.shape[1]} cols (read as {encoding})")
        if key == "sp2024":
            raw, recovery_audit = recover_sp2024_text(raw, cache_path, args.recover_text, args.contact, rep)
        cleaned[key] = cleaners[key](raw, rep)
    artists = cleaned.pop("artists2025")

    rep.section("2. Matching songs across files")
    frames = assign_song_ids(cleaned, rep)
    total_rows = sum(len(f) for f in frames.values())
    rep.add(f"  {total_rows:,} song rows across the four files -> "
            f"{len(set().union(*[set(f['song_id']) for f in frames.values()])):,} unique songs")
    frames, dropped = dedupe_within_sources(frames, rep)

    rep.section("3. Building the master")
    master = build_master(frames, artists, rep)

    validate(master, frames, rep)
    report_conflicts(master, rep)
    master_out = master.drop(columns=OUTPUT_DROP_COLUMNS, errors="ignore")  # what gets written
    summarise(master_out, rep)

    master_path = output_dir / MASTER_FILENAME
    master_out.to_csv(master_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    dropped.to_csv(output_dir / DROPPED_FILENAME, index=False, encoding="utf-8-sig")
    if len(recovery_audit):
        recovery_audit.to_csv(output_dir / RECOVERED_FILENAME, index=False, encoding="utf-8-sig")
        rep.add(f"Wrote {output_dir / RECOVERED_FILENAME}  ({len(recovery_audit)} rows: every recovery / rejection)")
    rep.add("")
    rep.add(f"Wrote {master_path}")
    rep.add(f"Wrote {output_dir / DROPPED_FILENAME}  ({len(dropped)} rows)")
    rep.save(output_dir / REPORT_FILENAME)
    print(f"Wrote {output_dir / REPORT_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
