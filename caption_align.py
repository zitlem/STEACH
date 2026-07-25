"""Align YouTube auto-captions onto STT session DB rows to auto-label training data.

Pure, stdlib-only, network-free logic so it can be unit-tested without yt-dlp or a
network. The Flask layer (training_server.py) owns all yt-dlp calls and hands the
downloaded WebVTT text + the DB rows to the functions here.

Pipeline:
    parse_vtt(vtt_text)            -> list[Cue]
    to_word_stream(cues)          -> list[{"word", "t_s"}]   (deduped, YouTube clock)
    estimate_offset(rows, words)  -> float                   (caption_time - db_time)
    label_rows(rows, words, off)  -> list[RowLabel]          (per-row corrected_text)
    filter_labels(labels)         -> (kept, dropped)         (drop junk for auto mode)
    match_report(kept)            -> {row_count, mean_similarity, ...}

Video resolution (channel + session date -> video id):
    session_datetime(rows)        -> {"date", "start_wallclock", "duration_s", ...}
    pick_video(candidates, sess)  -> {"video_id", "ambiguous", "ranked"}
"""

from __future__ import annotations

import bisect
import html
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Text normalization (mirrors the main STT app's hallucination-check normalizer:
# lowercase, drop apostrophes, strip punctuation, collapse whitespace).
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: Optional[str]) -> str:
    """Lowercase, strip apostrophes/punctuation, and collapse whitespace."""
    if not text:
        return ""
    t = text.strip().lower()
    t = t.replace("'", "").replace("’", "").replace("‘", "")
    t = _PUNCT_RE.sub("", t)
    return " ".join(t.split())


def _words(text: Optional[str]) -> List[str]:
    return normalize(text).split()


# Known YouTube-outro hallucinations Whisper/auto-subs emit. Rows matching these
# are dropped so the model never learns them. Same intent as the main app filter.
HALLUCINATION_PHRASES: Tuple[str, ...] = (
    "subscribe",
    "for watching",
    "please subscribe",
    "like and subscribe",
    "dont forget to subscribe",
    "see you in the next video",
)


# ---------------------------------------------------------------------------
# WebVTT parsing
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    start_s: float
    end_s: float
    text: str  # raw cue text, may still contain <ts>/<c> tags


_TIMING_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)
_TS_TAG_RE = re.compile(r"<(\d{2}):(\d{2}):(\d{2})\.(\d{3})>")
_C_TAG_RE = re.compile(r"</?c[^>]*>")


def _hms_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(vtt_text: str) -> List[Cue]:
    """Parse WebVTT (incl. yt-dlp auto-sub output) into a list of Cues.

    Cue text keeps inline word-timing tags (`<00:00:01.234>`) so `to_word_stream`
    can recover per-word times; `<c>` styling tags are dropped. Header/NOTE/STYLE
    blocks are ignored.
    """
    cues: List[Cue] = []
    lines = vtt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    n = len(lines)
    while i < n:
        m = _TIMING_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start = _hms_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _hms_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1
        text_lines: List[str] = []
        while i < n and lines[i].strip() != "" and not _TIMING_RE.search(lines[i]):
            text_lines.append(lines[i])
            i += 1
        text = html.unescape(" ".join(tl.strip() for tl in text_lines if tl.strip()))
        if text:
            cues.append(Cue(start_s=start, end_s=end, text=text))
    return cues


def _cue_words(cue: Cue) -> List[Tuple[str, float]]:
    """Split a cue into (raw_word, time_s) pairs using inline word timings.

    Words before the first `<ts>` tag inherit the cue start time; each `<ts>`
    tag sets the time for the words that follow it (the yt-dlp auto-sub layout
    `word0<ts1><c> word1</c><ts2><c> word2</c>`).
    """
    raw = _C_TAG_RE.sub("", cue.text)
    pairs: List[Tuple[str, float]] = []
    pos = 0
    cur_t = cue.start_s
    for m in _TS_TAG_RE.finditer(raw):
        for w in raw[pos:m.start()].split():
            pairs.append((w, cur_t))
        cur_t = _hms_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        pos = m.end()
    for w in raw[pos:].split():
        pairs.append((w, cur_t))
    return pairs


def _overlap_len(prev: Sequence[str], cur: Sequence[str], max_check: int = 60) -> int:
    """Largest k such that prev's last k words == cur's first k words."""
    tail = list(prev[-max_check:])
    for k in range(min(len(tail), len(cur)), 0, -1):
        if tail[-k:] == list(cur[:k]):
            return k
    return 0


def to_word_stream(cues: Sequence[Cue]) -> List[Dict[str, object]]:
    """Flatten cues into a chronological, normalized, de-duplicated word stream.

    yt-dlp auto-subs emit rolling cues where each cue repeats most of the previous
    cue's words and appends a few new ones. We de-duplicate by rolling-suffix
    overlap: append only the tail of each cue that isn't already the head-overlap
    of what we've emitted. This keeps genuinely repeated words spoken far apart
    (their cues won't overlap) while collapsing the caption rolling repeats.
    """
    out_words: List[str] = []
    out_times: List[float] = []
    for cue in cues:
        pairs = [(normalize(raw_w), t) for raw_w, t in _cue_words(cue)]
        pairs = [(w, t) for w, t in pairs if w]
        if not pairs:
            continue
        cue_norm = [w for w, _ in pairs]
        k = _overlap_len(out_words, cue_norm)
        for w, t in pairs[k:]:
            out_words.append(w)
            out_times.append(float(t))
    return [{"word": w, "t_s": t} for w, t in zip(out_words, out_times)]


# ---------------------------------------------------------------------------
# DB row helpers
# ---------------------------------------------------------------------------

def _row_bounds(row: Dict[str, object]) -> Tuple[float, float]:
    start = float(row.get("start_time") or 0.0)  # type: ignore[arg-type]
    end_raw = row.get("end_time")
    end = float(end_raw) if end_raw is not None else start  # type: ignore[arg-type]
    if end < start:
        end = start
    return start, end


def _db_word_stream(db_rows: Sequence[Dict[str, object]]) -> List[Tuple[str, float]]:
    """Approximate a (normalized_word, time_s) stream for the DB transcript by
    spreading each row's words evenly across its [start, end] window."""
    out: List[Tuple[str, float]] = []
    for row in db_rows:
        words = _words(row.get("text"))  # type: ignore[arg-type]
        if not words:
            continue
        start, end = _row_bounds(row)
        span = end - start
        n = len(words)
        for idx, w in enumerate(words):
            out.append((w, start + span * (idx + 0.5) / n))
    return out


# ---------------------------------------------------------------------------
# Offset estimation
# ---------------------------------------------------------------------------

def _densest_cluster(values: Sequence[float], window: float) -> float:
    """Return the median of the densest `window`-wide cluster of values."""
    vs = sorted(values)
    if not vs:
        return 0.0
    best_lo = 0
    best_hi = 0
    best_count = 0
    j = 0
    for i in range(len(vs)):
        while vs[i] - vs[j] > window:
            j += 1
        if i - j + 1 > best_count:
            best_count = i - j + 1
            best_lo, best_hi = j, i
    return statistics.median(vs[best_lo:best_hi + 1])


def estimate_offset(
    db_rows: Sequence[Dict[str, object]],
    caption_words: Sequence[Dict[str, object]],
    max_common: int = 5,
    cluster_window: float = 2.0,
) -> float:
    """Estimate the offset between the caption clock and the session clock.

    Returns `caption_time - db_time`, so a caption word at `cap_t` maps to the
    session timeline at `cap_t - offset`. Works by collecting, for each DB word
    that also appears (not too commonly) in the captions, the time differences of
    every matching caption occurrence, then taking the densest cluster of those
    differences. Returns 0.0 when there is no usable overlap.
    """
    db_words = _db_word_stream(db_rows)
    if not db_words or not caption_words:
        return 0.0

    cap_index: Dict[str, List[float]] = defaultdict(list)
    for cw in caption_words:
        cap_index[str(cw["word"])].append(float(cw["t_s"]))  # type: ignore[arg-type]

    diffs: List[float] = []
    for w, t in db_words:
        times = cap_index.get(w)
        if not times or len(times) > max_common:
            continue  # skip absent or overly common (ambiguous) words
        for ct in times:
            diffs.append(ct - t)

    if not diffs:
        return 0.0
    return _densest_cluster(diffs, cluster_window)


# ---------------------------------------------------------------------------
# Anchor-based piecewise alignment
#
# A single global offset drifts over a long service (the STT and YouTube clocks
# don't advance identically — pauses, re-syncs). Instead, find anchor points where
# a distinctive word occurs exactly once in each stream, giving trustworthy
# (caption_time -> db_time) pairs, keep only a monotonically consistent subset, and
# interpolate between them. This tracks drift across the whole recording.
# ---------------------------------------------------------------------------

def build_anchors(
    db_rows: Sequence[Dict[str, object]],
    caption_words: Sequence[Dict[str, object]],
    min_word_len: int = 4,
) -> List[Tuple[float, float]]:
    """Return sorted, monotonically-consistent (caption_time, db_time) anchor pairs.

    Anchors come from distinctive words (length >= min_word_len) that appear exactly
    once in both the DB transcript and the captions, so each pair is unambiguous.
    A longest-increasing-subsequence filter then drops pairs that would require the
    timelines to run backwards relative to each other (mismatched coincidences).
    """
    db_idx: Dict[str, List[float]] = defaultdict(list)
    for w, t in _db_word_stream(db_rows):
        db_idx[w].append(t)
    cap_idx: Dict[str, List[float]] = defaultdict(list)
    for cw in caption_words:
        cap_idx[str(cw["word"])].append(float(cw["t_s"]))  # type: ignore[arg-type]

    pairs: List[Tuple[float, float]] = []
    for w, cts in cap_idx.items():
        dts = db_idx.get(w)
        if dts and len(cts) == 1 and len(dts) == 1 and len(w) >= min_word_len:
            pairs.append((cts[0], dts[0]))
    pairs.sort()
    return _longest_increasing_by_db(pairs)


def _longest_increasing_by_db(pairs: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Keep the longest subsequence whose db_time is non-decreasing as caption_time
    increases (pairs already sorted by caption_time)."""
    n = len(pairs)
    if n <= 1:
        return list(pairs)
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if pairs[j][1] <= pairs[i][1] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: best[i])
    out: List[Tuple[float, float]] = []
    while end != -1:
        out.append(pairs[end])
        end = prev[end]
    out.reverse()
    return out


def map_caption_time(anchors: Sequence[Tuple[float, float]], cap_t: float) -> float:
    """Map a caption timestamp to the session timeline via piecewise-linear
    interpolation between anchors (constant-offset extrapolation past the ends).
    Falls back to identity when there are no anchors."""
    if not anchors:
        return cap_t
    caps = [a[0] for a in anchors]
    if cap_t <= caps[0]:
        return cap_t + (anchors[0][1] - anchors[0][0])
    if cap_t >= caps[-1]:
        return cap_t + (anchors[-1][1] - anchors[-1][0])
    i = bisect.bisect_right(caps, cap_t) - 1
    c0, d0 = anchors[i]
    c1, d1 = anchors[i + 1]
    if c1 == c0:
        return d0
    return d0 + (d1 - d0) * (cap_t - c0) / (c1 - c0)


# ---------------------------------------------------------------------------
# Row labeling
# ---------------------------------------------------------------------------

@dataclass
class RowLabel:
    transcription_id: Optional[int]
    start_time: float
    end_time: float
    db_text: str
    corrected_text: str
    similarity: float
    drop_reason: Optional[str] = None

    def to_segment(self) -> Dict[str, object]:
        """Shape expected by training_server's clip-extraction helper."""
        return {
            "transcription_id": self.transcription_id,
            "corrected_text": self.corrected_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def to_dict(self) -> Dict[str, object]:
        """Full serializable view for the review UI (before training)."""
        return {
            "transcription_id": self.transcription_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "db_text": self.db_text,
            "corrected_text": self.corrected_text,
            "similarity": round(self.similarity, 3),
            "drop_reason": self.drop_reason,
        }


def _similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na.split(), nb.split()).ratio()


def _assign_row(t: float, bounds: Sequence[Tuple[float, float]], pad_s: float) -> Optional[int]:
    """Pick the single row a caption word at session-time `t` belongs to: the row
    whose [start, end] contains it, else the nearest row within `pad_s` seconds."""
    for i, (start, end) in enumerate(bounds):
        if start <= t <= end:
            return i
    best_i: Optional[int] = None
    best_d = pad_s
    for i, (start, end) in enumerate(bounds):
        d = min(abs(t - start), abs(t - end))
        if d <= best_d:
            best_d = d
            best_i = i
    return best_i


def label_rows(
    db_rows: Sequence[Dict[str, object]],
    caption_words: Sequence[Dict[str, object]],
    offset: float,
    pad_s: float = 0.5,
) -> List[RowLabel]:
    """Assign caption words to DB rows by time overlap (offset-corrected).

    Each caption word maps to the session timeline as `cap_t - offset` and is
    assigned to exactly one row — the row containing it, or the nearest row within
    `pad_s` (so a boundary word is never counted twice). The joined words become
    that row's `corrected_text`; `similarity` compares it to the row's DB text.
    """
    rows = list(db_rows)
    bounds = [_row_bounds(r) for r in rows]
    buckets: List[List[str]] = [[] for _ in rows]

    projected = sorted(
        ((float(cw["t_s"]) - offset, str(cw["word"])) for cw in caption_words),  # type: ignore[arg-type]
        key=lambda x: x[0],
    )
    for t, w in projected:
        idx = _assign_row(t, bounds, pad_s)
        if idx is not None:
            buckets[idx].append(w)

    labels: List[RowLabel] = []
    for row, (start, end), words in zip(rows, bounds, buckets):
        corrected = " ".join(words)
        db_text = str(row.get("text") or "")
        labels.append(RowLabel(
            transcription_id=row.get("id"),  # type: ignore[arg-type]
            start_time=start,
            end_time=end,
            db_text=db_text,
            corrected_text=corrected,
            similarity=_similarity(db_text, corrected),
        ))
    return labels


def label_rows_anchored(
    db_rows: Sequence[Dict[str, object]],
    caption_words: Sequence[Dict[str, object]],
    anchors: Sequence[Tuple[float, float]],
    pad_s: float = 0.5,
) -> List[RowLabel]:
    """Like `label_rows`, but map each caption word onto the session timeline with
    the piecewise-linear anchor map (tracking clock drift) instead of one offset."""
    rows = list(db_rows)
    bounds = [_row_bounds(r) for r in rows]
    buckets: List[List[str]] = [[] for _ in rows]

    projected = sorted(
        ((map_caption_time(anchors, float(cw["t_s"])), str(cw["word"])) for cw in caption_words),  # type: ignore[arg-type]
        key=lambda x: x[0],
    )
    for t, w in projected:
        idx = _assign_row(t, bounds, pad_s)
        if idx is not None:
            buckets[idx].append(w)

    labels: List[RowLabel] = []
    for row, (start, end), words in zip(rows, bounds, buckets):
        corrected = " ".join(words)
        db_text = str(row.get("text") or "")
        labels.append(RowLabel(
            transcription_id=row.get("id"),  # type: ignore[arg-type]
            start_time=start,
            end_time=end,
            db_text=db_text,
            corrected_text=corrected,
            similarity=_similarity(db_text, corrected),
        ))
    return labels


# ---------------------------------------------------------------------------
# Filtering (the "fully automatic" safety net)
# ---------------------------------------------------------------------------

def _is_hallucination(text: str) -> bool:
    n = normalize(text)
    return any(normalize(p) in n for p in HALLUCINATION_PHRASES)


def filter_labels(
    labels: Sequence[RowLabel],
    min_len_ratio: float = 0.34,
    max_len_ratio: float = 3.0,
) -> Tuple[List[RowLabel], List[RowLabel]]:
    """Split labels into (kept, dropped). Drops rows whose caption label is empty,
    is a known YouTube hallucination, or whose word-count vs the DB text is wildly
    off (a strong misalignment signal). Dropped rows carry a `drop_reason`.
    """
    kept: List[RowLabel] = []
    dropped: List[RowLabel] = []
    for lb in labels:
        corrected_words = len(_words(lb.corrected_text))
        db_words = len(_words(lb.db_text))
        reason: Optional[str] = None
        if corrected_words == 0:
            reason = "empty"
        elif _is_hallucination(lb.corrected_text):
            reason = "hallucination"
        elif db_words:
            ratio = corrected_words / db_words
            if ratio < min_len_ratio or ratio > max_len_ratio:
                reason = "length_mismatch"
        if reason:
            lb.drop_reason = reason
            dropped.append(lb)
        else:
            kept.append(lb)
    return kept, dropped


def match_report(
    kept: Sequence[RowLabel],
    low_match_threshold: float = 0.5,
) -> Dict[str, object]:
    """Summarize agreement between DB text and caption labels for the kept rows.

    A low `mean_similarity` (or many `low_match_rows`) means the captions and the
    session are poorly aligned — fix offset/language before training on them.
    """
    sims = [lb.similarity for lb in kept]
    low = [lb.transcription_id for lb in kept if lb.similarity < low_match_threshold]
    return {
        "row_count": len(kept),
        "mean_similarity": round(statistics.mean(sims), 4) if sims else 0.0,
        "low_match_count": len(low),
        "low_match_rows": low,
    }


# ---------------------------------------------------------------------------
# Video resolution: channel + session date -> video id
# ---------------------------------------------------------------------------

def session_datetime(db_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Derive the service's local date, wall-clock start, and duration from rows.

    Uses the first row's `timestamp` ("%Y-%m-%d %H:%M:%S") for the date/start and
    the span of `start_time`..`end_time` for the duration. Fields are None when the
    inputs don't allow deriving them.
    """
    result: Dict[str, object] = {
        "date": None,
        "start_wallclock": None,
        "duration_s": 0.0,
        "start_epoch": None,
    }
    if not db_rows:
        return result

    first = db_rows[0]
    ts = first.get("timestamp")
    if ts:
        try:
            dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
            result["date"] = dt.strftime("%Y-%m-%d")
            result["start_wallclock"] = dt.strftime("%H:%M:%S")
            result["start_epoch"] = dt.timestamp()
        except (ValueError, OverflowError):
            pass

    starts = [_row_bounds(r)[0] for r in db_rows]
    ends = [_row_bounds(r)[1] for r in db_rows]
    if starts and ends:
        result["duration_s"] = round(max(ends) - min(starts), 2)
    return result


def _parse_upload_date(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y%m%d")
    except ValueError:
        return None


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}
_MONTHS.update({name[:3]: i for name, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_TITLE_MONTH_RE = re.compile(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})")
_TITLE_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_TITLE_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp][Mm])?")


def parse_title_datetime(title: Optional[str]) -> Optional[datetime]:
    """Parse a date (and optional time) from a video title like
    "July 5, 2026 - 10:00 AM" or "2026-07-05". Returns None if no date is found.

    Titles carry the exact service date/time, unlike YouTube's approximate flat-list
    upload dates (which round coarser for older videos), so this is the reliable key
    for matching a channel video to a session.
    """
    if not title:
        return None
    t = str(title)
    dt: Optional[datetime] = None
    m = _TITLE_MONTH_RE.search(t)
    if m and m.group(1).lower() in _MONTHS:
        try:
            dt = datetime(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            dt = None
    if dt is None:
        mi = _TITLE_ISO_RE.search(t)
        if mi:
            try:
                dt = datetime(int(mi.group(1)), int(mi.group(2)), int(mi.group(3)))
            except ValueError:
                dt = None
    if dt is None:
        return None
    tm = _TITLE_TIME_RE.search(t)
    if tm:
        hh, mm, ap = int(tm.group(1)), int(tm.group(2)), tm.group(3)
        if ap:
            ap = ap.lower()
            if ap == "pm" and hh != 12:
                hh += 12
            elif ap == "am" and hh == 12:
                hh = 0
        if 0 <= hh < 24 and 0 <= mm < 60:
            dt = dt.replace(hour=hh, minute=mm)
    return dt


def _title_minutes(title: Optional[str]) -> Optional[int]:
    """Minutes-since-midnight from a title's time, or None when no explicit time."""
    dt = parse_title_datetime(title)
    if dt is None or (dt.hour == 0 and dt.minute == 0):
        return None
    return dt.hour * 60 + dt.minute


def _session_minutes(session: Dict[str, object]) -> Optional[int]:
    sw = session.get("start_wallclock")
    if not sw:
        return None
    try:
        hh, mm, _ss = str(sw).split(":")
        return int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        return None


def pick_video(
    candidates: Sequence[Dict[str, object]],
    session: Dict[str, object],
    day_window: int = 1,
) -> Dict[str, object]:
    """Choose the channel video that matches the session date.

    `candidates` are yt-dlp video dicts ({id, upload_date "YYYYMMDD", title,
    duration}). Keeps videos whose upload_date is within +/- `day_window` days of
    the session date (tolerating UTC/local timezone edges), then ranks by date
    closeness, then by duration closeness to the session. Flags `ambiguous` when
    more than one video shares the exact session date.
    """
    result: Dict[str, object] = {"video_id": None, "ambiguous": False, "ranked": []}
    session_date_str = session.get("date")
    if not session_date_str:
        return result
    try:
        session_date = datetime.strptime(str(session_date_str), "%Y-%m-%d")
    except ValueError:
        return result

    session_duration = float(session.get("duration_s") or 0.0)  # type: ignore[arg-type]
    session_min = _session_minutes(session)

    scored: List[Tuple[int, float, float, Dict[str, object]]] = []
    same_day = 0
    for cand in candidates:
        # Prefer the exact date parsed from the title; fall back to upload_date.
        cdt = parse_title_datetime(cand.get("title")) or _parse_upload_date(cand.get("upload_date"))  # type: ignore[arg-type]
        if cdt is None:
            continue
        day_diff = abs((cdt.date() - session_date.date()).days)
        if day_diff > day_window:
            continue
        if day_diff == 0:
            same_day += 1
        # Same-day tie-break: closeness of the service time (title) to the session.
        tmin = _title_minutes(cand.get("title"))  # type: ignore[arg-type]
        time_diff = (
            abs(tmin - session_min)
            if (tmin is not None and session_min is not None)
            else float("inf")
        )
        cand_duration = float(cand.get("duration") or 0.0)  # type: ignore[arg-type]
        dur_diff = (
            abs(cand_duration - session_duration)
            if (cand_duration and session_duration)
            else float("inf")
        )
        scored.append((day_diff, time_diff, dur_diff, cand))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    ranked = [
        {
            "id": c.get("id"),
            "title": c.get("title"),
            "upload_date": c.get("upload_date"),
            "duration": c.get("duration"),
            "day_diff": dd,
        }
        for dd, _tdiff, _dur, c in scored
    ]
    result["ranked"] = ranked
    if ranked:
        result["video_id"] = ranked[0]["id"]
    result["ambiguous"] = same_day > 1
    return result
