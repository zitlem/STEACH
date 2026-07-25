import io
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import soundfile as sf
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO

import caption_align as ca

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    with open("config.json") as f:
        return json.load(f)

CONFIG = load_config()


def _resolve_ytdlp():
    """Locate yt-dlp independent of PATH (start.sh may launch us without ~/.local/bin).

    Prefers a yt-dlp binary; falls back to `python -m yt_dlp` so it works whenever the
    module is importable by this interpreter."""
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    for cand in (os.path.expanduser("~/.local/bin/yt-dlp"), "/usr/local/bin/yt-dlp"):
        if os.path.exists(cand):
            return [cand]
    return [sys.executable, "-m", "yt_dlp"]


YTDLP_CMD = _resolve_ytdlp()

# Remembered UI state (channel, last DB path) — kept out of the git-tracked
# config.json so it never conflicts with a git pull. Gitignored.
STATE_FILE = Path("ui_state.json")


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))
TRAINING_DATA_DIR = Path(CONFIG["paths"]["training_data_dir"])
STT_DIR = TRAINING_DATA_DIR / "stt"
STT_AUDIO_DIR = STT_DIR / "audio"
STT_MANIFEST = STT_DIR / "manifest.jsonl"
MODELS_OUTPUT_DIR = Path(CONFIG["paths"]["models_output_dir"])
MAIN_APP_DB = CONFIG["paths"]["main_app_db"]
MAIN_APP_AUDIO_BACKUP = CONFIG["paths"]["main_app_audio_backup"]

TRANSLATION_DIR = TRAINING_DATA_DIR / "translation"
TRANSLATION_MANIFEST = TRANSLATION_DIR / "manifest.jsonl"

# Local caption store: raw VTT keyed by <video_id>.<lang>.vtt, plus resolutions.json
# mapping a session DB stem -> the resolved video. Lets repeat runs (and re-uploads)
# skip YouTube entirely. Gitignored.
CAPTION_CACHE_DIR = TRAINING_DATA_DIR / "caption_cache"
RESOLUTIONS_FILE = CAPTION_CACHE_DIR / "resolutions.json"

STT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not os.access(MODELS_OUTPUT_DIR, os.W_OK):
    print(
        f"\033[33mWARNING: Models directory is not writable by this user.\n"
        f"  Fix: sudo chown -R {os.getenv('USER', 'ai')} {MODELS_OUTPUT_DIR}\033[0m",
        flush=True,
    )
if not STT_MANIFEST.exists():
    STT_MANIFEST.write_text("")
if not TRANSLATION_MANIFEST.exists():
    TRANSLATION_MANIFEST.write_text("")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "steach-training-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_training_proc = None
_training_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def read_manifest(path=None):
    path = path or STT_MANIFEST
    entries = []
    if not Path(path).exists():
        return entries
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry["id"] = i
                entries.append(entry)
            except json.JSONDecodeError:
                pass
    return entries


def write_manifest(entries, path=None):
    path = path or STT_MANIFEST
    with open(path, "w") as f:
        for e in entries:
            row = {k: v for k, v in e.items() if k != "id"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_manifest(entry, path=None):
    path = path or STT_MANIFEST
    row = {k: v for k, v in entry.items() if k != "id"}
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --- Dataset ---

@app.route("/api/dataset")
def api_dataset():
    entries = read_manifest()
    total_duration = sum(e.get("duration", 0) for e in entries)
    return jsonify({"entries": entries, "total": len(entries), "total_duration": total_duration})


@app.route("/api/dataset/<int:entry_id>", methods=["PUT"])
def api_dataset_update(entry_id):
    data = request.get_json()
    new_text = data.get("text", "").strip()
    if not new_text:
        return jsonify({"error": "text required"}), 400
    entries = read_manifest()
    if entry_id >= len(entries):
        return jsonify({"error": "not found"}), 404
    entries[entry_id]["text"] = new_text
    write_manifest(entries)
    return jsonify({"ok": True})


@app.route("/api/dataset/<int:entry_id>", methods=["DELETE"])
def api_dataset_delete(entry_id):
    entries = read_manifest()
    if entry_id >= len(entries):
        return jsonify({"error": "not found"}), 404
    entry = entries[entry_id]
    audio_file = STT_AUDIO_DIR / entry.get("audio", "")
    if audio_file.exists():
        audio_file.unlink()
    entries.pop(entry_id)
    write_manifest(entries)
    return jsonify({"ok": True})


# --- Translation dataset ---

@app.route("/api/dataset/translation")
def api_translation_dataset():
    entries = read_manifest(TRANSLATION_MANIFEST)
    return jsonify({"entries": entries, "total": len(entries)})


@app.route("/api/dataset/translation/<int:entry_id>", methods=["PUT"])
def api_translation_update(entry_id):
    data = request.get_json()
    entries = read_manifest(TRANSLATION_MANIFEST)
    if entry_id >= len(entries):
        return jsonify({"error": "not found"}), 404
    if "source" in data:
        entries[entry_id]["source"] = data["source"].strip()
    if "target" in data:
        entries[entry_id]["target"] = data["target"].strip()
    write_manifest(entries, TRANSLATION_MANIFEST)
    return jsonify({"ok": True})


@app.route("/api/dataset/translation/<int:entry_id>", methods=["DELETE"])
def api_translation_delete(entry_id):
    entries = read_manifest(TRANSLATION_MANIFEST)
    if entry_id >= len(entries):
        return jsonify({"error": "not found"}), 404
    entries.pop(entry_id)
    write_manifest(entries, TRANSLATION_MANIFEST)
    return jsonify({"ok": True})


@app.route("/api/upload/translation", methods=["POST"])
def api_upload_translation():
    """Save one or more source→target text pairs to the translation manifest.

    Expects JSON: [{source, target, source_lang, target_lang}]
    """
    items = request.get_json()
    if not items:
        return jsonify({"error": "empty body"}), 400
    results = []
    for item in items:
        source = item.get("source", "").strip()
        target = item.get("target", "").strip()
        if not source or not target:
            results.append({"error": "source and target required"})
            continue
        entry = {
            "source": source,
            "target": target,
            "source_lang": item.get("source_lang", ""),
            "target_lang": item.get("target_lang", ""),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_manifest(entry, TRANSLATION_MANIFEST)
        results.append({"ok": True})
    return jsonify({"results": results})


# --- Upload ---

SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}

def resample_to_16k_mono(src_path: Path, dst_path: Path):
    """Convert any audio file to 16kHz mono WAV using pydub."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(src_path))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(str(dst_path), format="wav")
    duration = len(audio) / 1000.0
    return duration


@app.route("/api/upload", methods=["POST"])
def api_upload():
    results = []
    files = request.files.getlist("audio")
    texts = request.form.getlist("text")

    if len(files) != len(texts):
        return jsonify({"error": "audio/text count mismatch"}), 400

    for audio_file, text in zip(files, texts):
        text = text.strip()
        if not text:
            results.append({"error": "empty text skipped"})
            continue

        ext = Path(audio_file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            results.append({"error": f"unsupported format {ext}"})
            continue

        clip_name = f"{uuid.uuid4().hex}.wav"
        tmp_path = STT_AUDIO_DIR / f"_tmp_{clip_name}"
        dst_path = STT_AUDIO_DIR / clip_name

        try:
            audio_file.save(str(tmp_path))
            duration = resample_to_16k_mono(tmp_path, dst_path)
        except Exception as e:
            results.append({"error": str(e)})
            continue
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        entry = {
            "audio": clip_name,
            "text": text,
            "duration": round(duration, 2),
            "source": "upload",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_manifest(entry)
        results.append({"ok": True, "clip": clip_name, "duration": round(duration, 2)})

    return jsonify({"results": results})


# --- DB Import (read-only from main STT app) ---

def _find_companion_wav(db_path: str) -> list:
    """Find WAV files in the same directory that share the DB's timestamp prefix."""
    db = Path(db_path)
    prefix = db.stem[:10]  # first 10 chars covers YYYY-MM-DD
    try:
        candidates = sorted(
            str(p) for p in db.parent.iterdir()
            if p.suffix.lower() == ".wav" and p.stem.startswith(prefix)
        )
    except Exception:
        candidates = []
    return candidates


@app.route("/api/import/transcriptions")
def api_import_list():
    db_path = request.args.get("db_path", MAIN_APP_DB)
    if not os.path.exists(db_path):
        return jsonify({"error": f"DB not found: {db_path}"}), 404
    offset = int(request.args.get("offset", 0))
    limit_param = request.args.get("limit")
    try:
        with sqlite3.connect(f"file:{db_path}?immutable=1", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            where = "WHERE text IS NOT NULL AND trim(text) != '' AND start_time IS NOT NULL"
            total_count = conn.execute(
                f"SELECT COUNT(*) FROM transcriptions {where}"
            ).fetchone()[0]
            query = (
                f"SELECT id, timestamp, text, start_time, end_time, translated_text "
                f"FROM transcriptions {where} ORDER BY start_time ASC"
            )
            params = []
            if limit_param is not None:
                query += " LIMIT ? OFFSET ?"
                params = [int(limit_param), offset]
            elif offset:
                query += " OFFSET ?"
                params = [offset]
            rows = conn.execute(query, params).fetchall()
        wav_candidates = _find_companion_wav(db_path)
        return jsonify({
            "transcriptions": [dict(r) for r in rows],
            "wav_candidates": wav_candidates,
            "total_count": total_count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/import/transcriptions", methods=["POST"])
def api_import_save():
    """Import selected DB transcriptions as training pairs.

    Expects JSON: [{"transcription_id": int, "corrected_text": str, "audio_path": str|null}]
    If audio_path provided, that WAV is copied/resampled into training_data.
    Otherwise saved as text-only (no audio clip).
    """
    items = request.get_json()
    if not items:
        return jsonify({"error": "empty body"}), 400

    results = []
    for item in items:
        text = item.get("corrected_text", "").strip()
        if not text:
            results.append({"error": "empty text"})
            continue

        audio_src = item.get("audio_path")
        clip_name = None
        duration = 0.0

        if audio_src and os.path.exists(audio_src):
            clip_name = f"{uuid.uuid4().hex}.wav"
            dst_path = STT_AUDIO_DIR / clip_name
            try:
                duration = resample_to_16k_mono(Path(audio_src), dst_path)
            except Exception as e:
                results.append({"error": str(e)})
                continue

        entry = {
            "audio": clip_name or "",
            "text": text,
            "duration": round(duration, 2),
            "source": "db_import",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_manifest(entry)
        results.append({"ok": True})

    return jsonify({"results": results})


def _extract_clips(wav_path, segments, source="session_extract"):
    """Slice a session WAV into 16kHz mono clips per segment and append them to
    the STT manifest. Returns (results, error) where error is (message, status)
    or None. Each segment is {corrected_text, start_time, end_time}.
    """
    if not wav_path or not os.path.exists(wav_path):
        return None, (f"WAV not found: {wav_path}", 404)
    if not segments:
        return None, ("no segments provided", 400)

    from pydub import AudioSegment as PydubAudio
    try:
        session_audio = PydubAudio.from_file(wav_path)
    except Exception as e:
        return None, (f"Failed to load WAV: {e}", 500)

    results = []
    for seg in segments:
        text = (seg.get("corrected_text") or "").strip()
        start_time = seg.get("start_time", 0.0)
        end_time = seg.get("end_time", 0.0)

        if not text:
            results.append({"error": "empty text"})
            continue
        if end_time <= start_time:
            results.append({"error": "invalid timestamps"})
            continue

        clip = session_audio[int(start_time * 1000):int(end_time * 1000)]
        clip = clip.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        clip_name = f"{uuid.uuid4().hex}.wav"
        try:
            clip.export(str(STT_AUDIO_DIR / clip_name), format="wav")
        except Exception as e:
            results.append({"error": f"Export failed: {e}"})
            continue

        duration = round(len(clip) / 1000.0, 2)
        append_manifest({
            "audio": clip_name,
            "text": text,
            "duration": duration,
            "source": source,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        results.append({"ok": True, "clip": clip_name, "duration": duration})

    return results, None


@app.route("/api/import/extract", methods=["POST"])
def api_import_extract():
    """Extract audio clips from a session WAV using DB timestamps.

    Accepts JSON:
    {
      "wav_path": "/path/to/session.wav",
      "segments": [{"corrected_text": str, "start_time": float, "end_time": float}]
    }
    Slices the WAV, resamples each clip to 16kHz mono, saves to training_data/stt/audio/.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "empty body"}), 400

    wav_path = data.get("wav_path", "").strip()
    segments = data.get("segments", [])
    results, err = _extract_clips(wav_path, segments)
    if err:
        return jsonify({"error": err[0]}), err[1]
    return jsonify({"results": results})


# --- YouTube auto-labeling (channel + session date -> captions -> training data) ---

def _read_transcription_rows(db_path):
    """Read the labelable transcription rows from a session DB (read-only).

    Same filter/order as api_import_list so alignment sees exactly the rows the
    import table would. Returns list[dict] with id/timestamp/text/start_time/end_time.
    """
    where = "WHERE text IS NOT NULL AND trim(text) != '' AND start_time IS NOT NULL"
    with sqlite3.connect(f"file:{db_path}?immutable=1", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, timestamp, text, start_time, end_time FROM transcriptions "
            f"{where} ORDER BY start_time ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _channel_tab_urls(channel, tabs):
    """Build the tab URLs to search for a channel handle/id/URL.

    Live services live under /streams for many churches, uploads under /videos, so
    by default we search both and merge. An explicit URL (with its own tab) is used
    as-is.
    """
    c = (channel or "").strip()
    if not c:
        return []
    if c.startswith("http"):
        return [c]
    if c.startswith("@"):
        base = f"https://www.youtube.com/{c}"
    elif c.startswith("UC"):
        base = f"https://www.youtube.com/channel/{c}"
    else:
        base = f"https://www.youtube.com/@{c}"
    return [f"{base}/{t}" for t in tabs]


def _dt_yyyymmdd(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y%m%d")
    except ValueError:
        return None


def _parse_video_lines(stdout):
    """Parse tab-separated 'id\\tupload_date\\tduration\\ttitle' yt-dlp --print lines."""
    out = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        up = parts[1] if len(parts) > 1 else ""
        dur = 0.0
        if len(parts) > 2:
            try:
                dur = float(parts[2])
            except ValueError:
                dur = 0.0
        title = parts[3] if len(parts) > 3 else ""
        out.append({"id": parts[0], "upload_date": up, "duration": dur, "title": title})
    return out


def _yt_flat_list(url, scan_limit):
    """Fast flat channel-tab listing with approximate upload dates.

    approximate_date is exact for recent videos and coarsens for older ones, so it
    is used only to shortlist candidates; exact dates are confirmed later when the
    fast tight-window match comes up empty. Returns (entries, error).
    """
    cmd = [
        *YTDLP_CMD, "--ignore-errors", "--no-warnings", "--skip-download",
        "--flat-playlist", "--playlist-end", str(scan_limit),
        "--extractor-args", "youtubetab:approximate_date",
        "--print", "%(id)s\t%(upload_date)s\t%(duration)s\t%(title)s",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return None, "yt-dlp is not installed (pip install yt-dlp)"
    except subprocess.TimeoutExpired:
        return None, "yt-dlp channel listing timed out"
    return _parse_video_lines(proc.stdout), None


def _yt_exact_meta(video_id):
    """Extract the exact upload_date/duration for one video. Returns dict or None."""
    cmd = [
        *YTDLP_CMD, "--no-warnings", "--skip-download",
        "--print", "%(id)s\t%(upload_date)s\t%(duration)s\t%(title)s",
        f"https://youtu.be/{video_id}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    parsed = _parse_video_lines(proc.stdout)
    return parsed[0] if parsed else None


def _yt_channel_videos(channel, date_str, day_window, scan_limit, tabs, coarse_days):
    """Resolve candidate videos near `date_str` (YYYY-MM-DD) across the channel tabs.

    Flat-lists each tab (~1s). The reliable match key is the video TITLE date
    (exact), because YouTube's flat-list "approximate" dates round coarser for older
    videos and can put the wrong videos in the target window. Title-dated videos are
    shortlisted directly; any without a parseable title date fall back to a wider
    approximate window and get their exact upload_date confirmed. Returns
    (candidates, error); candidates are {id, upload_date, title, duration}.
    """
    urls = _channel_tab_urls(channel, tabs)
    if not urls:
        return None, "no channel configured"
    center = datetime.strptime(date_str, "%Y-%m-%d")

    merged = {}
    listing_err = None
    any_ok = False
    for url in urls:
        entries, err = _yt_flat_list(url, scan_limit)
        if err:
            listing_err = err
            continue
        any_ok = True
        for e in entries:
            merged.setdefault(e["id"], e)  # first tab wins on dup
    if not any_ok:
        return None, listing_err or "yt-dlp could not list the channel"

    candidates = []
    for vid, e in merged.items():
        title_dt = ca.parse_title_datetime(e.get("title"))
        if title_dt is not None:
            # Exact date from the title — keep if within the tight window.
            if abs((title_dt.date() - center.date()).days) <= day_window:
                candidates.append({
                    "id": vid,
                    "upload_date": title_dt.strftime("%Y%m%d"),
                    "duration": e.get("duration") or 0.0,
                    "title": e.get("title", ""),
                })
            continue
        # No title date: approximate date is our only cheap signal — keep a generous
        # shortlist (approx rounds newer for old videos), confirm exact dates below.
        adt = _dt_yyyymmdd(e.get("upload_date"))
        if adt and (center - timedelta(days=2)) <= adt <= (center + timedelta(days=coarse_days)):
            exact = _yt_exact_meta(vid)
            candidates.append({
                "id": vid,
                "upload_date": (exact or {}).get("upload_date") or e.get("upload_date"),
                "duration": (exact or {}).get("duration") or e.get("duration") or 0.0,
                "title": e.get("title", ""),
            })
    return candidates, None


_VID_RE = re.compile(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def _video_id_of(target):
    """Extract an 11-char YouTube video id from an id or any watch/live/short URL."""
    s = str(target or "").strip()
    m = _VID_RE.search(s)
    if m:
        return m.group(1)
    return s if re.fullmatch(r"[A-Za-z0-9_-]{11}", s) else s


def _caption_cache_path(video_id, lang):
    return CAPTION_CACHE_DIR / f"{video_id}.{lang}.vtt"


def _cache_get_caption(video_id, candidates):
    """Return (vtt_text, lang) from the local store, or (None, None).

    Tries the requested candidate languages in order; otherwise falls back to any
    cached track for the video, preferring the original (*-orig)."""
    if candidates:
        for c in candidates:
            p = _caption_cache_path(video_id, c)
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace"), c
    found = sorted(CAPTION_CACHE_DIR.glob(f"{video_id}.*.vtt"))
    pick = [p for p in found if p.stem.endswith("-orig")] or found
    if pick:
        lang = pick[0].name[len(video_id) + 1:-4]  # strip '<id>.' and '.vtt'
        return pick[0].read_text(encoding="utf-8", errors="replace"), lang
    return None, None


def _cache_put_caption(video_id, lang, text):
    try:
        _caption_cache_path(video_id, lang or "orig").write_text(text, encoding="utf-8")
    except OSError:
        pass


def _load_resolutions():
    try:
        return json.loads(RESOLUTIONS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_resolution(db_stem, video_id, caption_lang):
    if not db_stem or not video_id:
        return
    res = _load_resolutions()
    res[db_stem] = {"video_id": video_id, "caption_lang": caption_lang}
    try:
        RESOLUTIONS_FILE.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    except OSError:
        pass


def _video_original_lang(yturl):
    """Return the video's original spoken-language code (e.g. 'ru'), or None."""
    cmd = [*YTDLP_CMD, "--no-warnings", "--skip-download", "--print", "%(language)s", yturl]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    out = proc.stdout.strip().splitlines()
    lang = out[0].strip() if out else ""
    return lang if lang and lang != "NA" else None


def _run_caption_download(yturl, video_id, request_langs):
    """Download the requested caption tracks; return (vtt_text, lang, error).

    request_langs may be exact codes (e.g. 'ru-orig') or yt-dlp patterns (e.g.
    '.*-orig' to match the original track regardless of language). The chosen track
    is cached by its actual language."""
    tmp_dir = Path(TRAINING_DATA_DIR) / "_yt_captions"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cap_{uuid.uuid4().hex}"
    out_tmpl = str(tmp_dir / f"{stem}.%(ext)s")
    cmd = [
        *YTDLP_CMD, "--no-warnings", "--skip-download", "--retries", "3",
        "--write-auto-sub", "--sub-langs", ",".join(request_langs),
        "--convert-subs", "vtt",
        "-o", out_tmpl, yturl,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return None, None, "yt-dlp is not installed (pip install yt-dlp)"
    except subprocess.TimeoutExpired:
        return None, None, "yt-dlp caption download timed out"

    def cleanup():
        for f in tmp_dir.glob(f"{stem}*"):
            try:
                f.unlink()
            except OSError:
                pass

    produced = list(tmp_dir.glob(f"{stem}*.vtt"))
    # Prefer an exact code match; else take whatever was produced (e.g. when a
    # pattern like '.*-orig' was requested), deriving the language from the filename.
    chosen, chosen_lang = None, None
    for c in request_langs:
        for p in produced:
            if p.name == f"{stem}.{c}.vtt":
                chosen, chosen_lang = p, c
                break
        if chosen:
            break
    if not chosen and produced:
        chosen = produced[0]
        parts = chosen.name.split(".")
        chosen_lang = parts[-2] if len(parts) >= 3 else "?"

    if not chosen:
        cleanup()
        blob = (proc.stderr or "") + (proc.stdout or "")
        if "429" in blob or "Too Many Requests" in blob:
            return None, None, "YouTube rate-limited this server (HTTP 429) — wait a few minutes and retry"
        return None, None, "no matching auto-captions produced"
    try:
        text = chosen.read_text(encoding="utf-8", errors="replace")
    finally:
        cleanup()
    _cache_put_caption(video_id, chosen_lang, text)  # store for reuse
    return text, chosen_lang, None


def _yt_download_captions(target, sub_lang, allow_fetch=True, force=False):
    """Get original-language auto-captions (WebVTT) for a video id/URL.

    Prefers the local caption store; only hits YouTube on a cache miss (unless
    force=True). For "auto" it grabs the ORIGINAL track directly via a '.*-orig'
    pattern — no fragile language probe — which matches the spoken audio (the right
    reference for training). Returns (vtt_text, used_lang, error).
    """
    yturl = target if str(target).startswith("http") else f"https://youtu.be/{target}"
    video_id = _video_id_of(target)

    sub_lang_norm = (sub_lang or "auto").strip()
    explicit = None
    if sub_lang_norm.lower() != "auto":
        explicit = [c.strip() for c in sub_lang_norm.split(",") if c.strip()]

    if not force:
        cached_text, cached_lang = _cache_get_caption(video_id, explicit)
        if cached_text is not None:
            return cached_text, cached_lang, None
    if not allow_fetch:
        return None, None, "no cached captions for this video (upload a bundle or enable fetching)"

    if explicit:
        return _run_caption_download(yturl, video_id, explicit)

    # auto: grab the original ASR track without a language probe
    text, lang, err = _run_caption_download(yturl, video_id, [".*-orig"])
    if text is not None:
        return text, lang, None
    if err and "429" in err:
        return None, None, err
    # Fallback: no *-orig track — probe the language and try it directly.
    lang_code = _video_original_lang(yturl)
    if lang_code:
        text, lang, err2 = _run_caption_download(
            yturl, video_id, [f"{lang_code}-orig", lang_code]
        )
        if text is not None:
            return text, lang, None
        if err2 and "429" in err2:
            return None, None, err2
    return None, None, "this video has no usable auto-captions (some services aren't captioned)"


def _youtube_cfg():
    yt = CONFIG.get("youtube", {}) if isinstance(CONFIG, dict) else {}
    tabs = yt.get("tabs") or ["streams", "videos"]
    return {
        "channel": yt.get("channel", ""),
        "sub_lang": yt.get("sub_lang", "auto"),
        "day_window": int(yt.get("day_window", 1)),
        "scan_limit": int(yt.get("scan_limit", 300)),
        "tabs": list(tabs),
        "coarse_days": int(yt.get("coarse_days", 30)),
    }


@app.route("/api/youtube/resolve", methods=["POST"])
def api_youtube_resolve():
    """Resolve which channel video matches a session DB (by date). No download."""
    data = request.get_json() or {}
    ycfg = _youtube_cfg()
    db_path = (data.get("db_path") or MAIN_APP_DB).strip()
    channel = (data.get("channel") or ycfg["channel"]).strip()
    if not os.path.exists(db_path):
        return jsonify({"error": f"DB not found: {db_path}"}), 404
    if not channel:
        return jsonify({"error": "no YouTube channel provided or configured"}), 400

    try:
        rows = _read_transcription_rows(db_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    session = ca.session_datetime(rows)
    if not session.get("date"):
        return jsonify({"error": "could not determine session date from DB"}), 400

    candidates, err = _yt_channel_videos(
        channel, session["date"], ycfg["day_window"], ycfg["scan_limit"],
        ycfg["tabs"], ycfg["coarse_days"]
    )
    if err:
        return jsonify({"error": err}), 502
    pick = ca.pick_video(candidates, session, day_window=ycfg["day_window"])
    return jsonify({"session": session, "pick": pick, "candidate_count": len(candidates)})


def _youtube_align(data):
    """Resolve video by date -> download captions -> align to DB rows -> filter.

    Does NOT extract audio or touch the manifest — it produces the labels for
    review. Returns (payload, error). error is (body, status) where body is a
    dict or message string. payload carries per-row `kept`/`dropped` labels plus
    the resolved video, offset, match report and detected wav_path.
    """
    ycfg = _youtube_cfg()
    db_path = (data.get("db_path") or MAIN_APP_DB).strip()
    channel = (data.get("channel") or ycfg["channel"]).strip()
    sub_lang = (data.get("sub_lang") or ycfg["sub_lang"]).strip()
    if not os.path.exists(db_path):
        return None, (f"DB not found: {db_path}", 404)

    try:
        rows = _read_transcription_rows(db_path)
    except Exception as e:
        return None, (str(e), 500)
    if not rows:
        return None, ("no labelable rows in DB", 400)
    session = ca.session_datetime(rows)
    db_stem = Path(db_path).stem
    refresh = bool(data.get("refresh"))
    use_cache = data.get("use_cache", True)

    # 1) pick the video (explicit override wins, then cached resolution, then YouTube)
    resolved = None
    video_target = (data.get("video_url") or "").strip()
    if not video_target and use_cache and not refresh:
        cached_res = _load_resolutions().get(db_stem)
        if cached_res and cached_res.get("video_id"):
            video_target = cached_res["video_id"]
            resolved = {**cached_res, "from_cache": True}
    if not video_target:
        if not channel:
            return None, ("no YouTube channel or video_url provided", 400)
        if not session.get("date"):
            return None, ("could not determine session date from DB", 400)
        candidates, err = _yt_channel_videos(
            channel, session["date"], ycfg["day_window"], ycfg["scan_limit"],
            ycfg["tabs"], ycfg["coarse_days"]
        )
        if err:
            return None, (err, 502)
        pick = ca.pick_video(candidates, session, day_window=ycfg["day_window"])
        if not pick.get("video_id"):
            return None, ({"error": "no channel video matched the session date",
                           "pick": pick, "session": session}, 404)
        video_target = pick["video_id"]
        resolved = pick

    # 2) get captions (local store first; YouTube only on a miss unless refresh)
    vtt_text, used_lang, err = _yt_download_captions(
        video_target, sub_lang, allow_fetch=True, force=refresh
    )
    if err:
        return None, (err, 502)
    _save_resolution(db_stem, _video_id_of(video_target), used_lang)

    # 3) align + label + filter
    cues = ca.parse_vtt(vtt_text)
    caption_words = ca.to_word_stream(cues)
    if not caption_words:
        return None, ("captions parsed but contained no words", 422)
    offset_val = data.get("offset")
    anchors = []
    if offset_val is not None:
        offset = float(offset_val)
        labels = ca.label_rows(rows, caption_words, offset)
    else:
        # Anchor-based piecewise alignment tracks clock drift over a long service;
        # fall back to a single global offset when too few anchors are found.
        anchors = ca.build_anchors(rows, caption_words)
        if len(anchors) >= 3:
            labels = ca.label_rows_anchored(rows, caption_words, anchors)
            mids = sorted(d - c for c, d in anchors)
            offset = round(mids[len(mids) // 2], 2)  # representative offset (median)
        else:
            offset = ca.estimate_offset(rows, caption_words)
            labels = ca.label_rows(rows, caption_words, offset)
    kept, dropped = ca.filter_labels(labels)
    report = ca.match_report(kept)

    drop_summary = {}
    for lb in dropped:
        drop_summary[lb.drop_reason] = drop_summary.get(lb.drop_reason, 0) + 1

    wav_path = (data.get("wav_path") or "").strip()
    if not wav_path:
        wavs = _find_companion_wav(db_path)
        wav_path = wavs[0] if wavs else ""

    payload = {
        "resolved_video": resolved,
        "video": video_target,
        "caption_lang": used_lang,
        "wav_path": wav_path,
        "offset": round(offset, 2),
        "anchors": len(anchors),
        "report": report,
        "_captions_vtt": vtt_text,   # private: consumed by the debug export
        "_db_path": db_path,
        "session": session,
        "kept": [lb.to_dict() for lb in kept],
        "dropped": [lb.to_dict() for lb in dropped],
        "drop_summary": drop_summary,
    }
    return payload, None


def _youtube_error_response(err):
    body, status = err
    return jsonify(body if isinstance(body, dict) else {"error": body}), status


def _public(payload):
    """Drop private underscore-prefixed keys (raw VTT, db path) before returning JSON."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


@app.route("/api/youtube/preview", methods=["POST"])
def api_youtube_preview():
    """Resolve + caption + align + filter, and return per-row labels for REVIEW.

    Nothing is written to the training set — the operator eyeballs the rows in the
    import table and then builds the dataset explicitly. Body:
    {db_path?, channel?, sub_lang?, wav_path?, video_url?, offset?}
    """
    payload, err = _youtube_align(request.get_json() or {})
    if err:
        return _youtube_error_response(err)
    if not payload["kept"]:
        return jsonify({**_public(payload),
                        "error": "no rows survived filtering — check channel/date/language"}), 422
    return jsonify(_public(payload))


@app.route("/api/youtube/build_dataset", methods=["POST"])
def api_youtube_build_dataset():
    """Fully-automatic one-shot (no review): align then extract clips into the
    manifest. The UI defaults to the preview+review flow instead; this remains for
    scripted use. Body same as /api/youtube/preview."""
    payload, err = _youtube_align(request.get_json() or {})
    if err:
        return _youtube_error_response(err)
    if not payload["kept"]:
        return jsonify({**_public(payload), "error": "no rows survived filtering"}), 422

    segments = [
        {"corrected_text": lb["corrected_text"],
         "start_time": lb["start_time"], "end_time": lb["end_time"]}
        for lb in payload["kept"]
    ]
    results, xerr = _extract_clips(payload["wav_path"], segments, source="youtube_autolabel")
    if xerr:
        return jsonify({**_public(payload), "error": f"clip extraction failed: {xerr[0]}"}), xerr[1]

    out = _public(payload)
    out["ok"] = True
    out["clips_saved"] = sum(1 for r in results if r.get("ok"))
    return jsonify(out)


@app.route("/api/youtube/export", methods=["POST"])
def api_youtube_export():
    """Build a debug bundle (zip) for review: the session DB, the raw captions, and
    the full alignment result (rows, labels, similarities, offset, report). Body same
    as /api/youtube/preview."""
    payload, err = _youtube_align(request.get_json() or {})
    if err:
        return _youtube_error_response(err)

    vtt = payload.get("_captions_vtt", "") or ""
    db_path = payload.get("_db_path", "") or ""
    lang = payload.get("caption_lang") or "orig"
    public = {**_public(payload), "db_name": Path(db_path).name}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"captions.{lang}.vtt", vtt)
        z.writestr("alignment.json", json.dumps(public, ensure_ascii=False, indent=2))
        if db_path and os.path.exists(db_path):
            z.write(db_path, arcname="session.db")
        z.writestr("README.txt",
                   "STEACH YouTube auto-label debug bundle\n"
                   "- session.db      : the source STT session database\n"
                   "- captions.*.vtt  : raw YouTube captions used as the reference\n"
                   "- alignment.json  : resolved video, time offset, per-row db_text vs\n"
                   "                    caption-derived corrected_text, similarity, and\n"
                   "                    drop reasons, plus the match report.\n")
    buf.seek(0)
    stem = Path(db_path).stem or "session"
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"steach_debug_{stem}.zip")


@app.route("/api/youtube/import_bundle", methods=["POST"])
def api_youtube_import_bundle():
    """Re-upload a debug bundle (or a raw .vtt) into the local caption store so
    future runs skip YouTube.

    - multipart 'bundle': a steach_debug_*.zip (captions.*.vtt + alignment.json).
      Captions are cached by the alignment's video id, and the session DB stem is
      mapped to that video so resolution is skipped too.
    - multipart 'captions' + form 'video_id' (+ optional 'lang', 'db_stem'): store a
      single caption file directly.
    """
    stored = []
    f = request.files.get("bundle")
    if f:
        try:
            with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                names = z.namelist()
                video_id, caption_lang, db_name = None, None, None
                if "alignment.json" in names:
                    meta = json.loads(z.read("alignment.json").decode("utf-8", "replace"))
                    video_id = _video_id_of(meta.get("video") or "")
                    caption_lang = meta.get("caption_lang")
                    db_name = meta.get("db_name")
                for n in names:
                    if n.startswith("captions.") and n.endswith(".vtt"):
                        file_lang = n[len("captions."):-len(".vtt")]
                        vtt = z.read(n).decode("utf-8", "replace")
                        lang = caption_lang or file_lang
                        if video_id:
                            _cache_put_caption(video_id, lang, vtt)
                            stored.append(f"{video_id}.{lang}")
                if video_id and db_name:
                    _save_resolution(Path(db_name).stem, video_id, caption_lang)
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
            return jsonify({"error": f"invalid bundle: {e}"}), 400
        if not stored:
            return jsonify({"error": "bundle had no captions/alignment to import"}), 400
        return jsonify({"ok": True, "cached": stored})

    cap = request.files.get("captions")
    if cap:
        video_id = _video_id_of(request.form.get("video_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
            return jsonify({"error": "captions upload requires a valid 'video_id'"}), 400
        lang = (request.form.get("lang") or "orig").strip()
        _cache_put_caption(video_id, lang, cap.read().decode("utf-8", "replace"))
        db_stem = (request.form.get("db_stem") or "").strip()
        if db_stem:
            _save_resolution(db_stem, video_id, lang)
        return jsonify({"ok": True, "cached": [f"{video_id}.{lang}"]})

    return jsonify({"error": "no file uploaded (field 'bundle' or 'captions')"}), 400


@app.route("/api/youtube/cache")
def api_youtube_cache():
    """List the locally cached captions and session->video resolutions."""
    captions = []
    for p in sorted(CAPTION_CACHE_DIR.glob("*.vtt")):
        stem = p.stem  # <video_id>.<lang>
        vid, _, lang = stem.partition(".")
        captions.append({"video_id": vid, "lang": lang, "bytes": p.stat().st_size})
    return jsonify({"captions": captions, "resolutions": _load_resolutions()})


# --- Training ---

@app.route("/api/start_training", methods=["POST"])
def api_start_training():
    global _training_proc
    with _training_lock:
        if _training_proc and _training_proc.poll() is None:
            return jsonify({"error": "training already running"}), 409

        data = request.get_json() or {}
        model_type = data.get("model_type", "stt")

        if model_type == "translation":
            cfg = CONFIG["translation"]
            entries = read_manifest(TRANSLATION_MANIFEST)
            valid = [e for e in entries if e.get("source") and e.get("target")]
            min_samples = cfg.get("min_samples", 10)
            if len(valid) < min_samples:
                return jsonify({"error": f"need at least {min_samples} translation pairs (have {len(valid)})"}), 400
            base_model = data.get("base_model", cfg["base_model"])
            epochs = int(data.get("epochs", cfg["epochs"]))
            lr = float(data.get("learning_rate", cfg["learning_rate"]))
            batch_size = int(data.get("batch_size", cfg["batch_size"]))
            lora_rank = int(data.get("lora_rank", cfg["lora_rank"]))
            cmd = [
                sys.executable, "finetune_whisper.py",
                "--model_type", "translation",
                "--base_model", base_model,
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--batch_size", str(batch_size),
                "--lora_rank", str(lora_rank),
                "--manifest", str(TRANSLATION_MANIFEST),
                "--output_dir", str(MODELS_OUTPUT_DIR),
            ]
        else:
            cfg = CONFIG["training"]
            entries = read_manifest(STT_MANIFEST)
            valid = [e for e in entries if e.get("audio") and e.get("text")]
            min_samples = cfg.get("min_samples", 10)
            if len(valid) < min_samples:
                return jsonify({"error": f"need at least {min_samples} samples (have {len(valid)})"}), 400
            base_model = data.get("base_model", cfg["base_model"])
            epochs = int(data.get("epochs", cfg["epochs"]))
            lr = float(data.get("learning_rate", cfg["learning_rate"]))
            batch_size = int(data.get("batch_size", cfg["batch_size"]))
            lora_rank = int(data.get("lora_rank", cfg["lora_rank"]))
            cmd = [
                sys.executable, "finetune_whisper.py",
                "--model_type", "stt",
                "--base_model", base_model,
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--batch_size", str(batch_size),
                "--lora_rank", str(lora_rank),
                "--manifest", str(STT_MANIFEST),
                "--audio_dir", str(STT_AUDIO_DIR),
                "--output_dir", str(MODELS_OUTPUT_DIR),
            ]

        _training_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    threading.Thread(target=_stream_training_output, daemon=True).start()
    return jsonify({"ok": True})


def _discover_nllb_models():
    models_parent = MODELS_OUTPUT_DIR
    found = []
    if models_parent.exists():
        for p in sorted(models_parent.iterdir()):
            if p.is_dir() and p.name.startswith("facebook--nllb"):
                found.append({"name": p.name, "path": str(p)})
    if not found:
        fallback = CONFIG["translation"]["base_model"]
        found.append({"name": Path(fallback).name, "path": fallback})
    return found


@app.route("/api/config")
def api_config():
    warnings = []
    if not os.access(MODELS_OUTPUT_DIR, os.W_OK):
        warnings.append(
            f"Models directory is not writable by this user. "
            f"Run: sudo chown -R {os.getenv('USER', 'ai')} {MODELS_OUTPUT_DIR}"
        )
    state = load_state()
    return jsonify({
        "main_app_db": MAIN_APP_DB,
        "main_app_audio_backup": MAIN_APP_AUDIO_BACKUP,
        "nllb_models": _discover_nllb_models(),
        "youtube": _youtube_cfg(),
        "remember": {
            "channel": state.get("channel") or CONFIG.get("youtube", {}).get("channel", ""),
            "db_path": state.get("db_path", ""),
        },
        "warnings": warnings,
    })


@app.route("/api/remember", methods=["POST"])
def api_remember():
    """Persist remembered UI state (channel, last DB path) to ui_state.json."""
    data = request.get_json() or {}
    state = load_state()
    if "channel" in data:
        state["channel"] = (data.get("channel") or "").strip()
    if "db_path" in data:
        state["db_path"] = (data.get("db_path") or "").strip()
    try:
        save_state(state)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "remember": state})


@app.route("/api/browse")
def api_browse():
    raw = request.args.get("path", "/home")
    try:
        p = Path(raw).resolve()
    except Exception:
        return jsonify({"error": "invalid path"}), 400
    if not p.exists():
        p = p.parent if p.parent.exists() else Path("/home")
    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "is_file": child.is_file(),
                })
            except PermissionError:
                pass
    except PermissionError:
        return jsonify({"error": "permission denied"}), 403
    parent = str(p.parent) if p.parent != p else str(p)
    return jsonify({"path": str(p), "parent": parent, "entries": entries})


def _stream_training_output():
    global _training_proc
    proc = _training_proc
    for line in proc.stdout:
        line = line.rstrip()
        socketio.emit("training_log", {"line": line})
        try:
            data = json.loads(line)
            if "epoch" in data:
                socketio.emit("training_progress", data)
                if data.get("done"):
                    socketio.emit("training_done", data)
        except (json.JSONDecodeError, TypeError):
            pass
    proc.wait()
    socketio.emit("training_log", {"line": f"[DONE] exit code {proc.returncode}"})
    if proc.returncode == 0:
        socketio.emit("training_done", {"exit_code": 0})


@app.route("/api/stop_training", methods=["POST"])
def api_stop_training():
    global _training_proc
    with _training_lock:
        if not _training_proc or _training_proc.poll() is not None:
            return jsonify({"error": "no training running"}), 409
        _training_proc.send_signal(signal.SIGTERM)
    return jsonify({"ok": True})


@app.route("/api/training_status")
def api_training_status():
    global _training_proc
    running = _training_proc is not None and _training_proc.poll() is None
    return jsonify({"running": running})


# --- Model conversion and listing ---

@app.route("/api/convert_model", methods=["POST"])
def api_convert_model():
    data = request.get_json()
    model_path = data.get("model_path", "").strip()
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "model_path not found"}), 400

    out_path = model_path.rstrip("/") + "_ct2"

    def _run():
        cmd = [
            "ct2-whisper-converter",
            "--model", model_path,
            "--output_dir", out_path,
            "--quantization", "float16",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            socketio.emit("training_log", {"line": line.rstrip()})
        proc.wait()
        if proc.returncode == 0:
            socketio.emit("convert_done", {"output": out_path})
        else:
            socketio.emit("training_log", {"line": f"[ERROR] Conversion failed (exit {proc.returncode})"})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "output": out_path})


@app.route("/api/models")
def api_models():
    models = []
    if not MODELS_OUTPUT_DIR.exists():
        return jsonify({"models": []})

    for p in sorted(MODELS_OUTPUT_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        meta_file = p / "training_meta.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                pass
        is_ct2 = (p / "model.bin").exists() or (p / "model.bin.gz").exists()
        is_lora = (p / "adapter_model.bin").exists() or (p / "adapter_model.safetensors").exists()
        if not (is_ct2 or is_lora):
            continue
        models.append({
            "name": p.name,
            "path": str(p),
            "format": "CTranslate2" if is_ct2 else "LoRA",
            "label": "Fine-tuned",
            "model_type": meta.get("model_type", "stt"),
            "base_model": meta.get("base_model", ""),
            "pair_count": meta.get("pair_count", 0),
            "wer": meta.get("wer"),
            "bleu": meta.get("bleu"),
            "created_at": meta.get("created_at", ""),
        })

    return jsonify({"models": models})


@app.route("/training_data/stt/audio/<path:filename>")
def serve_audio(filename):
    from flask import send_from_directory
    return send_from_directory(str(STT_AUDIO_DIR), filename)


@app.route("/api/preview_audio")
def api_preview_audio():
    """Stream a sliced, resampled WAV segment for in-browser playback.
    Query params: wav (path), start (seconds), end (seconds)
    """
    import io
    from flask import Response
    wav_path = request.args.get("wav", "").strip()
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
    except ValueError:
        return jsonify({"error": "invalid timestamps"}), 400

    if not wav_path or not os.path.exists(wav_path):
        return jsonify({"error": "wav not found"}), 404
    if end <= start:
        return jsonify({"error": "invalid range"}), 400

    try:
        from pydub import AudioSegment as PydubAudio
        audio = PydubAudio.from_file(wav_path)
        clip = audio[int(start * 1000):int(end * 1000)]
        clip = clip.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        buf = io.BytesIO()
        clip.export(buf, format="wav")
        buf.seek(0)
        return Response(buf.read(), mimetype="audio/wav",
                        headers={"Content-Disposition": "inline"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<path:model_name>", methods=["DELETE"])
def api_model_delete(model_name):
    import shutil
    target = MODELS_OUTPUT_DIR / model_name
    if not target.exists() or not target.is_dir():
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(target)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = CONFIG["server"]["host"]
    port = CONFIG["server"]["port"]
    print(f"[STEACH] Training server starting on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
