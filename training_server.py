import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import soundfile as sf
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    with open("config.json") as f:
        return json.load(f)

CONFIG = load_config()
TRAINING_DATA_DIR = Path(CONFIG["paths"]["training_data_dir"])
STT_DIR = TRAINING_DATA_DIR / "stt"
STT_AUDIO_DIR = STT_DIR / "audio"
STT_MANIFEST = STT_DIR / "manifest.jsonl"
MODELS_OUTPUT_DIR = Path(CONFIG["paths"]["models_output_dir"])
MAIN_APP_DB = CONFIG["paths"]["main_app_db"]
MAIN_APP_AUDIO_BACKUP = CONFIG["paths"]["main_app_audio_backup"]

STT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not STT_MANIFEST.exists():
    STT_MANIFEST.write_text("")

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

def read_manifest():
    entries = []
    if not STT_MANIFEST.exists():
        return entries
    with open(STT_MANIFEST) as f:
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


def write_manifest(entries):
    with open(STT_MANIFEST, "w") as f:
        for e in entries:
            row = {k: v for k, v in e.items() if k != "id"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_manifest(entry):
    row = {k: v for k, v in entry.items() if k != "id"}
    with open(STT_MANIFEST, "a") as f:
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

@app.route("/api/import/transcriptions")
def api_import_list():
    db_path = request.args.get("db_path", MAIN_APP_DB)
    if not os.path.exists(db_path):
        return jsonify({"error": f"DB not found: {db_path}"}), 404
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, text, start_time, end_time FROM transcriptions "
                "WHERE text IS NOT NULL AND trim(text) != '' ORDER BY id DESC LIMIT 500"
            ).fetchall()
        return jsonify({"transcriptions": [dict(r) for r in rows]})
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


# --- Training ---

@app.route("/api/start_training", methods=["POST"])
def api_start_training():
    global _training_proc
    with _training_lock:
        if _training_proc and _training_proc.poll() is None:
            return jsonify({"error": "training already running"}), 409

        entries = read_manifest()
        valid = [e for e in entries if e.get("audio") and e.get("text")]
        min_samples = CONFIG["training"].get("min_samples", 10)
        if len(valid) < min_samples:
            return jsonify({"error": f"need at least {min_samples} samples (have {len(valid)})"}), 400

        data = request.get_json() or {}
        base_model = data.get("base_model", CONFIG["training"]["base_model"])
        epochs = int(data.get("epochs", CONFIG["training"]["epochs"]))
        lr = float(data.get("learning_rate", CONFIG["training"]["learning_rate"]))
        batch_size = int(data.get("batch_size", CONFIG["training"]["batch_size"]))
        lora_rank = int(data.get("lora_rank", CONFIG["training"]["lora_rank"]))

        cmd = [
            sys.executable, "finetune_whisper.py",
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
            "base_model": meta.get("base_model", ""),
            "pair_count": meta.get("pair_count", 0),
            "wer": meta.get("wer"),
            "created_at": meta.get("created_at", ""),
        })

    return jsonify({"models": models})


@app.route("/training_data/stt/audio/<path:filename>")
def serve_audio(filename):
    from flask import send_from_directory
    return send_from_directory(str(STT_AUDIO_DIR), filename)


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
