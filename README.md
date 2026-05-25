# STEACH — Whisper & NLLB Fine-Tuning

A web UI for fine-tuning speech recognition (Whisper) and translation (NLLB) models using LoRA, sourced from STT app session backups.

## What it does

- **STT training** — upload audio files with transcriptions, or import directly from STT app session databases with audio extraction. Fine-tunes a Whisper model via LoRA and converts to CTranslate2.
- **Translation training** — import transcriptions from session databases, pre-filled with existing machine translations for correction. Fine-tunes an NLLB model via LoRA.

## Requirements

- Python 3.10+
- CUDA GPU (fp16 training)
- STT app backup databases at `main_app_audio_backup` path (optional)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

**1. Edit `config.json`** to match your paths:

```json
{
  "paths": {
    "training_data_dir": "training_data/",
    "models_output_dir": "/home/ai/STT/models/",
    "main_app_db": "/home/ai/STT/transcriptions.db",
    "main_app_audio_backup": "/home/ai/STT/_AUTOMATIC_BACKUP/"
  }
}
```

**2. Fix model directory permissions** if the models directory is owned by root:

```bash
sudo chown -R $USER /home/ai/STT/models/
```

The app will warn you at startup if this is needed.

**3. Run the server:**

```bash
python training_server.py
```

Open [http://localhost:5001](http://localhost:5001).

## Workflow

### 1 — Add Training Data

**STT — Audio + Transcription**
Drop audio files (WAV/MP3/M4A/FLAC/OGG/OPUS) and enter the correct transcription for each. Click *Add All to Dataset*.

**Import from STT DB**
Browse to a session `.db` file from the STT app backup. Set optional start/end row numbers to load a slice. Click *Load* — the companion WAV is auto-detected.

- **STT Training tab** — rows show timestamps, editable transcriptions, and a per-row *Add* button. Audio is extracted automatically if a WAV is present.
- **Translation Training tab** — rows show the transcription pre-filled with the existing machine translation. Edit as needed, then click *Add* per row or *Add All Translation Pairs* to bulk-save.

### 2 — Dataset

Review all collected STT pairs (audio + transcription) and translation pairs (source + target text). Edit or delete individual entries.

### 3 — Train

Select model type (**STT / Whisper** or **Translation / NLLB**), configure hyperparameters, and click *Start Training*. Training progress streams live to the log panel.

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `training.base_model` | `openai/whisper-medium` | Whisper variant to fine-tune |
| `translation.base_model` | `/home/ai/STT/models/facebook--nllb-200-distilled-600M` | NLLB model path |
| `training.lora_rank` | `16` | LoRA rank (higher = more parameters) |
| `training.epochs` | `3` | Training epochs |
| `training.batch_size` | `4` | Per-step batch size |
| `training.min_samples` | `10` | Minimum pairs required to start training |

## Training data storage

```
training_data/
  stt/
    manifest.jsonl      # audio + transcription pairs
    audio/              # extracted 16kHz mono WAV clips
  translation/
    manifest.jsonl      # source + target text pairs
```
