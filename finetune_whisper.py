"""
LoRA fine-tuning for Whisper (STT) or NLLB (translation) using PEFT.

STT:  audio clips + correct transcription text → Whisper LoRA adapter + CTranslate2 model
Translation: source transcription text + correct translation → NLLB LoRA adapter (no conversion needed)

Progress lines emitted to stdout as JSON:
  {"epoch": 1, "total_epochs": 3, "loss": 0.42, "wer": 0.18, "gpu_mb": 8400, "done": false}
Final line has "done": true.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="stt", choices=["stt", "translation"])
    p.add_argument("--base_model", default="openai/whisper-medium")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio_dir", default="")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def gpu_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024 ** 2)
    return 0


def load_manifest(manifest_path: str, audio_dir: str):
    pairs = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio = e.get("audio", "")
            text = e.get("text", "").strip()
            if not audio or not text:
                continue
            audio_path = os.path.join(audio_dir, audio)
            if not os.path.exists(audio_path):
                continue
            pairs.append({"audio": audio_path, "text": text})
    return pairs


def load_translation_pairs(manifest_path: str):
    pairs = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = e.get("source", "").strip()
            target = e.get("target", "").strip()
            if not source or not target:
                continue
            pairs.append({
                "source": source,
                "target": target,
                "source_lang": e.get("source_lang", "rus_Cyrl"),
                "target_lang": e.get("target_lang", "eng_Latn"),
            })
    return pairs


def train_translation(args):
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import evaluate

    emit({"log": f"Loading NLLB model from {args.base_model} ..."})
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    emit({"log": "Loading translation pairs ..."})
    pairs = load_translation_pairs(args.manifest)
    emit({"log": f"Found {len(pairs)} translation pairs"})
    if not pairs:
        emit({"log": "[ERROR] No valid pairs found.", "done": True, "error": True})
        sys.exit(1)

    class TranslationDataset(torch.utils.data.Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            p = self.pairs[idx]
            tokenizer.src_lang = p["source_lang"]
            inputs = tokenizer(p["source"], max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(p["target"], max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            label_ids = labels["input_ids"][0]
            label_ids[label_ids == tokenizer.pad_token_id] = -100
            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": label_ids,
            }

    dataset = TranslationDataset(pairs)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        label_str = [[s] for s in label_str]
        result = bleu_metric.compute(predictions=pred_str, references=label_str)
        return {"bleu": round(result["score"], 2)}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    lora_output = os.path.join(args.output_dir, f"nllb-lora-{timestamp}")
    os.makedirs(lora_output, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=lora_output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        predict_with_generate=True,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        load_best_model_at_end=False,
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    class EmitCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            epoch = state.epoch
            if loss is not None and epoch is not None:
                emit({
                    "epoch": round(epoch, 2),
                    "total_epochs": args.num_train_epochs,
                    "loss": round(float(loss), 4),
                    "gpu_mb": gpu_mb(),
                    "done": False,
                })

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EmitCallback()],
    )

    emit({"log": "Starting NLLB translation training ..."})
    trainer.train()

    emit({"log": f"Saving LoRA adapter to {lora_output} ..."})
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)

    meta = {
        "model_type": "translation",
        "base_model": args.base_model,
        "pair_count": len(pairs),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(lora_output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    emit({"epoch": args.epochs, "total_epochs": args.epochs, "loss": 0.0, "gpu_mb": gpu_mb(), "done": True})


def main():
    args = parse_args()

    if args.model_type == "translation":
        train_translation(args)
        return

    emit({"log": f"Loading model {args.base_model} ..."})

    from transformers import (
        WhisperForConditionalGeneration,
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import evaluate
    import soundfile as sf

    # --- Load model & processor ---
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    tokenizer = WhisperTokenizer.from_pretrained(args.base_model, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.base_model, task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # --- LoRA ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    emit({"log": "Loading training data ..."})
    pairs = load_manifest(args.manifest, args.audio_dir)
    emit({"log": f"Found {len(pairs)} training pairs"})

    if not pairs:
        emit({"log": "[ERROR] No valid training pairs found.", "done": True, "error": True})
        sys.exit(1)

    class WhisperDataset(torch.utils.data.Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            p = self.pairs[idx]
            audio, sr = sf.read(p["audio"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features[0]
            labels = tokenizer(p["text"], return_tensors="pt").input_ids[0]
            return {"input_features": input_features, "labels": labels}

    from transformers import DataCollatorForSeq2Seq
    import dataclasses

    class WhisperDataCollator:
        def __call__(self, features):
            input_features = torch.stack([f["input_features"] for f in features])
            label_list = [f["labels"] for f in features]
            max_len = max(l.size(0) for l in label_list)
            padded_labels = torch.full((len(label_list), max_len), -100, dtype=torch.long)
            for i, l in enumerate(label_list):
                padded_labels[i, : l.size(0)] = l
            return {"input_features": input_features, "labels": padded_labels}

    dataset = WhisperDataset(pairs)
    collator = WhisperDataCollator()

    # --- WER metric ---
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    # --- Timestamp for output ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    lora_output = os.path.join(args.output_dir, f"whisper-lora-{timestamp}")
    os.makedirs(lora_output, exist_ok=True)

    # --- Training args ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=lora_output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        load_best_model_at_end=False,
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    class ProgressCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            epoch = logs.get("epoch")
            if loss is not None and epoch is not None:
                emit({
                    "epoch": round(epoch, 2),
                    "total_epochs": args.num_train_epochs,
                    "loss": round(float(loss), 4),
                    "gpu_mb": gpu_mb(),
                    "done": False,
                })

    from transformers import TrainerCallback

    class EmitCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            epoch = state.epoch
            if loss is not None and epoch is not None:
                emit({
                    "epoch": round(epoch, 2),
                    "total_epochs": args.num_train_epochs,
                    "loss": round(float(loss), 4),
                    "gpu_mb": gpu_mb(),
                    "done": False,
                })

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EmitCallback()],
    )

    emit({"log": "Starting training ..."})
    trainer.train()

    # --- Save LoRA adapter ---
    emit({"log": f"Saving LoRA adapter to {lora_output} ..."})
    model.save_pretrained(lora_output)
    processor.save_pretrained(lora_output)

    # --- Save metadata ---
    meta = {
        "model_type": "stt",
        "base_model": args.base_model,
        "pair_count": len(pairs),
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(lora_output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Merge LoRA + convert to CTranslate2 ---
    emit({"log": "Merging LoRA weights and converting to CTranslate2 ..."})
    try:
        merged_model = model.merge_and_unload()
        merged_path = lora_output + "_merged"
        os.makedirs(merged_path, exist_ok=True)
        merged_model.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)

        ct2_path = os.path.join(args.output_dir, f"faster-whisper-finetuned-{timestamp}")
        import subprocess
        result = subprocess.run(
            [
                "ct2-whisper-converter",
                "--model", merged_path,
                "--output_dir", ct2_path,
                "--quantization", "float16",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Save metadata in CTranslate2 output too
            with open(os.path.join(ct2_path, "training_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            emit({"log": f"CTranslate2 model saved to {ct2_path}"})
        else:
            emit({"log": f"[WARNING] ct2 conversion failed: {result.stderr[:300]}"})
    except Exception as e:
        emit({"log": f"[WARNING] Merge/convert failed: {e}"})

    # --- Save training history ---
    history_path = Path(args.manifest).parent / "training_history.jsonl"
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception:
        pass

    emit({"epoch": args.epochs, "total_epochs": args.epochs, "loss": 0.0, "gpu_mb": gpu_mb(), "done": True})


if __name__ == "__main__":
    main()
