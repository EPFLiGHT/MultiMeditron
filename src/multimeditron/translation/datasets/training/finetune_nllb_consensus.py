"""
NLLB-200-3.3B Fine-tuning - Multi-GPU with Robust Checkpointing
Using consensus translations from Qwen2.5/TranslateGemma, M2M100, MADLAD-400

Features:
- Automatic checkpoint resume after preemption
- Frequent saves (every 50 steps)
- Graceful signal handling (SIGTERM)
- 4-GPU optimized settings
"""

import os
import sys
import json
import signal
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
import evaluate

# Flush output immediately for real-time logging
sys.stdout.reconfigure(line_buffering=True)

def log(msg: str):
    """Print with timestamp and flush."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("="*70)
log("NLLB Fine-tuning - 4-GPU with Robust Checkpointing")
log("Languages: Amharic, Swahili, Tamil, Hausa, Yoruba, Zulu")
log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*70)

# =============================================================================
# CONFIGURATION - 4-GPU OPTIMIZED WITH CHECKPOINTING
# =============================================================================

DATA_DIR = "/mloscratch/users/grieder/MeditronPolyglot/MultiMeditron/src/multimeditron/translation/datasets/generated_datasets/consensus"
OUTPUT_DIR = "/mloscratch/users/grieder/MeditronPolyglot/MultiMeditron/src/multimeditron/translation/models/nllb-consensus-finetuned"
CACHE_DIR = "/mloscratch/users/grieder/MeditronPolyglot/MultiMeditron/src/multimeditron/translation/models/.cache"
MODEL_NAME = "facebook/nllb-200-3.3B"

LANGUAGE_MAPPING = {
    "amh": "amh_Ethi",
    "swa": "swh_Latn",
    "tam": "tam_Taml",
    "hau": "hau_Latn",
    "yor": "yor_Latn",
    "zul": "zul_Latn",
}

# Multi-GPU optimized settings
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
log(f"\n🖥️  Detected {num_gpus} GPU(s)")

# 4x H100 80GB optimized settings
BATCH_SIZE = 4  # per GPU
GRADIENT_ACCUMULATION = 4  # Reduced since we have 4 GPUs

NUM_EPOCHS = 1
LEARNING_RATE = 1e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256

SAMPLES_PER_LANGUAGE_ENV = os.environ.get("SAMPLES_PER_LANGUAGE")
if SAMPLES_PER_LANGUAGE_ENV:
    try:
        SAMPLES_PER_LANGUAGE = int(SAMPLES_PER_LANGUAGE_ENV)
    except ValueError as err:
        raise ValueError(
            f"Invalid SAMPLES_PER_LANGUAGE='{SAMPLES_PER_LANGUAGE_ENV}'. Expected integer."
        ) from err
else:
    SAMPLES_PER_LANGUAGE = None  # Use all data

# CHECKPOINTING - More frequent for preemption resilience
SAVE_EVERY = 100  # Save every 100 steps (must divide EVAL_STEPS evenly)
LOG_EVERY = 10    # Log more frequently
EVAL_STEPS = 100  # Evaluate every 200 steps

effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION * num_gpus

log(f"\n📊 Configuration:")
log(f"  GPUs: {num_gpus}")
log(f"  Batch per GPU: {BATCH_SIZE}")
log(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
log(f"  Effective batch size: {effective_batch}")
log(f"  Languages: {len(LANGUAGE_MAPPING)}")
log(f"  Epochs: {NUM_EPOCHS}")
log(f"  Learning rate: {LEARNING_RATE}")
log(f"  Checkpoint every: {SAVE_EVERY} steps")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# CHECKPOINT DETECTION
# =============================================================================

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in output directory."""
    output_path = Path(output_dir)
    checkpoints = list(output_path.glob("checkpoint-*"))
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints_with_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.name.split("-")[1])
            checkpoints_with_steps.append((ckpt, step))
        except (IndexError, ValueError):
            continue
    
    if not checkpoints_with_steps:
        return None
    
    latest = max(checkpoints_with_steps, key=lambda x: x[1])
    return str(latest[0])

# Check for existing checkpoint
resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
if resume_checkpoint:
    log(f"\n🔄 FOUND CHECKPOINT: {resume_checkpoint}")
    log(f"   Training will RESUME from this checkpoint")
else:
    log(f"\n🆕 No checkpoint found - starting fresh training")

# =============================================================================
# SIGNAL HANDLING FOR GRACEFUL SHUTDOWN
# =============================================================================

class GracefulShutdownCallback(TrainerCallback):
    """Callback to handle graceful shutdown on SIGTERM."""
    
    def __init__(self):
        self.should_stop = False
        self.trainer = None
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        log(f"\n⚠️  Received {sig_name} - initiating graceful shutdown...")
        self.should_stop = True
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.should_stop:
            log(f"🛑 Stopping training at step {state.global_step}")
            control.should_save = True  # Force save
            control.should_training_stop = True
        return control
    
    def on_save(self, args, state, control, **kwargs):
        log(f"💾 Checkpoint saved at step {state.global_step}")
        return control

shutdown_callback = GracefulShutdownCallback()

# =============================================================================
# SAVE CONFIG
# =============================================================================

config_path = os.path.join(OUTPUT_DIR, "training_config.json")
config_data = {
    "version": "consensus_4gpu_checkpoint",
    "num_gpus": num_gpus,
    "batch_per_gpu": BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "effective_batch_size": effective_batch,
    "languages": list(LANGUAGE_MAPPING.keys()),
    "samples_per_language": SAMPLES_PER_LANGUAGE,
    "start_time": datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "checkpoint_steps": SAVE_EVERY,
    "resume_from": resume_checkpoint,
    "data_source": "consensus_translations",
}

# Only write config if starting fresh (don't overwrite on resume)
if not resume_checkpoint:
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

# =============================================================================
# LOAD OR CREATE TOKENIZED DATASETS (with caching)
# =============================================================================

tokenized_train_path = os.path.join(CACHE_DIR, "tokenized_train")
tokenized_eval_path = os.path.join(CACHE_DIR, "tokenized_eval")

# Load tokenizer first (needed for both paths)
log(f"\n🔤 Loading tokenizer...")
tokenizer = NllbTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang="eng_Latn",
    cache_dir=os.environ.get("HF_HOME", None)
)
log(f"✓ Tokenizer loaded")

if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_eval_path):
    # Load cached tokenized datasets
    log(f"\n📂 Loading CACHED tokenized datasets...")
    tokenized_train = load_from_disk(tokenized_train_path)
    tokenized_eval = load_from_disk(tokenized_eval_path)
    log(f"✓ Loaded from cache: {len(tokenized_train):,} train, {len(tokenized_eval):,} eval")
    
    # Load language counts from config if available
    try:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            language_counts = saved_config.get('language_counts', {})
            skipped_counts = saved_config.get('skipped_counts', {})
    except Exception as config_err:
        log(f"⚠️  Failed loading cached language counts: {config_err}")
        language_counts = {}
        skipped_counts = {}
else:
    # Create datasets from scratch
    log(f"\n📂 Loading consensus translation datasets...")
    
    def load_jsonl(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    all_data = []
    language_counts = {}
    skipped_counts = {}
    
    for lang_code, nllb_code in LANGUAGE_MAPPING.items():
        filepath = os.path.join(DATA_DIR, f"{lang_code}_train.jsonl")
        
        if not os.path.exists(filepath):
            log(f"⚠️  Warning: {filepath} not found, skipping {lang_code}")
            continue
        
        log(f"  Loading {lang_code}...")
        lang_data = load_jsonl(filepath)
        
        if SAMPLES_PER_LANGUAGE is not None:
            lang_data = lang_data[:SAMPLES_PER_LANGUAGE]
            log(f"    Sampled {len(lang_data)} entries")
        
        skipped = 0
        for entry in lang_data:
            eng_text = entry.get('text', entry.get('content', ''))
            translation = entry.get('translated_text', '')
            
            if not eng_text or not translation:
                skipped += 1
                continue
            
            status = entry.get('translation_status', '')
            if status != 'success':
                skipped += 1
                continue
            
            all_data.append({
                'english': eng_text,
                'translation': translation,
                'target_lang': nllb_code,
                'source_lang_code': lang_code,
                'translation_model': entry.get('translation_model', ''),
                'consensus_scores': entry.get('consensus_scores', {}),
            })
        
        language_counts[lang_code] = len([d for d in all_data if d['source_lang_code'] == lang_code])
        skipped_counts[lang_code] = skipped
        log(f"    ✓ {language_counts[lang_code]} examples (skipped {skipped} low-quality)")
    
    log(f"\n✓ Total examples loaded: {len(all_data):,}")
    log(f"\n📊 Language distribution:")
    for lang, count in language_counts.items():
        pct = count / len(all_data) * 100
        log(f"  {lang}: {count:,} ({pct:.1f}%)")
    
    # Update config with language counts
    config_data['language_counts'] = language_counts
    config_data['skipped_counts'] = skipped_counts
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # =============================================================================
    # TRAIN/VAL SPLIT
    # =============================================================================
    
    log(f"\n✂️  Creating train/validation split...")
    
    dataset = Dataset.from_list(all_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    log(f"  Train: {len(train_dataset):,}")
    log(f"  Validation: {len(eval_dataset):,}")
    
    # =============================================================================
    # PREPROCESSING - BIDIRECTIONAL
    # =============================================================================
    
    log(f"\n⚙️  Preprocessing (bidirectional training)...")
    
    def preprocess_function(examples):
        """Creates BOTH EN→X and X→EN examples."""
        sources = []
        targets = []
        src_langs = []
        tgt_langs = []
        
        for eng, trans, tgt_lang in zip(
            examples['english'],
            examples['translation'],
            examples['target_lang']
        ):
            if not eng or not trans:
                continue
            
            # EN → Target
            sources.append(eng)
            targets.append(trans)
            src_langs.append('eng_Latn')
            tgt_langs.append(tgt_lang)
            
            # Target → EN
            sources.append(trans)
            targets.append(eng)
            src_langs.append(tgt_lang)
            tgt_langs.append('eng_Latn')
        
        model_inputs = tokenizer(
            sources,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        
        labels_list = []
        for target, tgt_lang in zip(targets, tgt_langs):
            tokenizer.src_lang = tgt_lang
            labels = tokenizer(
                target,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length"
            )
            labels_list.append(labels["input_ids"])
        
        model_inputs["labels"] = labels_list
        return model_inputs
    
    # Use multiple workers for preprocessing
    num_proc = min(num_gpus * 4, 16)
    
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        num_proc=num_proc
    )
    
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation",
        num_proc=num_proc
    )
    
    # Save tokenized datasets for future resume
    log(f"\n💾 Caching tokenized datasets...")
    tokenized_train.save_to_disk(tokenized_train_path)
    tokenized_eval.save_to_disk(tokenized_eval_path)
    log(f"✓ Cached to {CACHE_DIR}")

log(f"✓ Train: {len(tokenized_train):,} examples (bidirectional)")
log(f"✓ Validation: {len(tokenized_eval):,} examples (bidirectional)")

# =============================================================================
# MODEL
# =============================================================================

log(f"\n🤖 Loading model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=os.environ.get("HF_HOME", None)
)

num_params = sum(p.numel() for p in model.parameters())
log(f"✓ Model loaded: {num_params/1e9:.2f}B parameters")

# =============================================================================
# METRICS
# =============================================================================

log(f"\n📏 Loading evaluation metrics...")
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    """Compute BLEU and chrF scores"""
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    vocab_size = len(tokenizer)
    preds = np.clip(preds, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)
    
    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        log(f"⚠️  Decoding error: {e}")
        return {"bleu": 0.0, "chrf": 0.0}
    
    valid_pairs = [
        (pred, ref) for pred, ref in zip(decoded_preds, decoded_labels)
        if pred.strip() and ref.strip()
    ]
    
    if not valid_pairs:
        return {"bleu": 0.0, "chrf": 0.0}
    
    decoded_preds, decoded_labels = zip(*valid_pairs)
    
    try:
        bleu_result = bleu_metric.compute(
            predictions=list(decoded_preds),
            references=[[label] for label in decoded_labels]
        )
        
        chrf_result = chrf_metric.compute(
            predictions=list(decoded_preds),
            references=list(decoded_labels)
        )
        
        return {
            "bleu": bleu_result["bleu"] * 100,
            "chrf": chrf_result["score"],
        }
    except Exception as e:
        log(f"⚠️  Metric computation error: {e}")
        return {"bleu": 0.0, "chrf": 0.0}

log(f"✓ Metrics: BLEU & chrF")

# =============================================================================
# TRAINING SETUP
# =============================================================================

log(f"\n🎯 Setting up training...")

steps_per_epoch = len(tokenized_train) // effective_batch
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

log(f"  Effective batch: {effective_batch}")
log(f"  Steps per epoch: {steps_per_epoch}")
log(f"  Total steps: {total_steps}")
log(f"  Warmup steps: {warmup_steps}")

# Estimate time
time_per_step = 0.8  # 4x H100 is fast
estimated_hours = total_steps * time_per_step / 3600
log(f"  Estimated time: ~{estimated_hours:.1f} hours on {num_gpus} GPU(s)")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training schedule
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    # Optimization
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=1.0,
    label_smoothing_factor=0.0,
    
    # Precision - NLLB/M2M100 has issues with fp16 + accelerate
    fp16=False,
    bf16=True,  # Use bf16 instead - works better with NLLB
    # Evaluation
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    
    # CHECKPOINTING - Critical for preemption resilience
    save_strategy="steps",
    save_steps=SAVE_EVERY,
    save_total_limit=5,  # Keep more checkpoints for safety
    load_best_model_at_end=True,
    metric_for_best_model="eval_chrf",
    greater_is_better=True,
    
    # Logging
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOG_EVERY,
    report_to=["tensorboard"],
    
    # Generation
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=2,
    
    # Performance - Multi-GPU optimized
    dataloader_num_workers=4,  # Per process
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    dataloader_prefetch_factor=2,
    
    # Multi-GPU / DDP
    ddp_find_unused_parameters=False,
    ddp_broadcast_buffers=False,
    
    # Misc
    push_to_hub=False,
    seed=42,
    
    # Ignore data skip for faster resume (we save frequently enough)
    ignore_data_skip=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # Pass model - it uses prepare_decoder_input_ids_from_labels()
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100  # Explicitly set
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5),
        shutdown_callback,
    ]
)

log(f"✓ Trainer ready")
log(f"✓ Multi-GPU training will use all {num_gpus} GPU(s) automatically")

# =============================================================================
# TRAINING WITH CHECKPOINT RESUME
# =============================================================================

log("\n" + "="*70)
log("🚀 STARTING TRAINING")
log("="*70)
log(f"\nDataset: {len(tokenized_train):,} bidirectional examples")
log(f"Languages: 6 African/Asian languages")
log(f"Training: Bidirectional (EN↔Target)")
log(f"GPUs: {num_gpus}")
log(f"Effective batch: {effective_batch}")
log(f"Checkpoint every: {SAVE_EVERY} steps")
if resume_checkpoint:
    log(f"RESUMING FROM: {resume_checkpoint}")
log(f"\nMonitor: tensorboard --logdir {OUTPUT_DIR}/logs")
log("="*70 + "\n")

try:
    # Resume from checkpoint if available
    if resume_checkpoint:
        log(f"🔄 Resuming training from {resume_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        log(f"🆕 Starting fresh training")
        train_result = trainer.train()
    
    log("\n" + "="*70)
    log("✅ TRAINING COMPLETED!")
    log("="*70)
    
    log(f"\n💾 Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metadata
    metadata = {
        "version": "consensus_4gpu_checkpoint",
        "status": "completed",
        "num_gpus": num_gpus,
        "effective_batch_size": effective_batch,
        "languages": list(LANGUAGE_MAPPING.keys()),
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "language_counts": language_counts if 'language_counts' in dir() else {},
        "skipped_counts": skipped_counts if 'skipped_counts' in dir() else {},
        "end_time": datetime.now().isoformat(),
        "total_steps": train_result.metrics.get('train_steps', 'N/A'),
        "epochs_completed": train_result.metrics.get('epoch', 'N/A'),
        "final_train_loss": train_result.metrics.get('train_loss', 'N/A'),
        "best_eval_bleu": train_result.metrics.get('eval_bleu', 'N/A'),
        "best_eval_chrf": train_result.metrics.get('eval_chrf', 'N/A'),
        "training_hours": train_result.metrics.get('train_runtime', 0) / 3600,
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    log(f"\n📊 Training Statistics:")
    log(f"  GPUs used: {num_gpus}")
    log(f"  Total steps: {metadata['total_steps']}")
    log(f"  Epochs: {metadata['epochs_completed']:.2f}" if isinstance(metadata['epochs_completed'], float) else f"  Epochs: {metadata['epochs_completed']}")
    log(f"  Final loss: {metadata['final_train_loss']:.4f}" if isinstance(metadata['final_train_loss'], float) else f"  Final loss: {metadata['final_train_loss']}")
    log(f"  Best BLEU: {metadata['best_eval_bleu']:.2f}" if isinstance(metadata['best_eval_bleu'], float) else f"  Best BLEU: {metadata['best_eval_bleu']}")
    log(f"  Best chrF: {metadata['best_eval_chrf']:.2f}" if isinstance(metadata['best_eval_chrf'], float) else f"  Best chrF: {metadata['best_eval_chrf']}")
    log(f"  Time: {metadata['training_hours']:.2f} hours" if isinstance(metadata['training_hours'], float) else f"  Time: {metadata['training_hours']}")
    
    log(f"\n✅ Model saved to: {OUTPUT_DIR}")
    
except KeyboardInterrupt:
    log("\n\n⚠️  Training interrupted by user")
    log("💾 Saving checkpoint before exit...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "checkpoint-interrupted"))
    log("✓ Checkpoint saved. Resume with same command.")
    
except Exception as e:
    log(f"\n\n❌ Training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Try to save on error
    try:
        log("💾 Attempting to save checkpoint on error...")
        trainer.save_model(os.path.join(OUTPUT_DIR, "checkpoint-error"))
        log("✓ Emergency checkpoint saved")
    except Exception as save_err:
        log(f"⚠️  Failed to save emergency checkpoint: {save_err}")
    
    raise

log("\n" + "="*70)
