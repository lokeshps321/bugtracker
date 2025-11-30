"""
FAST Team Assignment Model Training - GPU Optimized
- 10K samples for speed
- Batch size 32-64 (GPU)
- 2 epochs
- Target: 10 minutes
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU-optimized config
DATA_DIR = Path("datasets/preprocessed")
OUTPUT_DIR = Path("team_model_new")
CACHE_DIR = Path(".cache/transformers")

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_fp16 = device == 'cuda'
batch_size = 64 if device == 'cuda' else 32  # Larger batch for GPU

CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': batch_size,
    'learning_rate': 3e-5,
    'num_epochs': 2,
    'fp16': use_fp16,
    'sample_size': 10000,
}

LABEL_MAP = {'Backend': 0, 'Frontend': 1, 'Mobile': 2, 'DevOps': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def train_fast():
    logger.info("=" * 60)
    logger.info("FAST TEAM ASSIGNMENT MODEL TRAINING")
    logger.info("=" * 60)
    
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    train_df = pd.read_csv(DATA_DIR / "train.csv").head(CONFIG['sample_size'])
    val_df = pd.read_csv(DATA_DIR / "val.csv").head(2000)
    test_df = pd.read_csv(DATA_DIR / "test.csv").head(2000)
    
    train_df['label'] = train_df['team'].map(LABEL_MAP).astype(int)
    val_df['label'] = val_df['team'].map(LABEL_MAP).astype(int)
    test_df['label'] = test_df['team'].map(LABEL_MAP).astype(int)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Team distribution: {train_df['team'].value_counts().to_dict()}")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['description', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['description', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['description', 'label']])
    
    # Tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'], cache_dir=CACHE_DIR)
    
    def tokenize(batch):
        return tokenizer(batch['description'], padding='max_length', truncation=True, max_length=CONFIG['max_length'])
    
    logger.info("Tokenizing...")
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['description'])
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=['description'])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['description'])
    
    # Load model
    logger.info("\nLoading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'], num_labels=4, cache_dir=CACHE_DIR
    )
    
    if device == 'cuda':
        model = model.to(device)
        logger.info("✓ Model moved to GPU")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / 'checkpoints'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        fp16=CONFIG['fp16'],
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='none',
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(labels, preds)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info(f"Samples: {len(train_dataset)}, Batch: {CONFIG['batch_size']}, Epochs: {CONFIG['num_epochs']}")
    logger.info(f"FP16: {CONFIG['fp16']}, Device: {device}")
    logger.info("=" * 60 + "\n")
    
    # Train
    trainer.train()
    
    # Evaluate
    logger.info("\nEvaluating...")
    test_results = trainer.evaluate(test_dataset)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ TRAINING COMPLETE!")
    logger.info(f"Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    logger.info("=" * 60)
    
    # Detailed metrics
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    
    report = classification_report(true_labels, pred_labels, 
                                  target_names=list(LABEL_MAP.keys()), 
                                  zero_division=0)
    logger.info(f"\n{report}")
    
    # Save
    logger.info(f"\nSaving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    with open(OUTPUT_DIR / 'label_map.json', 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    metrics = {
        'test_accuracy': float(test_results['eval_accuracy']),
        'training_config': CONFIG,
        'classification_report': classification_report(
            true_labels, pred_labels,
            target_names=list(LABEL_MAP.keys()),
            output_dict=True, zero_division=0
        )
    }
    
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("✅ Model saved!")
    
    return test_results['eval_accuracy']

if __name__ == "__main__":
    try:
        accuracy = train_fast()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
