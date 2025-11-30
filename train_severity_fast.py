"""
FAST Training Version - Optimized for Speed
- Small dataset (10K samples)
- Larger batch size
- Fewer epochs
- Target: 10-15 minutes
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

# Fast configuration
DATA_DIR = Path("datasets/preprocessed")
OUTPUT_DIR = Path("severity_model_new")
CACHE_DIR = Path(".cache/transformers")

CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,  # Shorter for speed
    'batch_size': 32,   # Larger batch
    'learning_rate': 3e-5,
    'num_epochs': 2,    # Just 2 epochs
    'fp16': True,
    'sample_size': 10000,  # Only 10K samples
}

LABEL_MAP = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def train_fast():
    logger.info("=" * 60)
    logger.info("FAST SEVERITY MODEL TRAINING")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load SMALL dataset
    logger.info("\nLoading small dataset...")
    train_df = pd.read_csv(DATA_DIR / "train.csv").head(CONFIG['sample_size'])
    val_df = pd.read_csv(DATA_DIR / "val.csv").head(2000)
    test_df = pd.read_csv(DATA_DIR / "test.csv").head(2000)
    
    train_df['label'] = train_df['severity'].map(LABEL_MAP).astype(int)
    val_df['label'] = val_df['severity'].map(LABEL_MAP).astype(int)
    test_df['label'] = test_df['severity'].map(LABEL_MAP).astype(int)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create HF datasets
    train_dataset = Dataset.from_pandas(train_df[['description', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['description', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['description', 'label']])
    
    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'], cache_dir=CACHE_DIR)
    
    # Tokenize
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
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / 'checkpoints'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        fp16=CONFIG['fp16'] and device == 'cuda',
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
    logger.info(f"Estimated steps: {len(train_dataset) // CONFIG['batch_size'] * CONFIG['num_epochs']}")
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
    
    # Get detailed metrics
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    
    report = classification_report(true_labels, pred_labels, 
                                  target_names=list(LABEL_MAP.keys()), 
                                  zero_division=0)
    logger.info(f"\n{report}")
    
    # Save model
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
