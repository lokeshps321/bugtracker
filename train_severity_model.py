"""
Industry-Standard Severity Classification Training
- GPU optimized for RTX 3050 6GB
- Checkpoint system (resume from failure)
- Early stopping
- Comprehensive evaluation
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_severity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("datasets/preprocessed")
OUTPUT_DIR = Path("severity_model_new")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CACHE_DIR = Path(".cache/transformers")

# Training hyperparameters (optimized for RTX 3050 6GB)
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,  # Reduced for memory efficiency
    'batch_size': 16,   # Safe for 6GB VRAM
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'fp16': True,  # Mixed precision for memory efficiency
    'gradient_accumulation_steps': 2,  # Effective batch size = 32
    'early_stopping_patience': 2,
    'save_total_limit': 2,  # Keep only 2 best checkpoints
}

# Label mapping
LABEL_MAP = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def prepare_dataset():
    """Load and prepare datasets"""
    logger.info("Loading datasets...")
    
    # Load CSVs
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # Convert labels to IDs
    train_df['label'] = train_df['severity'].map(LABEL_MAP)
    val_df['label'] = val_df['severity'].map(LABEL_MAP)
    test_df['label'] = test_df['severity'].map(LABEL_MAP)
    
    # Remove any rows with invalid labels
    train_df = train_df.dropna(subset=['label'])
    val_df = val_df.dropna(subset=['label'])
    test_df = test_df.dropna(subset=['label'])
    
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def tokenize_function(examples, tokenizer):
    """Tokenize text data"""
    return tokenizer(
        examples['description'],
        padding='max_length',
        truncation=True,
        max_length=CONFIG['max_length']
    )

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    
    report = classification_report(
        labels, 
        preds,
        target_names=list(LABEL_MAP.keys()),
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': acc,
        'f1_macro': report['macro avg']['f1-score'],
        'f1_weighted': report['weighted avg']['f1-score']
    }

def train_model():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("SEVERITY CLASSIFICATION TRAINING")
    logger.info("=" * 60)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nStep 1/5: Loading data...")
    train_df, val_df, test_df = prepare_dataset()
    
    # Save to temporary files for datasets library
    train_df[['description', 'label']].to_csv('/tmp/train_severity.csv', index=False)
    val_df[['description', 'label']].to_csv('/tmp/val_severity.csv', index=False)
    test_df[['description', 'label']].to_csv('/tmp/test_severity.csv', index=False)
    
    # Load as HuggingFace datasets
    dataset = load_dataset('csv', data_files={
        'train': '/tmp/train_severity.csv',
        'validation': '/tmp/val_severity.csv',
        'test': '/tmp/test_severity.csv'
    })
    
    logger.info("✓ Data loaded")
    
    # Load tokenizer
    logger.info("\nStep 2/5: Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(
        CONFIG['model_name'],
        cache_dir=CACHE_DIR
    )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['description']
    )
    
    logger.info("✓ Tokenization complete")
    
    # Load model
    logger.info("\nStep 3/5: Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(LABEL_MAP),
        cache_dir=CACHE_DIR
    )
    
    if device == 'cuda':
        model = model.to(device)
    
    logger.info("✓ Model loaded")
    
    # Training arguments
    logger.info("\nStep 4/5: Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=CONFIG['warmup_steps'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        fp16=CONFIG['fp16'] and device == 'cuda',
        logging_dir=str(OUTPUT_DIR / 'logs'),
        logging_steps=100,
        save_total_limit=CONFIG['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        report_to='none',  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])]
    )
    
    logger.info("✓ Trainer initialized")
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Model: {CONFIG['model_name']}")
    logger.info(f"  Batch size: {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']})")
    logger.info(f"  Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  Epochs: {CONFIG['num_epochs']}")
    logger.info(f"  FP16: {CONFIG['fp16'] and device == 'cuda'}")
    
    # Train
    logger.info("\nStep 5/5: Training model...")
    logger.info("=" * 60)
    
    try:
        train_result = trainer.train()
        logger.info("\n✓ Training complete!")
        logger.info(f"  Best accuracy: {train_result.metrics.get('eval_accuracy', 'N/A')}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    
    logger.info("\nTest Set Results:")
    logger.info(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"  F1 (Macro): {test_results['eval_f1_macro']:.4f}")
    logger.info(f"  F1 (Weighted): {test_results['eval_f1_weighted']:.4f}")
    
    # Detailed classification report
    logger.info("\nGenerating detailed metrics...")
    predictions = trainer.predict(tokenized_datasets['test'])
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    
    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=list(LABEL_MAP.keys()),
        zero_division=0
    )
    
    logger.info("\nClassification Report:")
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    logger.info("\nConfusion Matrix:")
    logger.info(f"{'':>10} " + " ".join(f"{k:>10}" for k in LABEL_MAP.keys()))
    for i, label in enumerate(LABEL_MAP.keys()):
        logger.info(f"{label:>10} " + " ".join(f"{cm[i][j]:>10}" for j in range(len(LABEL_MAP))))
    
    # Save model
    logger.info("\nSaving final model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label mapping
    with open(OUTPUT_DIR / 'label_map.json', 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_results['eval_accuracy']),
        'test_f1_macro': float(test_results['eval_f1_macro']),
        'test_f1_weighted': float(test_results['eval_f1_weighted']),
        'training_config': CONFIG,
        'classification_report': classification_report(
            true_labels, pred_labels,
            target_names=list(LABEL_MAP.keys()),
            output_dict=True,
            zero_division=0
        )
    }
    
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✓ Model saved to {OUTPUT_DIR}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ SEVERITY MODEL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nFinal Results:")
    logger.info(f"  📊 Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    logger.info(f"  🎯 F1 Score (Macro): {test_results['eval_f1_macro']:.4f}")
    logger.info(f"  💾 Model saved to: {OUTPUT_DIR.absolute()}")
    logger.info(f"\nTarget: 85%+ accuracy")
    
    if test_results['eval_accuracy'] >= 0.85:
        logger.info("  ✅ TARGET ACHIEVED!")
    else:
        logger.info(f"  ⚠️  Close to target (need {(0.85 - test_results['eval_accuracy'])*100:.1f}% more)")
    
    return test_results['eval_accuracy']

if __name__ == "__main__":
    try:
        accuracy = train_model()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        raise
