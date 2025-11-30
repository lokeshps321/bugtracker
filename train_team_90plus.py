"""
Enhanced Team Model Training - 90%+ Accuracy Target
- 50K samples for better coverage
- 5 epochs for thorough learning
- Class weighting to balance teams
- Cosine scheduler for stability
- Works on CPU (GPU issue persists)
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced config for 90%+ accuracy
DATA_DIR = Path("datasets/preprocessed")
OUTPUT_DIR = Path("team_model_90plus")
CACHE_DIR = Path(".cache/transformers")

CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,  # Longer for more context
    'batch_size': 32,   # CPU-safe
    'learning_rate': 2e-5,  # Lower for stability
    'num_epochs': 5,    # More thorough training
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'sample_size': 50000,  # 5x more data
}

LABEL_MAP = {'Backend': 0, 'Frontend': 1, 'Mobile': 2, 'DevOps': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def train_enhanced():
    logger.info("=" * 70)
    logger.info("ENHANCED TEAM MODEL TRAINING - TARGET: 90%+ ACCURACY")
    logger.info("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load larger dataset
    logger.info(f"\nLoading {CONFIG['sample_size']} training samples...")
    train_df = pd.read_csv(DATA_DIR / "train.csv").head(CONFIG['sample_size'])
    val_df = pd.read_csv(DATA_DIR / "val.csv").head(5000)  # More validation
    test_df = pd.read_csv(DATA_DIR / "test.csv").head(5000)
    
    # Map labels
    train_df['label'] = train_df['team'].map(LABEL_MAP).astype(int)
    val_df['label'] = val_df['team'].map(LABEL_MAP).astype(int)
    test_df['label'] = test_df['team'].map(LABEL_MAP).astype(int)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Analyze class distribution
    team_dist = train_df['team'].value_counts()
    logger.info(f"\nTeam distribution:")
    for team, count in team_dist.items():
        logger.info(f"  {team}: {count} ({count/len(train_df)*100:.1f}%)")
    
    # Compute class weights to balance training
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info(f"\nClass weights (to balance training):")
    for i, w in class_weights_dict.items():
        logger.info(f"  {ID_TO_LABEL[i]}: {w:.2f}")
    
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
        CONFIG['model_name'], 
        num_labels=4, 
        cache_dir=CACHE_DIR
    )
    
    if device == 'cuda':
        model = model.to(device)
    
    # Custom Trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Weighted cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(list(class_weights_dict.values()), dtype=torch.float).to(logits.device)
            )
            loss = loss_fct(logits, labels)
            
            return (loss, outputs) if return_outputs else loss
    
    # Training args with cosine scheduler
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / 'checkpoints'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        warmup_ratio=CONFIG['warmup_ratio'],
        weight_decay=CONFIG['weight_decay'],
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='none',
        lr_scheduler_type='cosine',  # Cosine decay
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(labels, preds)}
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTING ENHANCED TRAINING")
    logger.info(f"Samples: {len(train_dataset)}, Batch: {CONFIG['batch_size']}, Epochs: {CONFIG['num_epochs']}")
    logger.info(f"Class weighting: ENABLED (balances DevOps)")
    logger.info(f"LR scheduler: Cosine with warmup")
    logger.info("=" * 70 + "\n")
    
    # Train
    trainer.train()
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ TRAINING COMPLETE!")
    logger.info(f"Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    logger.info("=" * 70)
    
    # Detailed metrics
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    
    report = classification_report(true_labels, pred_labels, 
                                  target_names=list(LABEL_MAP.keys()), 
                                  zero_division=0)
    logger.info(f"\nDetailed Classification Report:")
    logger.info(f"\n{report}")
    
    # Per-class F1 scores
    report_dict = classification_report(
        true_labels, pred_labels,
        target_names=list(LABEL_MAP.keys()),
        output_dict=True, zero_division=0
    )
    
    logger.info("\nPer-Team Performance:")
    for team in LABEL_MAP.keys():
        f1 = report_dict[team]['f1-score']
        status = "✅" if f1 >= 0.90 else "⚠️" if f1 >= 0.85 else "❌"
        logger.info(f"  {status} {team}: {f1*100:.1f}% F1-score")
    
    # Check if target achieved
    if test_results['eval_accuracy'] >= 0.90:
        logger.info("\n🎯 TARGET ACHIEVED: 90%+ overall accuracy!")
    else:
        logger.info(f"\n⚠️  Close to target: {test_results['eval_accuracy']*100:.1f}% (need {(0.90-test_results['eval_accuracy'])*100:.1f}% more)")
    
    # Save
    logger.info(f"\nSaving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    with open(OUTPUT_DIR / 'label_map.json', 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    metrics = {
        'test_accuracy': float(test_results['eval_accuracy']),
        'training_config': CONFIG,
        'class_weights': class_weights_dict,
        'classification_report': report_dict
    }
    
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("✅ Model saved!")
    
    return test_results['eval_accuracy']

if __name__ == "__main__":
    try:
        accuracy = train_enhanced()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
