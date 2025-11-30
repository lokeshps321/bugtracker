"""
Deduplication Model Training using Sentence Transformers
- Creates semantic embeddings for bug descriptions
- Uses cosine similarity to detect duplicates
- Target: 90%+ precision
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("datasets/preprocessed")
OUTPUT_DIR = Path("dedup_model")
CACHE_DIR = Path(".cache/sentence_transformers")

CONFIG = {
    'base_model': 'all-MiniLM-L6-v2',  # Fast and accurate
    'batch_size': 32,
    'epochs': 3,
    'warmup_steps': 100,
    'similarity_threshold': 0.85,  # 85%+ similarity = duplicate
    'max_triplets': 20000,  # For faster training
}

def create_triplet_data(df, max_triplets=20000):
    """Create triplet training data (anchor, positive, negative)"""
    logger.info("Creating triplet training data...")
    
    triplets = []
    
    # Group by project to find similar bugs
    projects = df['project'].unique()[:50]  # Use top 50 projects
    
    for project in projects:
        project_bugs = df[df['project'] == project]['description'].tolist()
        other_bugs = df[df['project'] != project]['description'].tolist()
        
        if len(project_bugs) < 2 or len(other_bugs) < 1:
            continue
        
        # Create triplets
        for i in range(min(len(project_bugs) - 1, max_triplets // len(projects))):
            anchor = project_bugs[i]
            positive = project_bugs[i + 1]  # Same project = similar
            negative_idx = np.random.randint(0, len(other_bugs))
            negative = other_bugs[negative_idx]  # Different project = different
            
            triplets.append(InputExample(texts=[anchor, positive, negative]))
            
            if len(triplets) >= max_triplets:
                break
        
        if len(triplets) >= max_triplets:
            break
    
    logger.info(f"Created {len(triplets)} triplets")
    return triplets

def evaluate_deduplication(model, test_df, threshold=0.85):
    """Evaluate deduplication performance"""
    logger.info("\nEvaluating deduplication model...")
    
    # Create test pairs
    # True duplicates: same project bugs (ground truth approximation)
    # True non-duplicates: different project bugs
    
    descriptions = test_df['description'].tolist()[:1000]  # Use 1000 samples
    projects = test_df['project'].tolist()[:1000]
    
    # Encode all descriptions
    logger.info("Encoding test descriptions...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    # Calculate similarities
    similarities = cosine_similarity(embeddings)
    
    # Create ground truth labels
    y_true = []
    y_pred = []
    
    for i in range(len(descriptions)):
        for j in range(i + 1, min(i + 50, len(descriptions))):  # Compare with next 50
            # Ground truth: same project = duplicate (approximation)
            is_duplicate = (projects[i] == projects[j])
            y_true.append(int(is_duplicate))
            
            # Prediction based on similarity threshold
            sim = similarities[i][j]
            y_pred.append(int(sim >= threshold))
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    logger.info(f"\nDeduplication Metrics:")
    logger.info(f"  Precision: {precision*100:.2f}% (of predicted duplicates, how many are real)")
    logger.info(f"  Recall: {recall*100:.2f}% (of real duplicates, how many detected)")
    logger.info(f"  F1-Score: {f1*100:.2f}%")
    logger.info(f"  Threshold: {threshold}")
    
    return precision, recall, f1

def train_deduplication():
    logger.info("=" * 70)
    logger.info("DEDUPLICATION MODEL TRAINING")
    logger.info("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    train_df = pd.read_csv(DATA_DIR / "train.csv").head(30000)
    test_df = pd.read_csv(DATA_DIR / "test.csv").head(5000)
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create triplet training data
    triplets = create_triplet_data(train_df, max_triplets=CONFIG['max_triplets'])
    
    # Load base model
    logger.info(f"\nLoading base model: {CONFIG['base_model']}...")
    model = SentenceTransformer(CONFIG['base_model'], cache_folder=str(CACHE_DIR))
    
    # Prepare training
    train_dataloader = DataLoader(triplets, shuffle=True, batch_size=CONFIG['batch_size'])
    train_loss = losses.TripletLoss(model)
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info(f"Triplets: {len(triplets)}, Batch: {CONFIG['batch_size']}, Epochs: {CONFIG['epochs']}")
    logger.info("=" * 70 + "\n")
    
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=CONFIG['epochs'],
        warmup_steps=CONFIG['warmup_steps'],
        show_progress_bar=True,
    )
    
    logger.info("\n✅ Training complete!")
    
    # Evaluate
    precision, recall, f1 = evaluate_deduplication(
        model, test_df, threshold=CONFIG['similarity_threshold']
    )
    
    # Check target
    logger.info("\n" + "=" * 70)
    if precision >= 0.90:
        logger.info("🎯 TARGET ACHIEVED: 90%+ precision!")
    else:
        logger.info(f"⚠️  Close to target: {precision*100:.1f}% precision (need {(0.90-precision)*100:.1f}% more)")
    logger.info("=" * 70)
    
    # Save model
    logger.info(f"\nSaving to {OUTPUT_DIR}...")
    model.save(str(OUTPUT_DIR))
    
    # Save config and metrics
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'similarity_threshold': CONFIG['similarity_threshold'],
        'training_config': CONFIG,
    }
    
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("✅ Model saved!")
    
    return precision

if __name__ == "__main__":
    try:
        precision = train_deduplication()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
