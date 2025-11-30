"""
Industry-Standard Dataset Preprocessing for BugFlow
- Robust error handling
- Checkpoint system (resume from failure)
- Memory-efficient batch processing
- Progress tracking
"""

import pandas as pd
import numpy as np
import re
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "github_issues.csv"
OUTPUT_DIR = Path("datasets/preprocessed")
CHECKPOINT_DIR = Path("datasets/checkpoints")
SAMPLE_SIZE = 100000  # Use 100K for faster training (you can increase later)
RANDOM_SEED = 42

# Label mappings (GitHub labels → BugFlow categories)
SEVERITY_MAPPING = {
    'critical': 'critical',
    'blocker': 'critical',
    'high': 'high',
    'urgent': 'high',
    'medium': 'medium',
    'normal': 'medium',
    'low': 'low',
    'minor': 'low',
    'trivial': 'low',
}

TEAM_KEYWORDS = {
    'Frontend': ['ui', 'frontend', 'react', 'vue', 'angular', 'css', 'html', 'button', 'display', 'visual', 'layout', 'responsive'],
    'Backend': ['api', 'backend', 'server', 'database', 'sql', 'endpoint', 'authentication', 'performance', 'query'],
    'Mobile': ['android', 'ios', 'mobile', 'app', 'touch', 'gesture', 'notification'],
    'DevOps': ['deploy', 'ci', 'cd', 'docker', 'kubernetes', 'build', 'pipeline', 'infrastructure', 'crash']
}

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def infer_severity(text):
    """Infer severity from text using keywords"""
    text_lower = text.lower()
    
    # Critical/Blocker keywords
    if any(word in text_lower for word in ['crash', 'critical', 'blocker', 'security', 'data loss', 'cannot', 'broken']):
        return 'critical'
    
    # High priority keywords
    if any(word in text_lower for word in ['bug', 'error', 'issue', 'problem', 'fail', 'not work']):
        return 'high'
    
    # Low priority keywords
    if any(word in text_lower for word in ['typo', 'minor', 'improvement', 'suggestion', 'enhancement']):
        return 'low'
    
    # Default to medium
    return 'medium'

def infer_team(text):
    """Infer team assignment from text using keywords"""
    text_lower = text.lower()
    scores = {team: 0 for team in TEAM_KEYWORDS.keys()}
    
    for team, keywords in TEAM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[team] += 1
    
    # Return team with highest score, default to Backend
    max_team = max(scores, key=scores.get)
    return max_team if scores[max_team] > 0 else 'Backend'

def process_batch(df_batch, batch_num):
    """Process a batch of data with error handling"""
    try:
        logger.info(f"Processing batch {batch_num} ({len(df_batch)} rows)")
        
        # Combine title and body for full text
        df_batch['description'] = (
            df_batch['issue_title'].fillna('') + ' ' + df_batch['body'].fillna('')
        )
        
        # Clean text
        df_batch['description'] = df_batch['description'].apply(clean_text)
        
        # Remove empty or very short descriptions
        df_batch = df_batch[df_batch['description'].str.len() > 20]
        
        # Infer labels
        df_batch['severity'] = df_batch['description'].apply(infer_severity)
        df_batch['team'] = df_batch['description'].apply(infer_team)
        
        # Add project name (extract from URL)
        df_batch['project'] = df_batch['issue_url'].apply(
            lambda x: x.split('/')[4] if '/' in str(x) else 'Unknown'
        )
        
        # Select final columns
        df_processed = df_batch[['description', 'project', 'severity', 'team']].copy()
        
        return df_processed
    
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return None

def save_checkpoint(data, checkpoint_file):
    """Save processing checkpoint"""
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(checkpoint_file, index=False)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")

def load_checkpoint(checkpoint_file):
    """Load processing checkpoint if exists"""
    if checkpoint_file.exists():
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        return pd.read_csv(checkpoint_file)
    return None

def main():
    """Main preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("BugFlow Dataset Preprocessing - Industry Standard")
    logger.info("=" * 60)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    final_checkpoint = CHECKPOINT_DIR / "processed_sample.csv"
    processed_df = load_checkpoint(final_checkpoint)
    
    if processed_df is not None:
        logger.info(f"✓ Found existing processed data ({len(processed_df)} rows)")
    else:
        logger.info("Step 1/4: Loading raw data...")
        try:
            # Read only necessary columns in chunks to save memory
            chunks = []
            chunksize = 50000
            total_rows = 0
            
            for chunk in tqdm(pd.read_csv(INPUT_FILE, chunksize=chunksize, nrows=SAMPLE_SIZE),
                            desc="Loading data", total=SAMPLE_SIZE//chunksize):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows >= SAMPLE_SIZE:
                    break
            
            df_raw = pd.concat(chunks, ignore_index=True)
            logger.info(f"✓ Loaded {len(df_raw)} rows")
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return
        
        logger.info("Step 2/4: Processing and cleaning data...")
        try:
            # Process in batches
            batch_size = 10000
            processed_batches = []
            
            for i in range(0, len(df_raw), batch_size):
                batch = df_raw.iloc[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                processed_batch = process_batch(batch, batch_num)
                if processed_batch is not None and len(processed_batch) > 0:
                    processed_batches.append(processed_batch)
                
                # Save intermediate checkpoint every 5 batches
                if batch_num % 5 == 0:
                    temp_df = pd.concat(processed_batches, ignore_index=True)
                    save_checkpoint(temp_df, CHECKPOINT_DIR / f"temp_batch_{batch_num}.csv")
            
            processed_df = pd.concat(processed_batches, ignore_index=True)
            logger.info(f"✓ Processed {len(processed_df)} valid rows")
            
            # Save final checkpoint
            save_checkpoint(processed_df, final_checkpoint)
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return
    
    logger.info("Step 3/4: Creating train/validation/test splits...")
    try:
        # Stratified split to maintain label distribution
        train_df, temp_df = train_test_split(
            processed_df, 
            test_size=0.3, 
            random_state=RANDOM_SEED,
            stratify=processed_df['severity']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_df['severity']
        )
        
        logger.info(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"Split failed: {str(e)}")
        return
    
    logger.info("Step 4/4: Saving processed datasets...")
    try:
        train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
        val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
        test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
        
        # Save label distributions for verification
        stats = {
            'total_samples': len(processed_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'severity_distribution': processed_df['severity'].value_counts().to_dict(),
            'team_distribution': processed_df['team'].value_counts().to_dict()
        }
        
        with open(OUTPUT_DIR / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("✓ All datasets saved successfully!")
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Total: {stats['total_samples']}")
        logger.info(f"  Train: {stats['train_samples']}")
        logger.info(f"  Val: {stats['val_samples']}")
        logger.info(f"  Test: {stats['test_samples']}")
        logger.info(f"\nSeverity Distribution:")
        for sev, count in stats['severity_distribution'].items():
            logger.info(f"  {sev}: {count} ({count/stats['total_samples']*100:.1f}%)")
        logger.info(f"\nTeam Distribution:")
        for team, count in stats['team_distribution'].items():
            logger.info(f"  {team}: {count} ({count/stats['total_samples']*100:.1f}%)")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PREPROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
        logger.info("Next step: Run fine-tuning scripts")
        
    except Exception as e:
        logger.error(f"Save failed: {str(e)}")
        return

if __name__ == "__main__":
    main()
