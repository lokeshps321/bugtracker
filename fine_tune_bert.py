import pandas as pd
import sqlite3
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import logging
import os
import json
from datetime import datetime

# ----------------------
# Logging setup
# ----------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------
# Dataset class
# ----------------------
class BugDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_length=128):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            description,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ----------------------
# Load feedback data from DB
# ----------------------
def load_feedback_data():
    try:
        conn = sqlite3.connect('bugflow.db')
        # Load feedback records where either severity or team correction is provided
        query = """
        SELECT b.description, f.correction_severity, f.correction_team
        FROM feedback f
        JOIN bugs b ON f.bug_id = b.id
        WHERE f.correction_severity IS NOT NULL OR f.correction_team IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("No feedback data found")
            return [], [], [], []

        logger.info(f"Loaded {len(df)} feedback records from database")

        # Separate feedback for severity and team models as they might have different records
        severity_feedback = df.dropna(subset=['correction_severity'])
        team_feedback = df.dropna(subset=['correction_team'])

        logger.info(f"Severity feedback records: {len(severity_feedback)}")
        logger.info(f"Team feedback records: {len(team_feedback)}")

        # Return separate data for each model
        severity_descriptions = severity_feedback['description'].tolist()
        severity_labels = severity_feedback['correction_severity'].tolist()
        team_descriptions = team_feedback['description'].tolist()
        team_labels = team_feedback['correction_team'].tolist()

        return severity_descriptions, severity_labels, team_descriptions, team_labels
    except Exception as e:
        logger.error(f"Error loading feedback data: {str(e)}")
        raise

# ----------------------
# Fine-tune models
# ----------------------
def fine_tune_model():
    logger.info("="*50)
    logger.info("Starting model fine-tuning process...")
    logger.info("="*50)
    
    severity_descriptions, severity_labels, team_descriptions, team_labels = load_feedback_data()

    if not severity_descriptions and not team_descriptions:
        logger.error("No feedback data available for fine-tuning")
        return

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ----------------------
    # Fine-tune severity model if we have severity feedback
    # ----------------------
    if severity_descriptions and len(severity_descriptions) >= 2:  # Need at least 2 samples for train/val split
        logger.info("="*50)
        logger.info(f"Fine-tuning severity model with {len(severity_descriptions)} samples")
        logger.info("="*50)
        
        severity_encoder = LabelEncoder()
        severity_labels_encoded = severity_encoder.fit_transform(severity_labels)
        
        # Only split if we have enough data, otherwise use all for training
        if len(severity_descriptions) >= 4:
            X_train, X_val, y_train, y_val = train_test_split(
                severity_descriptions, severity_labels_encoded, 
                test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = severity_descriptions, severity_labels_encoded
            X_val, y_val = severity_descriptions, severity_labels_encoded  # Use same for validation

        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        train_dataset_severity = BugDataset(X_train, y_train, tokenizer)
        val_dataset_severity = BugDataset(X_val, y_val, tokenizer)

        # Try to load existing model, or create new one
        try:
            severity_model = DistilBertForSequenceClassification.from_pretrained(
                './severity_model',
                num_labels=len(severity_encoder.classes_)
            ).to(device)
            logger.info("Loaded existing severity model for fine-tuning")
        except:
            severity_model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(severity_encoder.classes_)
            ).to(device)
            logger.info("Created new severity model from base DistilBERT")

        # Improved training arguments
        training_args = TrainingArguments(
            output_dir='./severity_model',
            num_train_epochs=5,  # Increased from 3 for better convergence
            per_device_train_batch_size=4 if len(X_train) >= 4 else 1,
            per_device_eval_batch_size=4 if len(X_val) >= 4 else 1,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,
            eval_strategy="epoch" if len(X_val) >= 2 else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if len(X_val) >= 2 else False,
            metric_for_best_model="eval_loss" if len(X_val) >= 2 else None,
            report_to="none",  # Disable wandb
        )

        trainer_severity = Trainer(
            model=severity_model,
            args=training_args,
            train_dataset=train_dataset_severity,
            eval_dataset=val_dataset_severity if len(X_val) >= 2 else None,
        )

        logger.info("Starting severity model training...")
        train_result = trainer_severity.train()
        logger.info(f"Severity model training completed. Loss: {train_result.training_loss:.4f}")
        
        # Save model
        trainer_severity.save_model('./severity_model')
        pd.Series(severity_encoder.classes_).to_json('severity_labels.json')
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(severity_descriptions),
            'classes': severity_encoder.classes_.tolist(),
            'final_loss': float(train_result.training_loss)
        }
        with open('./severity_model/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Severity model fine-tuned and saved")
    else:
        logger.warning(f"Not enough severity feedback data (need at least 2, have {len(severity_descriptions)})")

    # ----------------------
    # Fine-tune team model if we have team feedback
    # ----------------------
    if team_descriptions and len(team_descriptions) >= 2:
        logger.info("="*50)
        logger.info(f"Fine-tuning team model with {len(team_descriptions)} samples")
        logger.info("="*50)
        
        team_encoder = LabelEncoder()
        team_labels_encoded = team_encoder.fit_transform(team_labels)
        
        # Only split if we have enough data
        if len(team_descriptions) >= 4:
            X_train, X_val, y_train, y_val = train_test_split(
                team_descriptions, team_labels_encoded, 
                test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = team_descriptions, team_labels_encoded
            X_val, y_val = team_descriptions, team_labels_encoded

        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        train_dataset_team = BugDataset(X_train, y_train, tokenizer)
        val_dataset_team = BugDataset(X_val, y_val, tokenizer)

        # Try to load existing model, or create new one
        try:
            team_model = DistilBertForSequenceClassification.from_pretrained(
                './team_model',
                num_labels=len(team_encoder.classes_)
            ).to(device)
            logger.info("Loaded existing team model for fine-tuning")
        except:
            team_model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(team_encoder.classes_)
            ).to(device)
            logger.info("Created new team model from base DistilBERT")

        # Same improved training arguments
        training_args_team = TrainingArguments(
            output_dir='./team_model',
            num_train_epochs=5,
            per_device_train_batch_size=4 if len(X_train) >= 4 else 1,
            per_device_eval_batch_size=4 if len(X_val) >= 4 else 1,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,
            eval_strategy="epoch" if len(X_val) >= 2 else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if len(X_val) >= 2 else False,
            metric_for_best_model="eval_loss" if len(X_val) >= 2 else None,
            report_to="none",
        )

        trainer_team = Trainer(
            model=team_model,
            args=training_args_team,
            train_dataset=train_dataset_team,
            eval_dataset=val_dataset_team if len(X_val) >= 2 else None,
        )

        logger.info("Starting team model training...")
        train_result = trainer_team.train()
        logger.info(f"Team model training completed. Loss: {train_result.training_loss:.4f}")
        
        # Save model
        trainer_team.save_model('./team_model')
        pd.Series(team_encoder.classes_).to_json('team_labels.json')
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(team_descriptions),
            'classes': team_encoder.classes_.tolist(),
            'final_loss': float(train_result.training_loss)
        }
        with open('./team_model/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Team model fine-tuned and saved")
    else:
        logger.warning(f"Not enough team feedback data (need at least 2, have {len(team_descriptions)})")

    logger.info("="*50)
    logger.info("✅ Fine-tuning completed and models saved")
    logger.info("="*50)

# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    fine_tune_model()

