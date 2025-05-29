import os
# Disabling the MPS memory cap (use with caution)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import optuna
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ------------------------------
# Custom PyTorch Dataset
# ------------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ------------------------------
# Custom Trainer Override
# ------------------------------
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Higher penalty for false positives for class "1"
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([3, 0.05]).to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -------------------------------------
# 1. LOAD AND PREP TSV
# -------------------------------------
df = pd.read_csv("big_reli_hate_dataset.txt", sep="\t", encoding="latin1")
df["label"] = df["label"].astype(int)

# Shuffle the entire dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------------
# 2. CREATE TRAIN/EVAL SPLITS (10% for eval, with 1/3 positive)
# -------------------------------------
n_pos_eval = 200
n_neg_eval = 600

pos_df = df[df["label"] == 1].copy()
neg_df = df[df["label"] == 0].copy()

# Selecting the first n examples for evaluation
pos_eval = pos_df.iloc[:n_pos_eval]
neg_eval = neg_df.iloc[:n_neg_eval]
eval_df = pd.concat([pos_eval, neg_eval], ignore_index=True)

# Remaining data for training
pos_train = pos_df.iloc[n_pos_eval:]
neg_train = neg_df.iloc[n_neg_eval:]
train_df = pd.concat([pos_train, neg_train], ignore_index=True)

# Shuffling final splits
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)

# Extracting texts and labels
train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
eval_texts = eval_df["text"].tolist()
eval_labels = eval_df["label"].tolist()

# -------------------------------------
# 3. PREPARE DATASETS
# -------------------------------------
# Loading tokenizer from my pretrained model directory.
model_dir = "*/best_distilbert_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
eval_dataset = TextClassificationDataset(eval_texts, eval_labels, tokenizer)

# -------------------------------------
# 4. INIT MODEL
# -------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)

# Setting the device to MPS (M1) if available, otherwise CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Data collator to dynamically pad sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------------------------
# 5. SETUP TRAINER & HYPERPARAMETER SEARCH
# -------------------------------------
training_args = TrainingArguments(
    output_dir="./distilbert-output",
    num_train_epochs=3,  # initial value; will be tuned
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./distilbert-logs",
)

# Defining model_init for hyperparameter search
def model_init():
    new_model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    new_model.to(device)
    return new_model

trainer = CustomTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Defining hyperparameter search space 
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10)
    }

# Running hyperparameter search 
best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    direction="minimize",
    n_trials=10,
)

print("Best hyperparameters found:", best_run.hyperparameters)

# Updating training arguments with the best hyperparameters
trainer.args.learning_rate = best_run.hyperparameters["learning_rate"]
trainer.args.num_train_epochs = best_run.hyperparameters["num_train_epochs"]

# -------------------------------------
# 6. TRAINING THE MODEL
# -------------------------------------
trainer.train()

# -------------------------------------
# 7. SAVING THE FINAL MODEL
# -------------------------------------
# Setting a single, consistent save directory
save_directory = "*/best_distilbert_model_6"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Training complete! Model saved to '{save_directory}'.")

# -------------------------------------
# EVALUATING MODEL
# -------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predictions on Eval Set
preds_output = trainer.predict(eval_dataset)
pred_probs = preds_output.predictions
pred_labels = pred_probs.argmax(axis=1)

# Option: Extract true labels from dataset
true_labels = [example["labels"].item() for example in eval_dataset]

# Computing basic metrics
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='binary')
rec = recall_score(true_labels, pred_labels, average='binary')
f1 = f1_score(true_labels, pred_labels, average='binary')

print("\nEvaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["Not Hate", "Hate"], digits=4))

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hate", "Hate"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()