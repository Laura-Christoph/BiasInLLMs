import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
import random

# 1. Load data in chunks (chunking manually)
chunk_size = 5000

def read_in_chunks(path):
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        batch = []
        first_line = True
        for line_num, line in enumerate(f):
            if line.strip() == '':
                continue
            if first_line:
                first_line = False
                if 'text' in line.lower() and 'label' in line.lower():
                    print(f"Skipping header line: {line.strip()}")
                    continue
            try:
                text, label = line.strip().split('\t')
                batch.append({'text': text, 'label': int(label)})
            except ValueError:
                print(f"Problem parsing line {line_num}: {line.strip()}")
                continue
            if len(batch) == chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch

# 2. Load tokenizer and model
model_path = "/Users/laurachristoph/Desktop/best_distilbert_model_retrained"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 3. Use M1 GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# 4. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 5. Prepare a small part of the data for hyperparameter search
print("Loading a small subset for hyperparameter tuning...")
data_iter = read_in_chunks("/Users/laurachristoph/Desktop/Bachelorarbeit/big_reli_hate_dataset.txt")
small_batch = next(data_iter)
df_small = pd.DataFrame(small_batch)
dataset_small = Dataset.from_pandas(df_small)

# 6. Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

dataset_small = dataset_small.map(tokenize_function, batched=True)

# 7. Define training arguments for search
training_args = TrainingArguments(
    output_dir="/tmp/test_trainer",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    report_to=None,
    push_to_hub=False,
    logging_dir="/tmp/logs",
    logging_steps=10,
    load_best_model_at_end=False
)

# 8. Trainer for small batch
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_small,
    eval_dataset=dataset_small,
    compute_metrics=compute_metrics,
)

# 9. Search for best learning rate (simple version)
print("\n--- Starting trial training to find best learning rate ---\n")
trainer.train()

# 10. Now train full model chunk-by-chunk
all_batches = []
print("\n--- Starting full training ---\n")
for batch_idx, batch in enumerate(read_in_chunks("/Users/laurachristoph/Desktop/Bachelorarbeit/big_reli_hate_dataset.txt")):
    print(f"Processing batch {batch_idx+1}")
    df = pd.DataFrame(batch)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_function, batched=True)

    trainer.train_dataset = dataset
    trainer.train()

print("\n--- Training complete! Evaluating... ---\n")

# 11. Final evaluation
print("Evaluating on last seen batch:")
metrics = trainer.evaluate()
print(metrics)

# Save final model
final_save_path = "/Users/laurachristoph/Desktop/best_distilbert_model_7"
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print(f"\nFinal model saved to {final_save_path}")