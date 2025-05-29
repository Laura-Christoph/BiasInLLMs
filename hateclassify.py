import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

# ----------------------------
# 1) Loading my DistilBERT model
# ----------------------------
model_path = "*/best_distilbert_model_7"
#model_path = "*/best_distilbert_model_retrained"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# 2) Helper function for batched inference
# ----------------------------
def classify_texts(texts, batch_size=16):
    predicted_labels = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=1)
        predicted_labels.extend(preds.cpu().tolist())
    return predicted_labels

# ----------------------------
# 3) Processing the TSV file in chunks without pandas
# ----------------------------
input_file = "*/chunk3_preannotated_V2.txt"
output_file = "*/chunk3_annotated_V2.tsv"
chunk_size = 15000  # number of lines to process per chunk

# Open input and output files
with open(input_file, "r", encoding="utf-8", errors="replace") as infile, open(output_file, "w", encoding="utf-16", errors="replace") as outfile:
    # For each line, I only care about the first column (text).
    current_chunk = []
    total_lines = 0

    # Use tqdm to display progress (optional)
    for line in tqdm(infile, desc="Processing lines"):
        line = line.rstrip("\n")
        if not line:
            continue
        # Extract the first column 
        #text = line.split("\t")[3]
        text = line # for classified docs
        current_chunk.append(text)
        total_lines += 1

        # If I've collected a full chunk, it classifes and writes the results
        if len(current_chunk) >= chunk_size:
            labels = classify_texts(current_chunk)
            for text_val, label in zip(current_chunk, labels):
                outfile.write(f"{text_val}\t{label}\n")
            current_chunk = []  # clear the chunk

    # Processing any remaining lines
    if current_chunk:
        labels = classify_texts(current_chunk)
        for text_val, label in zip(current_chunk, labels):
            outfile.write(f"{text_val}\t{label}\n")

print(f"Done! Processed {total_lines} lines. Labeled data saved to '{output_file}'.")