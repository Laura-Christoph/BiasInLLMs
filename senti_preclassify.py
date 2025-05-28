import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the Sentiberta tokenizer and model
# Replace "sentiberta" with your actual model identifier if different.
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
model.to(device)
model.eval()

# Define file paths (adjust these as needed)
input_file = "/Volumes/X10 Pro/merged_predicted_chunks.tsv"
output_file = "/Volumes/X10 Pro/sentiment_classified_posts_train.tsv"

# Open the input file for reading and the output file for writing.
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:
    
    # Create a DictReader/DictWriter for TSV files.
    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = reader.fieldnames + ["sentiment"]  # add a new column for sentiment
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    
    # Process the file line by line.
    for row in reader:
        # Assuming the text to analyze is in the "content" column.
        text = row.get("text", "")
        
        # Tokenize the text for the model.
        inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        # Move inputs to the correct device.
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference in a no_grad context to save memory.
        with torch.no_grad():
            outputs = model(**inputs)
            # Get the predicted label (assumes binary classification).
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        # Map the predicted label to a sentiment label.
        # Adjust the mapping if your model outputs different label encoding.
        sentiment = "positive" if prediction == 1 else "negative"
        row["sentiment"] = sentiment
        
        # Write the updated row to the output file.
        writer.writerow(row)

print("Finished sentiment classification. Output saved to", output_file)