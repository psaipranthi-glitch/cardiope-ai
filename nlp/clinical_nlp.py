import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

print("Loading ClinicalBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("Clinical NLP model loaded!")


def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS embedding
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding


while True:

    text = input("\nEnter clinical information (or 'exit'): ")

    if text.lower() == "exit":
        break

    embedding = get_embedding(text)

    print("\nClinical embedding generated")
    print("Shape:", embedding.shape)
    print("Embedding size:", embedding.shape[-1])