import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


print("Loading clinical NLP model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("Clinical NLP model loaded!")


text = input(
    "\nEnter clinical information: "
)


inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256
)


with torch.no_grad():

    outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]


print("\nClinical embedding generated")
print("Shape:", embedding.shape)
print("Embedding size:", embedding.shape[-1])


torch.save(
    embedding,
    "data/nlp/clinical_features.pt"
)


print("\nSaved:")
print("data/nlp/clinical_features.pt")