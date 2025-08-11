import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib

# Load XGBoost model
xgb_model = joblib.load("xgb_model.pkl")

# Load Transformers
tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
model_bert = AutoModel.from_pretrained('bert-base-uncased')

tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
model_roberta = AutoModel.from_pretrained('roberta-base')

tokenizer_distilbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_distilbert = AutoModel.from_pretrained('distilbert-base-uncased')

# Prediction Function
def predict_depression(text):
    with torch.no_grad():
        inputs_bert = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs_bert = model_bert(**inputs_bert)
        cls_bert = outputs_bert.last_hidden_state[:, 0, :].numpy()

        inputs_roberta = tokenizer_roberta(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs_roberta = model_roberta(**inputs_roberta)
        cls_roberta = outputs_roberta.last_hidden_state[:, 0, :].numpy()

        inputs_distilbert = tokenizer_distilbert(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs_distilbert = model_distilbert(**inputs_distilbert)
        cls_distilbert = outputs_distilbert.last_hidden_state[:, 0, :].numpy()

    # Concatenate
    final_embedding = np.concatenate([cls_bert, cls_roberta, cls_distilbert], axis=1)

    prediction = xgb_model.predict(final_embedding)
    return "Depressed" if prediction[0] == 1 else "Not Depressed"

# samples
if __name__ == "__main__":
    samples = [
    "I’m fine. Just tired of everything, I guess.",
    "The sun is out, and I feel alive today!",
    "No one really understands me. I feel so alone.",
    "So so excited for the weekend trip with my cousins!",
    "It was the worst coffee beverage of my entire life",
    "Movie night with my favorite people!",
    "Some days I can’t even get out of bed, but I try.",
    "Life feels like a never-ending loop of disappointments.",
    "Grateful for small wins and sunny days",
    "I cried so much today my eyes hurt!",
    "Everything is falling apart and I don’t know how to fix it.",
    "Life is not perfect, but today felt pretty close",
    "Sometimes I wish I could disappear.",
    "Tried something new today and I am proud of myself ",
    "My anxiety is eating me from the inside.",
    "GOODLUCK MAN",
    "Lately I feel like I'm drowning but no one sees it.",
    "Amazing work done by the team.",
    "I feel numb more often than I’d like to admit.",
    "Woke up feeling fresh and motivated!"
]

for i, text in enumerate(samples, 1):
    print(f"Sample {i}: {predict_depression(text)}")
