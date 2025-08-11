import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

df = pd.read_csv("sentiment.csv", encoding='latin-1')
df.columns = ['label', 'id', 'timestamp', 'flag', 'user', 'text']

print(df.head())

# convert to binary
df['label'] = df['label'].apply(lambda x: 1 if x == 0 else 0)  # Let's say 0 = depressed (1), 4 = not depressed (0)
# Basic cleaning function
def clean_text(text):
    text = str(text).lower()                              # Lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)   # Remove URLs
    text = re.sub(r'\@[\w]*', '', text)                   # Remove @mentions
    text = re.sub(r'\#[\w]*', '', text)                   # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)               # Remove special characters, numbers
    text = re.sub(r'\s+', ' ', text).strip()              # Remove extra spaces
    return text
df['clean_text'] = df['text'].apply(clean_text)
# Drop unnecessary columns
df = df[['clean_text', 'label']]

# SAMPLE 1000 ROWS ONLY
df = df.sample(n=5001, random_state=42).reset_index(drop=True)

# Preview final cleaned dataset
print(df.head())

#EMBEDDING STARTS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all 3 transformer models and tokenizers
models_info = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased"
}

tokenizers = {name: AutoTokenizer.from_pretrained(model_name) for name, model_name in models_info.items()}
models = {name: AutoModel.from_pretrained(model_name).to(device).eval() for name, model_name in models_info.items()}

# Function to get CLS embedding from one model
def get_cls_embedding(text_list, tokenizer, model):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(text_list):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
            embeddings.append(cls_embedding.numpy())
    return embeddings

# Get CLS embeddings for all 3 models
texts = df['clean_text'].tolist()
bert_cls = get_cls_embedding(texts, tokenizers['bert'], models['bert'])
roberta_cls = get_cls_embedding(texts, tokenizers['roberta'], models['roberta'])
distilbert_cls = get_cls_embedding(texts, tokenizers['distilbert'], models['distilbert'])

# Combine all embeddings into a single feature vector per text
import numpy as np

X_combined = np.hstack([bert_cls, roberta_cls, distilbert_cls])  # shape = (n_samples, 2304)
y_labels = df['label'].values

#TRAINING
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_labels, test_size=0.2, random_state=42)

# 2. Initialize XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 3. Train
xgb_model.fit(X_train, y_train)

# 4. Predict
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

#Evaluation Metrics & Confusion Matrix
# Accuracy and Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Depressed", "Depressed"], yticklabels=["Not Depressed", "Depressed"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#ROC Curve + AUC Score
# ROC AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Feature importance
plt.figure(figsize=(12, 6))
importance = xgb_model.feature_importances_
plt.hist(importance, bins=50, color='purple', edgecolor='black')
plt.title("Histogram of XGBoost Feature Importances (CLS Embeddings)")
plt.xlabel("Importance Score")
plt.ylabel("Feature Count")
plt.grid(True)
plt.show()

import joblib

# Save the trained model
joblib.dump(xgb_model, "xgb_model.pkl")
print("Model saved as xgb_model.pkl")