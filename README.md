Depression Prediction from Social Media Text

This project is our attempt to explore how machine learning can help detect signs of depression in social media posts.  
It combines **Transformer-based embeddings** (BERT, RoBERTa, and DistilBERT) with an **XGBoost meta-classifier** to make predictions.  
The idea is simple: if text carries subtle emotional cues, can we teach a model to pick them up?

Project Structure

| File | Purpose |
|------|---------|
| `sentiment.csv` | Dataset used for training and testing the model |
| `embedding.py` | Cleans the text and generates embeddings from Transformer models |
| `test.py` | Loads the trained model and tests predictions on sample inputs |
| `xgb_model.pkl` | Saved XGBoost model (trained on embeddings) |

How It Works

1. **Preprocessing** – The text is cleaned, tokenized, and prepared.
2. **Embedding Extraction** – Each post is passed through BERT, RoBERTa, and DistilBERT to get numerical representations (CLS tokens).
3. **Meta-classifier** – The embeddings are concatenated and fed into XGBoost, which makes the final depression/non-depression prediction.
