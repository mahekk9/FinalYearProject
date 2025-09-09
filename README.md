**Depression Prediction from Social Media Text**

This project explores how **machine learning** can help detect signs of depression in social media posts.
It combines **Transformer-based embeddings** (BERT, RoBERTa, and DistilBERT) with an **XGBoost meta-classifier** to make predictions.
The goal is simple: if text carries subtle emotional cues, can we teach a model to pick them up?


Project Structure

| File            | Purpose                                                          |
| --------------- | ---------------------------------------------------------------- |
| `sentiment.csv` | Dataset used for training and testing the model                  |
| `embedding.py`  | Cleans the text and generates embeddings from Transformer models |
| `test.py`       | Loads the trained model and tests predictions on sample inputs   |
| `xgb_model.pkl` | Saved XGBoost model (trained on embeddings)                      |
| `app.py`        | Streamlit-based frontend for interactive depression prediction   |


How It Works

1. **Preprocessing** – The text is cleaned, tokenized, and prepared for embedding extraction.
2. **Embedding Extraction** – Each post is passed through **BERT, RoBERTa, and DistilBERT** to get numerical representations (CLS tokens).
3. **Meta-classifier** – The embeddings are concatenated and fed into **XGBoost**, which predicts whether a post shows signs of depression.
4. **Frontend (app.py)** – Users can input text in a simple **Streamlit interface** to get real-time predictions.

Usage

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Enter your social media text in the input box and see if the model predicts depression signs.


## Notes

* This project is for **educational purposes** and not a substitute for professional mental health advice.
* Accuracy depends on the dataset (`sentiment.csv`) and model training. Real-world social media posts can be more nuanced.
