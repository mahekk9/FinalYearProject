import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib

# ----------------- MUST BE FIRST STREAMLIT COMMAND -----------------
st.set_page_config(page_title="Depression Prediction", page_icon="💭", layout="centered")

# Load trained XGBoost model
xgb_model = joblib.load("xgb_model.pkl")

# Load Transformer models & tokenizers (load once)
@st.cache_resource
def load_models():
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")

    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_model = AutoModel.from_pretrained("roberta-base")

    distil_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distil_model = AutoModel.from_pretrained("distilbert-base-uncased")

    return bert_tokenizer, bert_model, roberta_tokenizer, roberta_model, distil_tokenizer, distil_model

bert_tokenizer, bert_model, roberta_tokenizer, roberta_model, distil_tokenizer, distil_model = load_models()

# Function: get CLS embedding
def get_cls_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Function: predict depression
def predict_depression(text):
    cls_bert = get_cls_embedding(text, bert_tokenizer, bert_model)
    cls_roberta = get_cls_embedding(text, roberta_tokenizer, roberta_model)
    cls_distil = get_cls_embedding(text, distil_tokenizer, distil_model)

    # Concatenate embeddings
    final_embedding = np.concatenate([cls_bert, cls_roberta, cls_distil], axis=1)

    prediction = xgb_model.predict(final_embedding)
    return "Hey, these words feel a little heavy.☹️ Maybe you are carrying sadness inside. It is okay💛, dark clouds do pass🌫️, and sunshine comes back✨. Sending a warm hug from here!💙" if prediction[0] == 1 else "Haha, this post is sparkling with joy!🤩 Feels like someone who just got free pizza🍕 and a day off.😌 The model says — not depressed at all.🥳 Carry on shining, star 🌟"

# ----------------- STREAMLIT UI -----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #CAEEF9;'>Depression Prediction Model</h1>
    <p style='text-align: center; font-style: italic; color: #CAEEF9;'>Using Machine Learning</p>
    <hr style='border: 2px dashed white;'>
    """,
    unsafe_allow_html=True,
)


# Input area
st.markdown("### ✍🏻Input your text/post")
user_input = st.text_area("", height=150, placeholder="Type or paste a social media post here...")

# Analyze button
if st.button("🧐Analyse"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analysing.")
    else:
        result = predict_depression(user_input)
        st.markdown("## 🔮 Prediction")
        st.success(result)
