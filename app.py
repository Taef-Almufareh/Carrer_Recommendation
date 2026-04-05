import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ---------------- UI ----------------
st.set_page_config(page_title="Career Recommender", layout="wide")
st.title("🎓 Career Recommendation System (AI Powered)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("career_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, preprocessor, label_encoder, feature_cols

model, preprocessor, label_encoder, feature_cols = load_assets()

st.success("Model loaded successfully!")

# ---------------- INPUT UI ----------------
st.subheader("Enter Student Profile")

user_input = {}

for col in feature_cols:
    user_input[col] = st.text_input(col, "0")

if st.button("Predict Career"):
    df = pd.DataFrame([user_input])

    # convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    X = preprocessor.transform(df)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    probs = model.predict(X)[0]
    top_idx = np.argmax(probs)

    st.success(f"🎯 Recommended Career: {label_encoder.classes_[top_idx]}")

    result = pd.DataFrame({
        "Career": label_encoder.classes_,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    st.dataframe(result)
