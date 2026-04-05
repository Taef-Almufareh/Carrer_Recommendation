import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Career Recommender", layout="wide")

st.title("🎓 Hybrid Career Recommendation System")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file, sheet_name="Data")
    else:
        data = pd.read_csv(uploaded_file)

    st.success("File loaded successfully!")
    st.write("Shape:", data.shape)

    data = data.drop_duplicates().dropna().reset_index(drop=True)

    target_col = "career"
    id_col = "student_id"

    feature_cols = [c for c in data.columns if c not in [target_col, id_col]]

    X = data[feature_cols].copy()
    y = data[target_col].copy()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)

    n_features = X_train_dense.shape[1]
    n_classes = len(label_encoder.classes_)

    st.subheader("Model Training")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    with st.spinner("Training model..."):
        history = model.fit(
            X_train_dense, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

    st.success("Training completed!")

    preds = np.argmax(model.predict(X_test_dense), axis=1)

    acc = accuracy_score(y_test, preds)

    st.metric("Test Accuracy", f"{acc:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, preds, target_names=label_encoder.classes_))

    st.subheader("Training Curves")

    fig1, ax1 = plt.subplots()
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Predict Career for New Student")

    input_data = {}

    for col in feature_cols:
        if col in numeric_cols:
            input_data[col] = st.number_input(col, value=0.0)
        else:
            input_data[col] = st.text_input(col, "unknown")

    if st.button("Predict Career"):
        new_df = pd.DataFrame([input_data])[feature_cols]
        new_encoded = preprocessor.transform(new_df)
        new_dense = new_encoded.toarray() if hasattr(new_encoded, "toarray") else np.asarray(new_encoded)

        prob = model.predict(new_dense)[0]
        top_idx = np.argmax(prob)

        st.success(f"🎯 Recommended Career: {label_encoder.classes_[top_idx]}")

        result_df = pd.DataFrame({
            "Career": label_encoder.classes_,
            "Probability": prob
        }).sort_values("Probability", ascending=False)

        st.dataframe(result_df)
