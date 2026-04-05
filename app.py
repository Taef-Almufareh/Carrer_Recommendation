import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
tf.random.set_seed(42)


from google.colab import files

uploaded = files.upload()
file_name = next(iter(uploaded.keys()))

if file_name.lower().endswith(".xlsx"):
    data = pd.read_excel(file_name, sheet_name="Data")
elif file_name.lower().endswith(".csv"):
    data = pd.read_csv(file_name)
else:
    raise ValueError("Please upload either an .xlsx or .csv file")

print("Loaded file:", file_name)
print("Shape:", data.shape)
data.head()

data = data.drop_duplicates().dropna().reset_index(drop=True)

target_col = "career"
id_col = "student_id"

if target_col not in data.columns:
    raise ValueError("Target column 'career' was not found in the dataset.")

feature_cols = [c for c in data.columns if c not in [target_col, id_col]]

X = data[feature_cols].copy()
y = data[target_col].copy()

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
print("Target classes:", sorted(y.unique()))

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

print("Encoded feature size:", n_features)
print("Number of classes:", n_classes)


ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(n_classes, activation="softmax")
])

ann_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

ann_model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = ann_model.fit(
    X_train_dense,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)


ann_probs_test = ann_model.predict(X_test_dense, verbose=0)
ann_preds_test = np.argmax(ann_probs_test, axis=1)

print("ANN Accuracy:", accuracy_score(y_test, ann_preds_test))
print()
print(classification_report(y_test, ann_preds_test, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, ann_preds_test))

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training vs Validation Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training vs Validation Loss")
plt.legend()
plt.show()


career_profile_matrix = []
for class_id in range(n_classes):
    class_mean = X_train_dense[y_train == class_id].mean(axis=0)
    career_profile_matrix.append(class_mean)

career_profile_matrix = np.vstack(career_profile_matrix)

def content_based_proba(student_vector):
    sims = cosine_similarity(student_vector.reshape(1, -1), career_profile_matrix)[0]
    sims = np.maximum(sims, 0)

    if sims.sum() == 0:
        probs = np.ones(n_classes) / n_classes
    else:
        probs = sims / sims.sum()
    return probs

def collaborative_proba(student_vector, k=7):
    sims = cosine_similarity(student_vector.reshape(1, -1), X_train_dense)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    top_sims = np.maximum(sims[top_idx], 0)

    probs = np.zeros(n_classes)
    for idx, sim in zip(top_idx, top_sims):
        probs[y_train[idx]] += sim

    if probs.sum() == 0:
        counts = np.bincount(y_train, minlength=n_classes)
        probs = counts / counts.sum()
    else:
        probs = probs / probs.sum()

    return probs


ANN_WEIGHT = 0.50
CB_WEIGHT = 0.25
CF_WEIGHT = 0.25

def hybrid_proba(student_vector,
                 ann_weight=ANN_WEIGHT,
                 cb_weight=CB_WEIGHT,
                 cf_weight=CF_WEIGHT):
    ann_probs = ann_model.predict(student_vector.reshape(1, -1), verbose=0)[0]
    cb_probs = content_based_proba(student_vector)
    cf_probs = collaborative_proba(student_vector)

    final_probs = ann_weight * ann_probs + cb_weight * cb_probs + cf_weight * cf_probs
    final_probs = final_probs / final_probs.sum()

    return final_probs, ann_probs, cb_probs, cf_probs

hybrid_preds = []
for row in X_test_dense:
    probs, _, _, _ = hybrid_proba(row)
    hybrid_preds.append(np.argmax(probs))

hybrid_preds = np.array(hybrid_preds)

print("ANN Accuracy   :", accuracy_score(y_test, ann_preds_test))
print("Hybrid Accuracy:", accuracy_score(y_test, hybrid_preds))
print()
print(classification_report(y_test, hybrid_preds, target_names=label_encoder.classes_))


print("Required input columns:")
print(feature_cols)

new_student = {
    "programming": 88,
    "algorithms": 84,
    "databases": 76,
    "networks": 62,
    "software_engineering": 90,
    "machine_learning": 68,
    "security": 60,
    "problem_solving": 4.5,
    "programming_tools": 4.6,
    "adaptability": 4.2,
    "teamwork": 4.1,
    "communication": 3.9,
    "leadership": 3.5,
    "preferred_environment": "private sector",
    "preferred_role_area": "software",
    "work_style": "builder",
    "gpa_equivalent": 4.10
}

new_student_df = pd.DataFrame([new_student])[feature_cols]
new_student_encoded = preprocessor.transform(new_student_df)
new_student_dense = new_student_encoded.toarray() if hasattr(new_student_encoded, "toarray") else np.asarray(new_student_encoded)

hybrid_probs, ann_probs, cb_probs, cf_probs = hybrid_proba(new_student_dense[0])

results = pd.DataFrame({
    "career": label_encoder.classes_,
    "hybrid_probability": hybrid_probs,
    "ann_probability": ann_probs,
    "content_based_score": cb_probs,
    "collaborative_score": cf_probs
}).sort_values("hybrid_probability", ascending=False)

results

top_recommendation = results.iloc[0]["career"]
print("Top Recommended Career:", top_recommendation)
print()
print("Top 3 Recommendations:")
display(results.head(3))



ann_model.save("career_hybrid_ann_model.keras")
joblib.dump(preprocessor, "career_preprocessor.pkl")
joblib.dump(label_encoder, "career_label_encoder.pkl")

print("Saved:")
print("- career_hybrid_ann_model.keras")
print("- career_preprocessor.pkl")
print("- career_label_encoder.pkl")
