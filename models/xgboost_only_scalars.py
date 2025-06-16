import json
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Feature Engineering ----------------

def keyword_overlap(query, bio):
    return len(set(query.lower().split()) & set(bio.lower().split()))

def extract_query_location(query):
    m = re.search(r"in\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    return m.group(1).strip().lower() if m else ""

def location_match(query_loc, profile_loc):
    if not query_loc:
        return 0
    return int(query_loc in profile_loc.lower())

def title_seniority_score(bio):
    for kw in ["student", "intern", "junior", "entry level"]:
        if kw in bio.lower():
            return 1
    return 0

def precision_at_k(y_true, y_pred, k=5, threshold=4):
    top_k_indices = np.argsort(y_pred)[-k:]
    top_k_true = [y_true[i] for i in top_k_indices]
    relevant = [1 if score >= threshold else 0 for score in top_k_true]
    return np.mean(relevant)

# ---------------- Load Precomputed Data ----------------

with open("data/labeled_profiles_embedded.json", "r") as f:
    profiles = json.load(f)

X, y = [], []

print("Building feature vectors from precomputed embeddings...")

for p in profiles:
    bio = p["bio"]
    query = p["query"]
    label = p["relevance_score"]
    profile_loc = p.get("location", "")

    bio_emb = np.array(p["bio_emb"])
    query_emb = np.array(p["query_emb"])

    sim = cosine_similarity([query_emb], [bio_emb])[0][0]
    overlap = keyword_overlap(query, bio)
    qloc = extract_query_location(query)
    loc_feat = location_match(qloc, profile_loc)
    title_feat = title_seniority_score(bio)

    features = [sim, overlap, loc_feat, title_feat]
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# ---------------- Train & Evaluate ----------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nXGBoost MSE (with features): {mean_squared_error(y_test, y_pred):.4f}")
print(f"Precision@5: {precision_at_k(y_test, y_pred, k=5, threshold=4):.2f}")

joblib.dump(model, "models/xgboost_only_scalars.pkl")
print(" Model saved to models/xgboost_only_scalars.pkl")
