import json
import os
import re
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

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

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data
with open("data/labeled_profiles.json", "r") as f:
    profiles = json.load(f)

X, y = [], []

print("Generating features and embeddings...")

for p in profiles:
    bio, query, label = p["bio"], p["query"], p["relevance_score"]
    profile_loc = p.get("location", "")

    # Embeddings
    bio_emb = client.embeddings.create(input=bio, model="text-embedding-ada-002").data[0].embedding
    query_emb = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

    # Features
    sim = cosine_similarity([query_emb], [bio_emb])[0][0]
    overlap = keyword_overlap(query, bio)
    qloc = extract_query_location(query)
    loc_feat = location_match(qloc, profile_loc)
    title_feat = title_seniority_score(bio)

    # Combine into one long vector
    features = [sim, overlap, loc_feat, title_feat]
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split, train, evaluate, save
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\nXGBoost MSE (with features): {mean_squared_error(y_test, y_pred):.4f}")
joblib.dump(model, "models/xgboost_only_scalars.pkl")
