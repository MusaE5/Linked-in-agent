import json
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data
with open("data/labeled_profiles.json", "r") as f:
    profiles = json.load(f)

X, y = [], []

print("Generating combined embeddings...")

for item in profiles:
    bio = item["bio"]
    query = item["query"]
    label = item["relevance_score"]

    bio_emb = client.embeddings.create(input=bio, model="text-embedding-ada-002").data[0].embedding
    query_emb = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

    combined = bio_emb + query_emb
    X.append(combined)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost MSE (combined embeddings): {mse:.4f}")

# Save
joblib.dump(model, "models/xgboost_model_combined_embedding.pkl")
