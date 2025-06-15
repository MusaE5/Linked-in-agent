import json
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data
with open("data/labeled_profiles.json", "r") as f:
    profiles = json.load(f)

X, y = [], []

print("Generating separate feature vectors...")

for item in profiles:
    bio = item["bio"]
    query = item["query"]
    label = item["relevance_score"]

    # Get embeddings
    bio_emb = client.embeddings.create(input=bio, model="text-embedding-ada-002").data[0].embedding
    query_emb = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

    # Compute cosine similarity as a scalar feature
    sim = cosine_similarity([query_emb], [bio_emb])[0][0]

    # Build feature vector: [cosine_sim, *query_embedding, *bio_embedding]
    features = [sim] + query_emb + bio_emb
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nXGBoost MSE (separate features): {mse:.4f}")

# Save model
joblib.dump(model, "models/xgboost_cosine_feature.pkl")
