import openai
import json
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # to save model

# Load API key securely

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#Load training data

with open("data/labeled_profiles.json", "r") as f:
    data = json.load(f)

X = []
y= []

print('Generating combined embeddings')

for item in data:
    bio = item["bio"]
    query = item["query"]
    label = item["relevance_score"]

    #Get query embeddings
    query_embeddings = openai.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding

    bio_embeddings = openai.embeddings.create(
        input = bio,
        model="text-embedding-ada-002"
    ).data[0].embedding

    combined = bio_embeddings + query_embeddings
    X.append(combined)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train random forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n MSE (combined embeddings): {mse:.4f}")
joblib.dump(model, "models/rf_model_combined_embedding.pkl")
