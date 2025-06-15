import json
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



# Load labeled profiles
with open("data/labeled_profiles.json",'r')as f:
    profiles = json.load(f)
bios = [p["bio"] for p in profiles]
labels = [p["relevance_score"] for p in profiles]

# Get embeddings for all bios
print("getting embeddings...")
bio_embeddings = []

for bio in bios:
    response = openai.embeddings.create(
        input = bio,
        model = "text-embedding-ada-002"
    )
    bio_embeddings.append(response.data[0].embedding)

x = np.array(bio_embeddings)
y = np.array(labels)

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

knn = KNeighborsRegressor(n_neighbors = 3, metric = "cosine")
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.4f}")
# Save trained model to disk
joblib.dump(knn, "models/knn_vanilla_model.pkl")
print(" Model saved to models/knn_model.pkl")