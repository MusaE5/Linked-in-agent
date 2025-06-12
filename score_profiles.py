import openai
import json
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load profile data
with open("data/sample_profiles.json", "r") as f:
    profiles = json.load(f)

# User query (will be input to the model later)
user_query =  "Looking for machine learning internships in Toronto"   

query_embedding = openai.embeddings.create(
    input = user_query,
    model = "text-embedding-ada-002"
).data[0].embedding

profile_scores = []

for profile in profiles:
    profile_text = profile["about"]

    profile_embedding = openai.embeddings.create(
        input = profile_text,
        model = "text-embedding-ada-002"
    ).data[0].embedding

    similarity = cosine_similarity(
        [query_embedding],[profile_embedding]

    )[0][0]

    profile_scores.append((profile["name"], similarity))

profile_scores.sort(key=lambda x: x[1], reverse = True)

print("\n Ranked Profiles by Relevance:\n")
for name, score in profile_scores:
    print(f"{name}: {score:.4f}")