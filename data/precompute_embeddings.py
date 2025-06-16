import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your final dataset
with open("data/labeled_profiles.json", "r") as f:
    profiles = json.load(f)

embedded_profiles = []
for i, profile in enumerate(profiles):
    try:
        bio_text = profile["bio"]
        query_text = profile["query"]

        bio_emb = client.embeddings.create(input=bio_text, model="text-embedding-ada-002").data[0].embedding
        query_emb = client.embeddings.create(input=query_text, model="text-embedding-ada-002").data[0].embedding

        profile["bio_emb"] = bio_emb
        profile["query_emb"] = query_emb
        embedded_profiles.append(profile)

        if i % 50 == 0:
            print(f"Embedded {i}/{len(profiles)}")

        time.sleep(0.2)  # avoid hitting rate limits

    except Exception as e:
        print(f"Error at index {i}: {e}")
        continue

# Save the new dataset
with open("data/labeled_profiles_embedded.json", "w") as f:
    json.dump(embedded_profiles, f, indent=2)

print("Saved 1100 embedded profiles to labeled_profiles_embedded.json")
