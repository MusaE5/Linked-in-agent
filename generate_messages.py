import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the API key securely
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sample profiles
with open("data/top_ranked_profiles.json", "r") as f:
    profiles = json.load(f)

# Load prompt template
with open("prompts/message_template.txt", "r") as f:
    template = f.read()

# Generate and print messages
for profile in profiles:
    prompt = template.format(**profile)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    message = response.choices[0].message.content
    print(f"\n--- Message to {profile['name']} ---\n{message}\n")
