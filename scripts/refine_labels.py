import json
import os
from pathlib import Path
from dotenv import load_dotenv
import openai
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
REFINED_LABELS = Path("data/refined_labels.json")
LABELED_PROFILES = Path("data/labeled_profiles.json")

def gpt4_score(query, bio):
    """Get a refined relevance score (1-5) from GPT-4 with more robust parsing."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Respond ONLY with a single number 1-5 to rate relevance of the bio to the query.\n"
                           "1: Irrelevant\n3: Partial match\n5: Perfect match\n"
                           "Focus on role alignment, skills, and location."
            },
            {
                "role": "user", 
                "content": f"Query: '{query}'\nBio: '{bio}'"
            }
        ],
        temperature=0.1
    )
    
    # Robust response parsing
    try:
        content = response.choices[0].message.content.strip()
        if content.isdigit() and 1 <= int(content) <= 5:
            return int(content)
        # Handle cases where GPT adds explanation
        if ':' in content:
            return int(content.split(':')[0])
    except (ValueError, AttributeError):
        pass
    
    # Fallback for malformed responses
    print(f"Warning: GPT returned invalid score '{content}'. Using default score 3.")
    return 3

def main():
    # Load existing refined labels
    refined = {}
    if REFINED_LABELS.exists():
        with open(REFINED_LABELS, "r") as f:
            refined = json.load(f)

    # Load all profiles
    with open(LABELED_PROFILES, "r") as f:
        profiles = json.load(f)

    # Smarter profile selection - 20 per query type
    query_groups = defaultdict(list)
    for idx, profile in enumerate(profiles):
        query_groups[profile["query"]].append((idx, profile))

    profiles_to_refine = []
    for query, group in query_groups.items():
        # Skip queries where we already have 20 refined samples
        existing = sum(1 for k in refined.keys() if profiles[int(k)]["query"] == query)
        needed = max(0, 20 - existing)
        profiles_to_refine.extend(group[:needed])

    # Process selected profiles
    for idx, profile in profiles_to_refine[:200]:  # Safety cap
        refined[str(idx)] = gpt4_score(profile["query"], profile["bio"])
        print(f"Processed profile {len(refined)}/{min(200, len(profiles_to_refine))}")

    # Save refined labels
    with open(REFINED_LABELS, "w") as f:
        json.dump(refined, f)

if __name__ == "__main__":
    main()