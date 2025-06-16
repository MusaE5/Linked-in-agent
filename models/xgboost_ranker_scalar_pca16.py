import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import joblib

# ---------------- Paths ----------------
EMBEDDED = Path("data/labeled_profiles_embedded.json")
REFINED  = Path("data/refined_labels.json")
OUTPUT_MODEL = Path("models/xgboost_ranker_scalar_pca16_refined.pkl")

# ---------------- Load Data ----------------
profiles = json.loads(EMBEDDED.read_text())

# ---------------- Load Refined Labels ----------------
# refined is a dict mapping string indices → new relevance scores
if REFINED.exists():
    refined = json.loads(REFINED.read_text())
    for idx_str, score in refined.items():
        idx = int(idx_str)
        profiles[idx]["relevance_score"] = score

# ---------------- Extract Embeddings for PCA ----------------
bio_embeddings   = np.array([p["bio_emb"]   for p in profiles])
query_embeddings = np.array([p["query_emb"] for p in profiles])

# ---------------- PCA Setup ----------------
pca_bio   = PCA(n_components=16, random_state=42)
pca_query = PCA(n_components=16, random_state=42)
bio_pca   = pca_bio.fit_transform(bio_embeddings)
query_pca = pca_query.fit_transform(query_embeddings)

# ---------------- Feature Engineering Helpers ----------------
def keyword_overlap(query, bio):
    return len(set(query.lower().split()) & set(bio.lower().split()))

def extract_query_location(query):
    m = re.search(r"in\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    return m.group(1).strip().lower() if m else ""

def location_match(query_loc, profile_loc):
    return int(bool(query_loc and query_loc in profile_loc.lower()))

def title_seniority_score(bio):
    for kw in ("student", "intern", "junior", "assistant"):
        if kw in bio.lower():
            return 1
    return 0

def precision_at_k_per_query(profiles, y_true, y_pred, k=5, threshold=4):
    groups = defaultdict(list)
    for i, p in enumerate(profiles):
        groups[p["query"]].append(i)
    precisions = []
    for idxs in groups.values():
        topk = sorted(idxs, key=lambda i: y_pred[i], reverse=True)[:k]
        precisions.append(sum(1 for i in topk if y_true[i] >= threshold) / k)
    return np.mean(precisions)

# ---------------- Build Feature Matrix ----------------
X, y, group_sizes = [], [], []
current_query, count = None, 0

for i, p in enumerate(profiles):
    bio, query, label = p["bio"], p["query"], p["relevance_score"]
    loc = p.get("location", "")

    bio_feat   = bio_pca[i]
    query_feat = query_pca[i]

    sim       = cosine_similarity([query_feat], [bio_feat])[0][0]
    overlap   = keyword_overlap(query, bio)
    qloc      = extract_query_location(query)
    loc_feat  = location_match(qloc, loc)
    title_feat= title_seniority_score(bio)

    features = [sim, overlap, loc_feat, title_feat] \
               + query_feat.tolist() + bio_feat.tolist()
    X.append(features)
    y.append(label)

    if query != current_query:
        if current_query is not None:
            group_sizes.append(count)
        current_query = query
        count = 1
    else:
        count += 1

group_sizes.append(count)

X = np.array(X)
y = np.array(y)

# ---------------- Train & Evaluate Ranker ----------------
model = xgb.XGBRanker(
    objective="rank:ndcg",
    learning_rate=0.05,
    n_estimators=200,
    max_depth=4
)
model.fit(X, y, group=group_sizes)

y_pred = model.predict(X)
avg_p5 = precision_at_k_per_query(profiles, y, y_pred)
print(f"Precision@5 with PCA-16 + refined labels: {avg_p5:.2f}")

# ---------------- Save Model ----------------
joblib.dump(model, OUTPUT_MODEL)
print(f"Model saved to {OUTPUT_MODEL}")
