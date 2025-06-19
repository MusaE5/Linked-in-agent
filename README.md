⚠️ **Legacy Repo Notice**

This repository contains early experiments and modeling tests for my LinkedIn Agent project, using synthetic data and simple baselines (KNN, Random Forest, etc.).

🚀 The **active, production-grade version** using real LinkedIn data, XGBoostRanker, and GPT messaging is now here: [linkedin-agent-ml](https://github.com/MusaE5/linkedin-agent-ml)

This repo is archived for reference only.


# LinkedIn AI Agent — Intelligent Outreach & Ranking System

An AI-powered system that helps users discover, rank, and generate personalized outreach messages for relevant LinkedIn-style profiles, based on custom career goals.

This project combines:
- OpenAI Embeddings for semantic understanding
- ML models for relevance prediction
- GPT-4o for message generation
- A modular backend (React frontend planned)

---

## Key Features

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| ML Scoring Model       | Trained KNN and Random Forest models on OpenAI embeddings                   |
| Relevance Ranking      | Predicts how relevant each profile is to a user query                       |
| GPT-4 Messaging Engine | Generates customized messages for top-ranked profiles                       |
| Frontend (Planned)     | React or Streamlit UI for input, ranking, message viewing, and export       |

---

## Use Case

A user inputs a career goal (e.g., "AI internships in Toronto"). The system:

1. Scores LinkedIn-style profiles based on relevance
2. Ranks them using a trained ML model
3. Generates personalized outreach messages with GPT-4
4. Returns a list of top profiles and their corresponding messages

---

## Project Structure

```
linkedin-agent/
├── data/
│   ├── labeled_profiles.json               # Full dataset with query, bio, and raw labels
│   ├── labeled_profiles_embedded.json      # Same dataset with OpenAI embeddings (bio_emb, query_emb)
│   ├── refined_labels.json                 # Refined labels for some profiles (indexed by ID)
│   └── top_ranked_profiles.json            # Output: top-ranked profiles based on user query
│
├── models/
│   ├── knn_baseline.py                     # KNN with raw bio embeddings
│   ├── knn_pca.py                          # KNN with PCA-reduced embeddings
│   ├── knn_combined_embedding.py           # KNN on concatenated query+bio embeddings
│   ├── rf_combined_embedding.py            # Random Forest on combined embeddings
│
│   ├── xgboost_cosine_feature.py           # XGBoost using only cosine similarity + handcrafted features
│   ├── xgboost_with_features.py            # XGBoost with all scalar features and basic query/bio embeddings
│   ├── xgboost_only_scalars.py             # XGBoost trained using only scalar features (no embeddings)
│   ├── xgboost_ranker_scalar_pca16.py      # XGBRanker with PCA(16)-reduced embeddings + scalar features + refined labels
│
├── scripts/
│   ├── precompute_embeddings.py            # Embed bios and queries using text-embedding-ada-002
│   └── refine_labels.py                    # Re-score selected samples using GPT-4 for better label quality
│
├── prompts/
│   └── message_template.txt                # Prompt template used by GPT-4 to generate outreach messages
│
├── generate_messages.py                    # Uses GPT-4 to write personalized messages for top profiles
├── select_top_profiles.py                  # Selects top-ranked profiles for a new user query
├── requirements.txt
└── README.md

```


## 📊 Model Performance


All models use OpenAI’s `text-embedding-ada-002` (1536-dim) for vectorization. Some models use raw embeddings directly, while others extract scalar features such as cosine similarity, keyword overlap, location match, or title seniority. Recent models combine these with dimensionality reduction and refined labels.

---

###  Small Dataset (~200 Profiles)

| Model         | Features Used                                                | MSE   |
|---------------|--------------------------------------------------------------|-------|
| KNN           | Raw bio embeddings                                           | 2.30  |
| KNN           | PCA-reduced bio embeddings                                   | 2.183 |
| KNN           | Combined query + bio embeddings                              | 2.10  |
| Random Forest | Combined query + bio embeddings                              | 2.04  |
| XGBoost       | Scalar + full embeddings + cosine similarity                 | 2.55  |
| XGBoost       | Query & bio embeddings (separate) + cosine similarity        | 2.33  |

---

###  Medium Dataset (~1100 Profiles)

| Model    | Features Used                                   | MSE   | Precision@5 |
|----------|--------------------------------------------------|-------|--------------|
| XGBoost  | Scalar-only (cos sim, overlap, location, title) | 0.396 | 0.20         |

---

###  Large Dataset (2000 Profiles)

| Model        | Features Used                                                                   | Precision@5 |
|--------------|----------------------------------------------------------------------------------|--------------|
| XGBRanker    | PCA(16)-reduced embeddings + scalar features (cos sim, overlap, loc, title) + refined labels | 0.13         |


---

### 📌 Notes

- **Scalar-only models** use hand-engineered features:
  - Cosine similarity between query and bio embeddings
  - Keyword overlap between query and bio
  - Location match between query-inferred city and profile location
  - Title seniority scores based on keywords (e.g., "intern", "student")

- **Combined embeddings** = `[query_embedding + bio_embedding]` → 3072-dim input to ML models

- **PCA-reduced models** apply dimensionality reduction to embeddings (e.g., 16D each for query and bio) to reduce noise and speed up training

- All models are trained as **regressors** or **rankers** using:
  - **MSE (Mean Squared Error)** to measure prediction error on relevance scores (1–5)
  - **Precision@5**: Fraction of top 5 predicted profiles for each query that have `relevance_score ≥ 4`

- Some labels are refined using **GPT-4** to improve supervision quality on a subset of profiles



---

## How It Works

### Phase 1: Model Training

- Each profile includes a `relevance_score` (1–5) associated with a specific user query
- Profile bios and queries are embedded using OpenAI’s `text-embedding-ada-002`
- Embeddings are combined (bio + query) and used to train ML models
- Experiments included:
  - Vanilla KNN
  - PCA-reduced KNN
  - Combined embedding (bio + query) with KNN and Random Forest

### Phase 2: Inference Pipeline

- A new user query is embedded
- Each profile bio is embedded and combined with the query embedding
- Combined vectors are passed into the trained model
- Model predicts relevance scores for all profiles
- Top N profiles are saved to `top_ranked_profiles.json`

### Phase 3: Message Generation (In Progress)

- GPT-4 generates personalized outreach messages using a prompt template
- Messages are generated for the top N profiles returned by the model
- Output includes: profile name, message content, and relevance score

### Phase 4: UI & Agent Behavior (Planned)

- Streamlit or React frontend for input, ranking, message viewing, and export
- Optional filters by location, title level, and domain
- Agent-like flow:
  1. User enters goal
  2. System ranks and selects top profiles
  3. Messages are auto-generated
  4. User reviews and exports

---

## Next Milestones

- Feature engineering:
  - Title seniority mapping
  - Keyword/skill overlap
- Try XGBoost model
- Add Precision@K and Top-N evaluation metrics
- Build Streamlit or React frontend
- Message quality scoring and feedback loop
- Export (CSV or clipboard)

---

## Notes

- All profiles are synthetic, generated to simulate real LinkedIn bios
- No scraping or LinkedIn automation is used
- This system is being developed as a prototype intelligent agent for job search and networking
