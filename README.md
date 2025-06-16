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
│   ├── labeled_profiles.json         # Labeled training data
│   └── top_ranked_profiles.json      # Filtered profiles for messaging
├── prompts/
│   └── message_template.txt          # GPT-4 prompt template
├── models/
│   ├── knn_baseline.py               # KNN with raw embeddings
│   ├── knn_pca.py                    # KNN with PCA-reduced embeddings
│   ├── knn_combined_embedding.py     # KNN with combined query + bio embeddings
│   ├── rf_combined_embedding.py      # Random Forest with combined embeddings
├── generate_messages.py              # GPT-4 messaging from top results
├── select_top_profiles.py            # Ranks profiles for a new user query
├── requirements.txt
└── README.md
```


## 📊 Model Performance

All models use OpenAI’s `text-embedding-ada-002` (1536-dim) for vectorization. Some models use full embeddings directly, while others extract scalar features such as cosine similarity, keyword overlap, or location match.

---

###  Small Dataset (≈200 profiles)

| Model         | Features Used                                       | MSE   |
|---------------|------------------------------------------------------|-------|
| KNN           | Raw bio embeddings                                  | 2.30  |
| KNN           | PCA-reduced bio embeddings                          | 2.183 |
| KNN           | Combined query + bio embeddings                     | 2.10  |
| Random Forest | Combined query + bio embeddings                     | 2.04  |
| XGBoost       | Scalar + full embeddings + cosine similarity        | 2.55  |
| XGBoost       | Query & bio embeddings (separate) + cosine sim      | 2.33  |
| XGBoost       | Scalar-only (cos sim, overlap, location, title)     | 2.40  |

---

###  Large Dataset (1100 profiles)

| Model    | Features Used                                   | MSE   | Precision@5 |
|----------|--------------------------------------------------|-------|--------------|
| XGBoost  | Scalar-only (cos sim, overlap, location, title) | 0.396 | 0.20         |

---

### 📌 Notes

- **Scalar-only models** include features like:
  - Cosine similarity
  - Keyword overlap
  - Location match
  - Title seniority scores

- **Combined embeddings** = `[query_embedding + bio_embedding]` → 3072-dim vector

- All models are trained as **regressors** and evaluated using:
  - **MSE** (Mean Squared Error)
  - **Precision@5**: Fraction of top 5 ranked profiles with `relevance_score ≥ 4`



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
