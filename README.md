#  LinkedIn AI Agent — Intelligent Outreach & Ranking System

A machine learning-powered agent that helps users discover and message relevant LinkedIn profiles based on custom career goals.

Built using:
- OpenAI Embeddings + GPT-4o for smart message generation
- A trained ML model (KNN + PCA) for profile relevance scoring
- Real-world LinkedIn-style profile data
- Modular backend with planned React-based frontend

---

##  Key Features

| Component                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
|  ML Scoring Model       | Uses KNN + PCA on OpenAI embeddings to rank profiles based on a user query |
|  Relevance Ranking     | Uses cosine similarity + labeled data to train predictive model             |
|  GPT-4 Messaging Engine | Generates customized outreach messages for top-ranked profiles             |
|  Frontend (React)       | In development — clean UI for input, ranking, message viewing, and export   |

---

##  Use Case

> A user inputs a goal (e.g., "AI internships in Toronto")

The AI Agent:
1. Scores LinkedIn-style profiles based on relevance to that goal
2. Ranks them using a trained ML model
3. Generates personalized outreach messages using GPT-4
4. Returns a ranked list + corresponding tailored messages

---

## 🗂️ Project Structure
```
linkedin-agent/
├── data/
│   ├── labeled_profiles.json         # Labeled training data
│   └── top_ranked_profiles.json      # Filtered profiles for messaging
├── prompts/
│   └── message_template.txt          # GPT-4 prompt template
├── models/
│   ├── knn_baseline.py               # KNN with PCA
│   ├── knn_combined_embedding.py     # KNN with combined embeddings
│   └── rf_combined_embedding.py      # Random Forest with combined embeddings
├── generate_messages.py              # GPT-4 messaging from top results
├── score_profiles.py                 # Cosine similarity profile ranking
├── main.py                           # (Legacy) original prompt-to-message script
├── requirements.txt
└── README.md
```

## 📊 Model Experiments & MSE Scores

All models use OpenAI's `text-embedding-ada-002` (1536-dim) for vectorization. Combined embeddings concatenate the query and the profile bio vectors.

| Model           | Strategy               | MSE   |
|----------------|------------------------|-------|
| KNN            | Vanilla (raw vectors)  | 2.30  |
| KNN            | PCA                    | 2.183 |
| KNN            | Combined embeddings    | 2.10  |
| Random Forest  | Combined embeddings    | 2.04  |



##  How It Works

### Phase 1: Model Training
- Each profile is labeled with a `relevance_score` (1–5)
- Bios are embedded using OpenAI’s `text-embedding-ada-002`
- KNN model is trained (optionally with PCA)

### Phase 2: Inference Pipeline
- User query is embedded
- Profiles are ranked by model based on similarity + learned structure
- Top N profiles are selected
- GPT-4 generates custom messages for each

---

##  ML Performance Snapshot

| Metric              | Value     |
|---------------------|-----------|
| Mean Squared Error  | ~2.07     |
| Model               | KNN (k=3) |
| Dimensionality      | Reduced with PCA |

---

##  Next Milestones

- Feature engineering (title mapping, keyword match, skill overlap)
- Top-N precision evaluation
- Streamlit or React frontend
- XGBoost model benchmarking
- Message quality scoring / feedback loop
- CSV/clipboard export for selected messages

---

##  Notes

- All data used is synthetic or anonymized
- No scraping or real LinkedIn automation is included
- This is a prototype for research and demonstration only

---
