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
│   ├── knn_baseline.py               # KNN with raw embeddings
│   ├── knn_pca.py                    # KNN with PCA-reduced embeddings
│   ├── knn_combined_embedding.py     # KNN with combined query + bio embeddings
│   ├── rf_combined_embedding.py      # Random Forest with combined embeddings
├── generate_messages.py              # GPT-4 messaging from top results
├── select_top_profiles.py            # Uses pre-trained KNN model to score and rank profiles based on user query
├── requirements.txt
└── README.md
```


## Model Experiments & MSE Scores

All models use OpenAI's `text-embedding-ada-002` (1536-dim) for vectorization. Combined embeddings concatenate the query and the profile bio vectors.

| Model           | Strategy               | MSE   |
|----------------|------------------------|-------|
| KNN            | Vanilla (raw vectors)  | 2.30  |
| KNN            | PCA                    | 2.183 |
| KNN            | Combined embeddings    | 2.10  |
| Random Forest  | Combined embeddings    | 2.04  |

```

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

- New user query is embedded
- Each profile bio is embedded and combined with the query embedding
- Combined vector is passed into the trained model
- Model predicts relevance scores for all profiles
- Top N profiles are saved to `top_ranked_profiles.json` for GPT-4 messaging

### Phase 3: Message Generation (In Progress)

- GPT-4 generates personalized outreach messages using a pre-written prompt template
- Messages are generated for the top N profiles returned by the model
- Output includes: profile name, message content, and relevance score

### Phase 4: UI & Agent Behavior (Planned)

- Streamlit or React frontend for inputting goals and viewing results
- Agent-like flow:
  1. User inputs a career goal
  2. Model ranks top profiles
  3. Messages are auto-generated
  4. User can view, export, or copy messages
- Add smart filtering (location, industry, etc.)

---
## Next Milestones

- Add feature engineering:
  - Title seniority mapping (e.g., intern < engineer < manager)
  - Keyword overlap between query and bio
  - Skill and interest match scoring
- Test and benchmark XGBoost model
- Add precision@K and Top-N evaluation
- Build frontend interface (React or Streamlit)
- Implement message quality scoring and feedback loop
- Add export option (CSV or clipboard)

---

## Notes

- All profile data currently used is synthetic, generated to mimic LinkedIn-style bios and queries
- No scraping or real LinkedIn automation is used in this version
- The system is being developed as a prototype AI agent for career networking, with future potential for real-world deployment