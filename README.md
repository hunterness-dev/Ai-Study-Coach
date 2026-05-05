# 🎓 AI-Study-Coach

An intelligent, adaptive study planning platform powered by **machine learning** (scikit-learn Ridge regression) and **reinforcement learning** (Stable-Baselines3 PPO).

---

## Architecture

```
ai-study-coach/
├── api/                 # FastAPI routes & Pydantic schemas
├── app/                 # Streamlit frontend
├── core/                # Settings, logging
├── services/            # Business logic (tracking, scheduler, plan)
├── ml/                  # Feature engineering + scikit-learn pipeline
├── rl/                  # Gym environment + PPO agent
├── utils/               # SQLAlchemy models & DB session
├── tests/               # pytest suite
├── scripts/             # Training scripts (ML + RL)
├── data/                # SQLite database (auto-created)
├── models/              # Saved model artifacts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Local (without Docker)

```bash
# Clone & install
git clone https://github.com/your-org/ai-study-coach.git
cd ai-study-coach
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env

# Start API
uvicorn main:app --reload

# Start frontend (new terminal)
streamlit run app/streamlit_app.py
```

### 2. Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

- API:      http://localhost:8000
- Docs:     http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## API Reference

| Method | Path      | Description                         |
|--------|-----------|-------------------------------------|
| GET    | /health   | Service health check                |
| POST   | /log      | Log a study session                 |
| GET    | /plan     | Get an optimised study plan         |

### POST /log

```json
{
  "subject": "Mathematics",
  "hours": 2.5,
  "score": 78.0
}
```

### GET /plan (response)

```json
{
  "allocations":      {"Mathematics": 3.2, "Physics": 4.8},
  "predicted_scores": {"Mathematics": 81.5, "Physics": 68.0},
  "features":         {"Mathematics": {"avg_score": 78.0, "trend": 0.5, "efficiency": 31.2, "session_count": 4}},
  "source":           "rl"
}
```

---

## Training Models

### Ridge ML Pipeline

```bash
python scripts/train_ml.py
```

Requires ≥ 2 subjects with ≥ 2 logged sessions each.

### PPO RL Agent

```bash
python scripts/train_rl.py --timesteps 50000
```

---

## Testing

```bash
pytest tests/ -v
```

---

## How It Works

### Scheduler (always available)

Priority = `difficulty_weight + weakness_weight + negative_trend_weight`  
Allocations normalised to `DAILY_STUDY_HOURS`.

### ML Pipeline (Ridge Regression)

Features: `[avg_score, trend, efficiency]`  
Target: next-session score prediction.

### RL Agent (PPO)

- **State**: flattened feature vector per subject
- **Action**: continuous allocation weights
- **Reward**: `mean(score improvement) - 0.5 × std(hours)` (performance vs fatigue)

The RL agent takes priority over the rule-based scheduler when a trained model exists.

---

## CI/CD

GitHub Actions runs on every push:
1. **lint** — `ruff check`
2. **test** — `pytest tests/`

---

## License

MIT
