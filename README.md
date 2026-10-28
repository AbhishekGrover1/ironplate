# *Ironplate*

![Python](https://img.shields.io/badge/Python-3.12-a8c8ec?style=flat-square&logo=python&logoColor=ebe8e2&labelColor=1b1f24)
![FastAPI](https://img.shields.io/badge/FastAPI-0.143-a6dcc6?style=flat-square&logo=fastapi&logoColor=ebe8e2&labelColor=1b1f24)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f0c9a0?style=flat-square&logo=scikitlearn&logoColor=ebe8e2&labelColor=1b1f24)
![SciPy](https://img.shields.io/badge/SciPy-1.17-c5bff0?style=flat-square&logo=scipy&logoColor=ebe8e2&labelColor=1b1f24)
![NumPy](https://img.shields.io/badge/NumPy-2.4-a9d6ee?style=flat-square&logo=numpy&logoColor=ebe8e2&labelColor=1b1f24)
![Render](https://img.shields.io/badge/Deploy-Render-dcc3ee?style=flat-square&logo=render&logoColor=ebe8e2&labelColor=1b1f24)

</div>

Ironplate turns plain messages into calorie targets, full-day meal plans, weekly workouts, food logs and weight-trend analysis. It runs entirely on its own models and data, with no outside chat service and no API keys.

## What is inside

| Part | How it works |
| --- | --- |
| Intent model | TF-IDF on word and character n-grams into logistic regression, trained at build time. Cross-validation is grouped so variants of one phrase never sit in both train and test. |
| Slot extraction | Rules for age, sex, height, weight, goal, diet, allergens and units (kg/lb, cm/ft-in). Missing details are asked for, then the original request resumes. |
| Food search | Character n-gram index plus alias matching, so typos and plurals still resolve. A portion parser turns "2 rotis and a bowl of dal" into grams. |
| Meal planner | A mixed-integer program (SciPy HiGHS) chooses foods and portions to hit calories and macros within diet, allergy and dislike limits. About one second per day. |
| Workout generator | Picks a split from training days, then fills each day by movement pattern, equipment and level. |
| Progress model | Theil-Sen weekly rate, time-aware moving average, goal ETA, and an adaptive maintenance estimate from logged intake and weight. |
| Guard rails | FAQ retrieval for common questions. Emergencies, self-harm and eating-disorder language never get targets. Calorie floors, no deficits when underweight, adults only. |

```mermaid
flowchart LR
    A["Browser<br/>profile, log and weights in localStorage"] -->|"message + state"| B["POST /api/chat"]
    B --> C{"Safety"}
    C -->|"clear"| D["FAQ"]
    D -->|"no match"| E["Slots + intent"]
    E --> F["Targets, planner, workouts, food search, trend"]
    F -->|"reply, blocks, state changes"| A
```

The server is stateless: the page keeps everything in the browser and sends it with each message.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                 # trains the intent model (also runs on first start if missing)
uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

## Deploy on Render

Push the repository to GitHub and create a Blueprint from `render.yaml`. The build installs the requirements and trains the model, and the free plan is enough.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/chat` | Message and browser state in, reply, blocks and state changes out |
| GET | `/api/foods?q=` | Fuzzy food search |
| GET | `/api/model` | Training metrics written by `train.py` |
| GET | `/health` | Status |

## Layout

```
app/        FastAPI app, dialogue, models, planner, trend
data/       foods.csv, exercises.csv, intents.json, faq.json
static/     index.html, style.css, script.js
tests/      pytest suite (pip install -r requirements-dev.txt && pytest)
train.py    trains the intent model
render.yaml Render blueprint
```

## Notes

Food values are approximate per-100 g reference figures for everyday foods. Targets are estimates for healthy adults and are not medical advice.

## License

MIT, see [LICENSE](LICENSE).

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-AbhishekGrover1-a8c8ec?style=flat-square&logo=github&logoColor=ebe8e2&labelColor=1b1f24)](https://github.com/AbhishekGrover1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abhishek--grover--07-a6dcc6?style=flat-square&logo=linkedin&logoColor=ebe8e2&labelColor=1b1f24)](https://www.linkedin.com/in/abhishek-grover-07)
[![Instagram](https://img.shields.io/badge/Instagram-abh1shekgrover-f0c9a0?style=flat-square&logo=instagram&logoColor=ebe8e2&labelColor=1b1f24)](https://www.instagram.com/abh1shekgrover)

AI ML Engineer Abhishek Grover

</div>
