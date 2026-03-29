# AI-Enabled Visa Processing Time Estimator

> Predict U.S. visa processing times using machine learning — live at **[ai-visa-prediction-system.onrender.com](https://ai-visa-prediction-system.onrender.com)**

---

## Overview

The **AI-Enabled Visa Processing Time Estimator** is a full-stack machine learning web application that predicts how long a U.S. H-1B visa application will take to process, based on historical LCA disclosure data.

Visa applicants face significant uncertainty about processing durations. This system analyzes factors such as wage level, processing center, service office location, and seasonal submission patterns to deliver **data-driven processing time estimates** with confidence intervals.

The project covers the complete ML lifecycle — data preprocessing, feature engineering, model training, evaluation, and deployment as a publicly accessible web application.

---

## Live Demo

**[https://ai-visa-prediction-system.onrender.com](https://ai-visa-prediction-system.onrender.com)**

Fill in your application details and get an instant prediction with a min–max processing range and data insights dashboard.

---

## Key Features

- AI-powered processing time prediction using XGBoost
- Confidence interval — predicted range (min / avg / max days)
- Data Insights Dashboard with 3 live charts:
  - Processing time by service office (country)
  - Visa type (wage level) comparison
  - Monthly seasonal trend — application volume vs processing days
- Full-stack Flask web application with glassmorphism UI
- Deployed on Render with Gunicorn WSGI server

---

## Dataset

**H-1B LCA Disclosure Data (2020–2024)**
Source: [Kaggle — H1B LCA Disclosure Data](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024)

Fields used for training:

| Feature | Description |
|---|---|
| `WAGE_RATE_OF_PAY_FROM / TO` | Offered wage range (USD) |
| `PREVAILING_WAGE` | Department of Labor prevailing wage |
| `PW_WAGE_LEVEL` | Wage level (1–4: Entry to Expert) |
| `SERVICE_OFFICE` | Embassy / consulate location |
| `PROCESSING_CENTER` | USCIS processing center |
| `SUBMISSION_MONTH / QUARTER` | Seasonal submission features |
| `NEW_EMPLOYMENT / CHANGE_EMPLOYER` | Application type flags |
| `PROCESSING_DAYS` | Target variable |

---

## Repository Structure

```
AI_VISA_PREDICTION_SYSTEM/
│
├── data/
│   └── chart_data.json           # Pre-computed chart data (generated from training CSV)
│
├── models/
│   ├── model.pkl                 # Trained XGBoost model
│   └── columns.pkl               # Feature column order
│
├── notebooks/
│   ├── milestone1.ipynb          # EDA and data preprocessing
│   ├── milestone2.ipynb          # Feature engineering
│   └── Milestone_3.ipynb         # Model training and evaluation
│
├── documents/
│   ├── Sakthi-agile-doc.xlsx
│   ├── Sakthi-Defect_tracker-doc.xlsx
│   └── Sakthi-Unit_test-doc.xlsx
│
├── src/
│   └── prediction_engine/
│       ├── app.py                # Flask application entry point
│       ├── predict.py            # Prediction blueprint + chart data loader
│       ├── test_cases.py         # Structured test cases
│       ├── templates/
│       │   └── index.html        # Frontend UI
│       └── static/
│           └── style.css         # Glassmorphism styling
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Machine Learning Workflow

1. Data collection from public H-1B LCA disclosure records
2. Data cleaning and null handling
3. Feature engineering (month, quarter, wage ratio, one-hot encoding)
4. Exploratory data analysis with Matplotlib / Seaborn
5. Model training — Linear Regression, Random Forest, XGBoost
6. Model evaluation and selection
7. Model serialization with Joblib
8. Web app integration and cloud deployment

---

## Models Evaluated

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | baseline | baseline | baseline |
| Random Forest Regressor | — | — | — |
| **XGBoost Regressor** | **best** | **best** | **best** |

XGBoost was selected as the final model for deployment.

Performance is measured using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## Installation

```bash
git clone https://github.com/sakthi13854/AI_visa_prediction_system
cd AI_visa_prediction_system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run locally:

```bash
cd src/prediction_engine
flask run
```

---

## Deployment

The application is deployed on **Render** using Gunicorn as the WSGI server.

- Platform: [Render](https://render.com)
- Runtime: Python 3.14
- Server: Gunicorn
- Root directory: `src/prediction_engine`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

Live URL: **[https://ai-visa-prediction-system.onrender.com](https://ai-visa-prediction-system.onrender.com)**

> Note: The free tier spins down after 15 minutes of inactivity. The first request after sleep may take ~30 seconds to wake up.

---

## Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| ML | XGBoost, Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Web Framework | Flask |
| Frontend | HTML, CSS (Glassmorphism) |
| Server | Gunicorn |
| Deployment | Render |
| Version Control | Git, GitHub |
| Notebooks | Jupyter Lab |

---

## License

This project is licensed under the **MIT License**.

---


For questions, feedback, or contributions, feel free to open an issue or submit a pull request.
