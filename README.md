# AI Visa Status and Processing Time Prediction System

AI Visa Status and Processing Time Prediction System is a small machine-learning project that demonstrates building a regression model to predict visa application outcomes from a cleaned dataset.

**Deliverables**

- Cleaned and structured dataset
- Feature-engineered columns
- Final target variable: `processing_time_days`
- Dataset ready for Exploratory Data Analysis (EDA)

**Repository Structure**

- data/: cleaned dataset(s)
	- [data/cleaned_sample.csv](data/cleaned_sample.csv)
- notebooks/: exploratory analysis and modeling
	- [notebooks/milestone1.ipynb](notebooks/milestone1.ipynb)
- requirements.txt: Python dependencies

**Quick Start**

1. Create a Python environment (Python 3.8+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Open the notebook to explore and run the pipeline interactively:

```bash
jupyter lab notebooks/milestone1.ipynb
```

**Data**

The repository includes a cleaned sample at [data/cleaned_sample.csv](data/cleaned_sample.csv). Use this file for quick experiments; see the notebook for data schema and preprocessing steps.

- Original source dataset: H1B LCA Disclosure Data 2020-2024 (Kaggle) — Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv
	- https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024?select=Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv



**Usage**

- Follow `notebooks/milestone1.ipynb` to reproduce the analysis and training steps.
- If you add scripts, include usage examples here.

**Dependencies**

- See `requirements.txt` for exact package versions.



**Contributing**

- Contributions welcome. Open an issue or PR describing your change.

**License & Contact**

- License: MIT (replace with your preferred license)
- Author: Sakthi S — open issues or email for questions.

---
