
# AI-Enabled Visa Status Prediction and Processing Time Estimator

## Overview

The **AI-Enabled Visa Status Prediction and Processing Time Estimator** is a machine learning project that predicts visa application processing time using historical visa data.

Visa applicants often face uncertainty regarding how long their applications will take to process. Processing times can vary depending on several factors such as visa category, applicant country, processing center workload, and seasonal trends.

This project applies **data analysis and machine learning techniques** to estimate visa processing durations based on past application records. By analyzing historical data, the system provides **data-driven predictions** that help applicants better understand potential waiting periods.

The project demonstrates the complete machine learning workflow including **data preprocessing, exploratory data analysis, feature engineering, model training, and prediction generation**.

---

## Key Features

- Data cleaning and preprocessing pipeline
- Feature engineering for predictive modeling
- Exploratory Data Analysis (EDA) with visual insights
- Machine learning regression models for processing time prediction
- Model evaluation using standard performance metrics
- Reproducible analysis using Jupyter notebooks

---

## Dataset

The dataset used in this project is derived from:

**H1B LCA Disclosure Data (2020–2024)**  
Source: https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024

The dataset contains information such as:

- Employer name
- Work location
- Wage information
- Visa case status
- Application processing details

After preprocessing, the dataset is cleaned and structured for machine learning analysis.

The final dataset includes the target variable:

```

processing_time_days

```

This represents the number of days taken between visa application submission and decision.

---

## Repository Structure

```

AI-Visa-Prediction/
│
├── data/
│   └── cleaned_sample.csv
│
├── notebooks/
│   └── milestone1.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── requirements.txt
│
└── README.md

````

---

## Machine Learning Workflow

The project follows a standard machine learning pipeline:

1. Data collection from public datasets
2. Data cleaning and preprocessing
3. Feature engineering
4. Exploratory data analysis
5. Model training
6. Model evaluation
7. Model selection
8. Prediction generation

Regression models are used to estimate visa processing time based on historical application patterns.

---

## Models Used

The following regression algorithms are implemented and evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Model performance is evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

The best-performing model can be exported for prediction use.

---

## Installation

Create a Python virtual environment:

```bash
python -m venv .venv
````

Activate the environment:

**Linux / Mac**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook to explore the dataset and run the machine learning pipeline:

```bash
jupyter lab notebooks/milestone1.ipynb
```

The notebook contains the full workflow including data preprocessing, analysis, and model training.

---

## Technologies Used

**Programming Language**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

**Tools**

* Jupyter Notebook
* Git
* GitHub

---

## Future Improvements

Potential extensions for this project include:

* Implementing advanced models such as **XGBoost or LightGBM**
* Adding visa approval probability prediction
* Building a web interface using **Flask or Streamlit**
* Deploying the prediction system to a cloud platform
* Integrating real-time application data for improved predictions

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Sakthi S**

For questions, feedback, or contributions, feel free to open an issue or submit a pull request.


