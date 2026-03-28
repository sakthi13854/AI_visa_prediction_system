from flask import Blueprint, request, render_template
import joblib
import os
import pandas as pd
import json

predict_bp = Blueprint('predict', __name__)

columns = joblib.load('/home/sakthi/projects/AI_visa_prediction_system/models/columns.pkl')
model = joblib.load('/home/sakthi/projects/AI_visa_prediction_system/models/model.pkl')

columns = list(dict.fromkeys(columns))
required_service_cols = [
    'EMPLOYER_STATE_FM',
    'SERVICE_OFFICE_CANBERRA',
    'SERVICE_OFFICE_JERUSALEM',
    'SERVICE_OFFICE_LUANDA',
    'SERVICE_OFFICE_SANTIAGO',
    'SERVICE_OFFICE_TBILISI',
    'SERVICE_OFFICE_TORONTO',
    'SERVICE_OFFICE_WASHINGTON DC'
]

for col in required_service_cols:
    if col not in columns:
        columns.append(col)

@predict_bp.route('/', methods=['POST'])
def predict():
    df = pd.DataFrame(0, index=[0], columns=columns)

    data = {
        "WAGE_RATE_OF_PAY_FROM": float(request.form['wage_from']),
        "WAGE_RATE_OF_PAY_TO": float(request.form['wage_to']),
        "PREVAILING_WAGE": float(request.form['prevailing_wage']),
        "PW_WAGE_LEVEL": int(request.form['wage_level']),
        "NEW_EMPLOYMENT": int(request.form['new_emp']),
        "CHANGE_EMPLOYER": int(request.form['change_emp']),
        "AMENDED_PETITION": int(request.form['amended']),
        "SUBMISSION_YEAR": int(request.form['year']),
        "SUBMISSION_MONTH": int(request.form['month']),
        "SUBMISSION_DAY": 1,
        "SUBMISSION_QUARTER": (int(request.form['month'])-1)//3 + 1
    }

    for key in data:
        if key in df.columns:
            df[key] = data[key]


    proc_col = f"PROCESSING_CENTER_{request.form['processing_center']}"
    serv_col = f"SERVICE_OFFICE_{request.form['service_center']}"

    if proc_col in df.columns:
        df[proc_col] = 1

    if serv_col in df.columns:
        df[serv_col] = 1


    model_features = model.get_booster().feature_names

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]
    df = df.astype(float)


    prediction = model.predict(df)[0]
    error_margin = max(prediction * 0.10, 2)

    low = prediction - error_margin
    high = prediction + error_margin


    df_data = pd.read_csv("/home/sakthi/data_for_training.csv")   # use your dataset

    trend = df_data.groupby("SUBMISSION_MONTH")["PROCESSING_DAYS"].mean()

    months = ["Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"]

    avg_days = [round(trend.get(i, 0), 2) for i in range(1, 13)]
    
    return render_template(
    "index.html",
    avg_days=json.dumps(avg_days),
    prediction=round(prediction, ),
    low=round(low),
    high=round(high)
    )


   