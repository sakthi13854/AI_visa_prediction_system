from flask import Blueprint, request, render_template
import joblib
import pandas as pd
import json
import os

predict_bp = Blueprint('predict', __name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

columns  = joblib.load(os.path.join(BASE_DIR, 'models', 'columns.pkl'))
model    = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))
_df_raw  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'data_for_training.csv'))

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



def _build_chart_data(df):

    
    country_labels, country_days = [], []

    
    office_name_map = {
        i: col.replace('SERVICE_OFFICE_', '').replace('_', ' ').title()
        for i, col in enumerate(required_service_cols)
        if col.startswith('SERVICE_OFFICE_')
    }

    if 'SERVICE_OFFICE' in df.columns:
        grouped = df.groupby('SERVICE_OFFICE')['PROCESSING_DAYS'].agg(['mean', 'count'])
        for code, row in grouped.iterrows():
            if row['count'] >= 3:
               
                try:
                    label = office_name_map.get(int(code),
                            str(code).replace('_', ' ').title())
                except (ValueError, TypeError):
                    label = str(code).replace('_', ' ').title()
                if label.lower() not in ('nan', 'none', ''):
                    country_labels.append(label)
                    country_days.append(round(row['mean'], 1))
    else:
       
        service_cols = [c for c in df.columns if c.upper().startswith('SERVICE_OFFICE_')]
        for col in service_cols:
            subset = df[df[col].astype(float) > 0]['PROCESSING_DAYS']
            if len(subset) >= 3:
                name = col.upper().replace('SERVICE_OFFICE_', '').replace('_', ' ').title()
                country_labels.append(name)
                country_days.append(round(subset.mean(), 1))

 
    paired = sorted(zip(country_days, country_labels), reverse=True)
    if paired:
        country_days, country_labels = zip(*paired)
        country_days   = list(country_days)
        country_labels = list(country_labels)

   
    level_map = {1: 'Entry (L1)', 2: 'Intermediate (L2)',
                 3: 'Experienced (L3)', 4: 'Expert (L4)'}

    visa_labels, visa_days, visa_approval = [], [], []

    if 'PW_WAGE_LEVEL' in df.columns:
        grouped = df.groupby('PW_WAGE_LEVEL')['PROCESSING_DAYS'].agg(['mean', 'count'])
        for lvl, row in grouped.iterrows():
            if row['count'] > 10:
                visa_labels.append(level_map.get(int(lvl), f'Level {lvl}'))
                visa_days.append(round(row['mean'], 1))

        
        if 'CASE_STATUS' in df.columns:
            for lvl in grouped.index:
                sub = df[df['PW_WAGE_LEVEL'] == lvl]
                rate = round((sub['CASE_STATUS'].str.upper() == 'CERTIFIED').mean() * 100, 1)
                visa_approval.append(rate)
        else:
            visa_approval = [None] * len(visa_labels)

    trend      = df.groupby('SUBMISSION_MONTH')['PROCESSING_DAYS'].mean()
    vol_trend  = df.groupby('SUBMISSION_MONTH').size()

    avg_days    = [round(trend.get(i, 0), 1)    for i in range(1, 13)]
    monthly_vol = [int(vol_trend.get(i, 0))      for i in range(1, 13)]

    return (
        json.dumps(country_labels),
        json.dumps(country_days),
        json.dumps(visa_labels),
        json.dumps(visa_days),
        json.dumps(visa_approval),
        json.dumps(avg_days),
        json.dumps(monthly_vol),
    )


    _country_labels,
    _country_days,
    _visa_labels,
    _visa_days,
    _visa_approval,
    _avg_days,
    _monthly_vol,
) = _build_chart_data(_df_raw)



@predict_bp.route('/', methods=['POST'])
def predict():
    df = pd.DataFrame(0, index=[0], columns=columns)

    data = {
        "WAGE_RATE_OF_PAY_FROM" : float(request.form['wage_from']),
        "WAGE_RATE_OF_PAY_TO"   : float(request.form['wage_to']),
        "PREVAILING_WAGE"       : float(request.form['prevailing_wage']),
        "PW_WAGE_LEVEL"         : int(request.form['wage_level']),
        "NEW_EMPLOYMENT"        : int(request.form['new_emp']),
        "CHANGE_EMPLOYER"       : int(request.form['change_emp']),
        "AMENDED_PETITION"      : int(request.form['amended']),
        "SUBMISSION_YEAR"       : int(request.form['year']),
        "SUBMISSION_MONTH"      : int(request.form['month']),
        "SUBMISSION_DAY"        : 1,
        "SUBMISSION_QUARTER"    : (int(request.form['month']) - 1) // 3 + 1,
    }

    for key, val in data.items():
        if key in df.columns:
            df[key] = val

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

    df = df[model_features].astype(float)

    prediction   = model.predict(df)[0]
    error_margin = max(prediction * 0.10, 2)
    low          = round(prediction - error_margin)
    high         = round(prediction + error_margin)

    return render_template(
        "index.html",
        prediction     = round(prediction),
        low            = low,
        high           = high,
      
        country_labels = _country_labels,
        country_days   = _country_days,
        visa_labels    = _visa_labels,
        visa_days      = _visa_days,
        visa_approval  = _visa_approval,
        avg_days       = _avg_days,
        monthly_vol    = _monthly_vol,
    )