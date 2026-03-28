import joblib
import pandas as pd


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



def build_input(data_dict):
    df = pd.DataFrame(0, index=[0], columns=columns)

    for key in data_dict:
        if key in df.columns:
            df[key] = data_dict[key]

    
    if data_dict.get("processing_center"):
        proc_col = f"PROCESSING_CENTER_{data_dict['processing_center']}"
        if proc_col in df.columns:
            df[proc_col] = 1

    if data_dict.get("service_center"):
        serv_col = f"SERVICE_OFFICE_{data_dict['service_center']}"
        if serv_col in df.columns:
            df[serv_col] = 1

    
    model_features = model.get_booster().feature_names

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]
    df = df.astype(float)

    return df



def predict_case(data_dict):
    df = build_input(data_dict)

    prediction = model.predict(df)[0]

   
    error_margin = max(prediction * 0.10, 2)

    low = prediction - error_margin
    high = prediction + error_margin
    spread_ratio = (high - low) / prediction

    confidence = max(0, min(100, 100 - (spread_ratio * 100)))
    

    return round(prediction), round(low), round(high), round(confidence, 2)



test_cases = [
    {
        "name": "India  (Jan)",
        "data": {
            "WAGE_RATE_OF_PAY_FROM": 30000,
            "WAGE_RATE_OF_PAY_TO": 60000,
            "PREVAILING_WAGE": 40000,
            "PW_WAGE_LEVEL": 1,
            "NEW_EMPLOYMENT": 1,
            "CHANGE_EMPLOYER": 0,
            "AMENDED_PETITION": 0,
            "SUBMISSION_YEAR": 2024,
            "SUBMISSION_MONTH": 1,
            "SUBMISSION_DAY": 1,
            "SUBMISSION_QUARTER": 1,
            "processing_center": "California Service Center",
            "service_center": "MUMBAI"
        }
    },
    {
        "name": "CA (June)",
        "data": {
            "WAGE_RATE_OF_PAY_FROM": 80000,
            "WAGE_RATE_OF_PAY_TO": 120000,
            "PREVAILING_WAGE": 90000,
            "PW_WAGE_LEVEL": 4,
            "NEW_EMPLOYMENT": 1,
            "CHANGE_EMPLOYER": 0,
            "AMENDED_PETITION": 0,
            "SUBMISSION_YEAR": 2025,
            "SUBMISSION_MONTH": 6,
            "SUBMISSION_DAY": 1,
            "SUBMISSION_QUARTER": 2,
            "processing_center": "Texas Service Center",
            "service_center": "WASHINGTON DC"
        }
    },
    {
        "name": "UK  (Dec)",
        "data": {
            "WAGE_RATE_OF_PAY_FROM": 40000,
            "WAGE_RATE_OF_PAY_TO": 70000,
            "PREVAILING_WAGE": 50000,
            "PW_WAGE_LEVEL": 2,
            "NEW_EMPLOYMENT": 0,
            "CHANGE_EMPLOYER": 1,
            "AMENDED_PETITION": 0,
            "SUBMISSION_YEAR": 2024,
            "SUBMISSION_MONTH": 12,
            "SUBMISSION_DAY": 1,
            "SUBMISSION_QUARTER": 4,
            "processing_center": "New York Service Center",
            "service_center": "LONDON"
        }
    }
]



if __name__ == "__main__":
    print("\n Running Test Cases...\n")

    for case in test_cases:
        pred, low, high, conf = predict_case(case["data"])

        print(f" {case['name']}")
        print(f"   Prediction: {pred} days")
        print(f"   Range: {low} – {high} days")
        print(f"   Confidence: {conf}%\n")