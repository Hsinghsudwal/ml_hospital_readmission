from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load full pipeline (preprocessing + model)
model = joblib.load("artifacts/best_model.joblib")


def predict_readmission_risk(input_data: dict):
    """
    input_data = {
        "time_in_hospital": 4,
        "num_lab_procedures": 42,
        ...
    }
    """

    input_df = pd.DataFrame([input_data])

    prob = model.predict_proba(input_df)[0][1]
    
    # readmit_flag = prob >= 0.5
    # readmit_label = "Yes" if prob >= 0.5 else "No"
    # if prob >= 0.50:
    #     readmit_label = "Yes"
    # else:
    #     readmit_label = "No"
        

    if prob < 0.20:
        risk_label = "Low"
        readmit_label = "No"
        msg = "Patient has a low risk of readmission within 30 days."
        

    else:
        risk_label = "High"
        readmit_label = "Yes"
        msg = "High risk — likely readmission within 30 days."


    
    return {
        "risk_probability": float(prob),
        "risk_category": risk_label,
        "message": msg,
        "readmitted_30_days": readmit_label,
        # "readmit_int": int(prob >= 0.5),
        # "readmit": bool(prob >= 0.5)
        
    }


@app.route("/predict", methods=["POST"])

def predict():
    data = request.get_json()

    result = predict_readmission_risk(data)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
