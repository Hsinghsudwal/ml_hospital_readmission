import requests


url = 'http://localhost:9696/predict'

readmission = {
    
    "diag_1": 250.7,
    "diag_2": 403,
    "diag_3": 996,
    "num_lab_procedures": 47,
    "discharge_disposition_id": 1,
    "num_medications": 17,
    "age_num": 45,
    "time_in_hospital": 9,
    "number_inpatient": 0,
    "admission_source_id": 7,
    "number_diagnoses": 9,
    "admission_type_id": 1,
    "num_procedures": 2,
    "number_outpatient": 0,
    "number_emergency": 0,
    "insulin": "Steady",
    "race": "AfricanAmerican",
    "gender": "Female",
    "A1Cresult": None,
    "diabetesMed": "Yes",

}


response = requests.post(url, json=readmission).json()
print(response)

# Decide on risk
risk = response["risk_category"]
readmitted = response["readmitted_30_days"]

print("\n--- RESULT ---")
if readmitted == "Yes":
    print("Patient WILL LIKELY be readmitted within 30 days!")
else:
    print("Patient NOT likely to be readmitted.")
    
    
if risk == "High":
    print("High risk.")

else:
    print("Low risk.")
