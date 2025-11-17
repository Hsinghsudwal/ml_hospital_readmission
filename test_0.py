import requests


url = 'http://localhost:9696/predict'

readmission = {
    
    "diag_1": 410,
    "diag_2": 427,
    "diag_3": 428,
    "num_lab_procedures": 66,
    "discharge_disposition_id": 1,
    "num_medications": 19,
    "age_num": 55,
    "time_in_hospital": 2,
    "number_inpatient": 0,
    "admission_source_id": 4,
    "number_diagnoses": 7,
    "admission_type_id": 2,
    "num_procedures": 1,
    "number_outpatient": 0,
    "number_emergency": 0,
    "insulin": "Down",
    "race": None,
    "gender": "Female",
    "A1Cresult": None,
    "diabetesMed": "Yes",
    "readmitted_map": 0,

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
