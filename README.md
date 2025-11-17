# Hospital Readmission Prediction for Patients

Predict hospital readmissions within 30 days using machine learning to identify high-risk patients and improve care outcomes.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Setup](#setup)
- [Research Notebook](#research-notebook)
- [Training Pipeline](#training-pipeline)
- [Flask API](#flask-api)
- [Docker Deployment](#docker-deployment)
- [Best Practices](#best-practices)
- [Next Steps](#next-steps)


---

## Overview

Hospital readmission within 30 days indicates potential gaps in patient care. This project uses machine learning to predict readmission risk for diabetic patients, enabling targeted interventions and reducing unnecessary hospitalizations.

---

## Problem Statement

Hospital readmissions are a critical healthcare challenge because they:

   - Suggest that the patient’s health didn’t improve as expected
   - Cause stress and financial problem for patients and families
   - May result in financial penalties for hospitals
   - Highlight gaps in follow-up care, medication management, or overall treatment planning

---

## Solution

End-to-end ML system featuring:
- **Models**: Balanced Random Forest, LightGBM, XGBoost, Random Forest, Logistic Regression
- **Pipeline**: Automated preprocessing (imputation, encoding, scaling)
- **Class Imbalance Handling**: Techniques to improve minority class prediction
- **Flask API**: Real-time inference for clinical integration
- **Docker**: Containerized deployment for reproducibility

Uses clinical features (lab procedures, medications, diagnoses, hospital time) to classify readmission risk.

---

#### Goal

The primary goal is to **keep patients healthy at home** and **reduce unnecessary hospital visits** by:

- Identifying patients at high risk of readmission within 30 days
- Enabling targeted interventions for high-risk patients
- Improving patient outcomes and healthcare efficiency

---
## Setup

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/Hsinghsudwal/ml_hospital_readmission.git
cd hospital-readmission
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements_1.txt
```
---

## Research Notebook

The research notebook (`notebook/notebooks.ipynb`) implements exploratory and modeling workflow:

1. **EDA**: Dataset structure, distributions, correlations
2. **Preprocessing**: Missing value handling, encoding, scaling
3. **Feature Engineering**: Features and selection
4. **Model Experimentation**: Multiple algorithms with hyperparameter tuning
5. **Evaluation**: ROC-AUC, recall, F1-score, confusion matrices
6. **Serialization**: Save best model and pipeline

**Run**:
```bash
jupyter notebook notebook/notebooks.ipynb
```

---

## Script - Train Model

The training script (`src/train.py`) implements an automated end-to-end workflow for model development.

**Flow**:
- Data loading and validation
- Cleaning (handle missing values, map categories)
- Preprocessing pipeline (numeric: median imputation + scaling, categorical: constant imputation + encoding)
- Train/test split with stratification
- Hyperparameter tuning via RandomizedSearchCV
- Model selection based on ROC-AUC, recall, F1-score
- Save artifacts to `artifacts/`

**Run**:
```bash
python main.py
```
---

## Flask Api

The Flask application (`src/app.py`) serves the trained model and provides API for real-time predictions.

### Key Features

- **Model Loading**: Loads the complete pipeline from `artifacts/best_model.joblib`
- **Prediction Endpoint**: Accepts patient data and returns readmission risk
- **Risk**:
  - **Low risk**: probability < 0.2
  - **High risk**: probability ≥ 0.2
- **JSON API**: Easy integration with clinical systems

### API Endpoint

**POST** `/predict`

**Request Body** (JSON):
```json
{
  "time_in_hospital": 3,
  "num_lab_procedures": 45,
  "num_procedures": 2,
  "num_medications": 15,
  "number_diagnoses": 7,
  "age": 65
}
```

**Response** (JSON):
```json
{
  "risk_probability": 0.35,
  "risk_category": "High",
  "message": "Patient has high risk of readmission.",
  "readmitted_30_days": "Yes"
}
```

**Run**:
```bash
python src/app.py
```

The API will be available at `http://localhost:9696/predict`

---

## Docker Deployment

The project includes Docker support for containerized deployment.
* Make sure Docker is installed

**with docker**: install fewer dependencies
```bash
COPY requirements.txt
```

### Build and Run
```bash
# Build image
docker build -t app .

# Run container
docker run -p 9696:9696 app
```
### Pull Pre-built Image

```bash
docker pull hsinghsudwal/hr-clinical:latest
docker run -p 9696:9696 hsinghsudwal/hr-clinical:latest
```

### Test API
Create `test.py`
```python
import requests

url = "http://localhost:9696/predict"

patient = {
    "time_in_hospital": 4,
    "num_lab_procedures": 42,
    .
    .
}...

response = requests.post(url, json=patient)
# print("Prediction:", response.json())
print(response)
```
**Run**:
```bash
python test.py
```
**output**:
```bash
{'message': 'High risk — likely readmission within 30 days.', 'readmitted_30_days': 'Yes', 'risk_category': 'High', 'risk_probability': 0.22219116985797882}      

--- RESULT ---
Patient WILL LIKELY be readmitted within 30 days!
High risk.
```
---
### Docker Compose (Optional)

```bash
docker-compose up -d
```

---

## Best Practices

- Use virtual environments for dependency isolation
- Version control all code and artifacts
- Anonymize patient data for privacy
- Log metrics and training artifacts
- Automate deployment via Docker

---
## Next Steps

- Extend to multi-class instead binary
- Integrate Optuna for advanced hyperparameter optimization
- Implement nested cross-validation for unbiased evaluation
- Add SHAP analysis for model explainability
- Build interactive dashboard for clinical staff
- Include temporal features for time-series predictions
- Deploy to cloud platforms (AWS)
- Add A/B testing framework for model comparison

