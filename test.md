Hospital Readmission Prediction for Diabetes Patients
This project focuses on predicting hospital readmission for diabetes patients using machine learning. Early prediction of readmission risk helps hospitals improve patient care, reduce costs, and allocate resources more efficiently.
The steps I use:
1. Dataset is publicly available Diabetes US hospitals readmission.
2. Exploratory Data Analysis (EDA)
Inside the Jupyter Notebook, I performed an in-depth EDA:
Understanding data shape and patient encounter structure
Studying readmission distribution (<30, >30, NO)
Identifying missing values, duplicates, outliers
Visualizing key relationships such as age vs. readmission, diagnoses vs. outcomes
Checking class imbalance and feature correlations

This helped define the preprocessing pipeline and modeling strategy.
3. Data Cleaning
I applied multiple cleaning steps:
Removing or fixing invalid/missing values
Handling categorical variables with consistent encoding
Converting readmission labels into a clean binary target
<30 and >30 → 1 (Readmitted)
NO → 0 (Not readmitted)
Dropping high-missing or irrelevant features
Normalizing numeric features where needed

4. Feature Engineering
Feature transformations included:
Encoding categorical features (Ordinal / Binary where needed)
Generating features from diagnosis codes
Creating interaction features (e.g., number_of_medications × time_in_hospital)
Bucketizing continuous features (age groups, number of visits, etc.)
Converting dates and durations into usable numeric values

5. Feature Selection
To improve model performance:
Removed low-variance or redundant features
Performed correlation and mutual information analysis
Evaluated feature importance from baseline models

This reduced dimensionality and improved training efficiency.
6. Model Training Workflow
After EDA and preprocessing in Jupyter Notebook, I transitioned to standalone training scripts.
Steps:
Split dataset into train/validation/test sets

2. Train baseline models (Logistic Regression, Random Forest, XGBoost)
3. Apply hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
4. Evaluate models using:
ROC-AUC, Precision/Recall, F1-Score, Confusion Matrix

Model Selection
The model with the highest ROC-AUC on the validation set was selected as the final model.
7. Saving the Best Model
After choosing the best tuned model:
Saved it as a serialized artifact (best_model.joblib)
Stored evaluation metrics in a JSON report
Organized outputs into an artifacts/ directory

8. Local Inference
Before deployment, I tested local inference:
Loaded the model inside a Python script
Ran predictions on a sample JSON test file
Verified output probability and classification logic

This ensured the model was production-ready.
9. Flask API for Model Serving
I built a simple Flask API (app.py) that:
Loads the saved model from artifacts/best_model.joblib
Accepts patient data in JSON format
Returns predicted readmission risk
Includes error handling and input validation

I also created a test JSON file to simulate real requests.
10. Docker Deployment
To make the model portable and reproducible, I containerized the API using Docker.
Steps:
Created a Dockerfile including:

Python + required dependencies
The saved model artifact
The Flask API (app.py)

2. Build and run
docker build -t app.
docker run -p 9696:9696 app

3. Used JSON test file to send requests and confirm the API worked:
python test.py
Everything ran successfully inside the container, confirming the API and model operated consistently across environments.