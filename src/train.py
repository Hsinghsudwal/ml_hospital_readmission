import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, confusion_matrix, classification_report

from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import config


def load_data(path):
    df = pd.read_csv(path)
    print('Raw data size: ', df.shape)
    # print(df.isna().sum())
    return df


def clean_data(df):
    
    df.replace('?', np.nan, inplace=True)
    
    age_map = {
    '[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45,
    '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95
    }
    
    df['age_num'] = df['age'].map(age_map)
    # df['readmitted_map'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 1})
    df['readmitted_map'] = df['readmitted'].map({
        '<30': 1,
        '>30': 1,
        'NO': 0
    })
    print('Clean data size: ', df.shape)
    # print(df.columns)
    return df
    

def preprocess_data(df):
   
    candidate_cols = [
        'diag_1', 'diag_2', 'diag_3', 'num_lab_procedures', 'discharge_disposition_id', 'num_medications', 
        'age_num', 'time_in_hospital', 'number_inpatient', 'admission_source_id', 'number_diagnoses', 
        'admission_type_id', 'num_procedures', 'number_outpatient', 'number_emergency', 'insulin',
        'race', 'gender', 'A1Cresult', 'diabetesMed'
    ]


    # Only features are used
    X = df[candidate_cols]
    y = df['readmitted_map']

    # Get numeric and categorical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical pipeline
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ], remainder='drop')
    
    # Split
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print('X train size: ', X_train.shape)
    
    return X_train, X_test, y_train, y_test, preprocessor


# Train Model
def model_tune():

    models = {
        # "LightGBM": LGBMClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        # "BalancedRF": BalancedRandomForestClassifier(random_state=42),
        # "RandomForest": RandomForestClassifier(random_state=42),
    }

    params = {
        "LightGBM": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [-1, 5, 10],
            "model__num_leaves": [31, 50, 70],
            "model__subsample": [0.8, 1.0]
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0]
        },
        "BalancedRF": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        },
        "RandomForest": {
            "model__n_estimators": [50, 100],
            # "model__max_depth": [None, 10, 20]
        }
    }

    return models, params


def train_model(X_train, y_train, preprocessor):
    
    models, params = model_tune()
    results = {}
    
    for name, model in models.items():
        print(f"\n--- TUNING {name} ---")
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])
    

        search = RandomizedSearchCV(
                pipe,
                params[name],
                n_iter=10,
                scoring="roc_auc",
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

        search.fit(X_train, y_train)
        results[name] = search
        
    return results

# select model
def select_and_save_best_model(results, X_test, y_test):

    print(" Selecting best model...")

    best_name, best_model = None, None
    best_scores = {"roc": -1, "recall": -1, "f1": -1}

    for name, tuned in results.items():
        model = tuned.best_estimator_

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\n{name}: ROC={roc:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        print('confusion_matrix: ', cm)
        print('classification_report: ',report)

        if (roc > best_scores["roc"]) or \
           (roc == best_scores["roc"] and recall > best_scores["recall"]) or \
           (roc == best_scores["roc"] and recall == best_scores["recall"] and f1 > best_scores["f1"]):

            best_name = name
            best_model = model
            best_scores = {"roc": roc, "recall": recall, "f1": f1}

    print("Best Model:", best_name)
    print(best_scores)

    model_path = config['output']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    report_path = config['output']['report_path']
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(best_scores, f, indent=2)

    print(f" Model saved to {model_path} + metrics saved to {report_path}.")

    return best_model, best_scores


# Evaluate Model
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X_test)[:, 1]
        except:
            prob = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if prob is not None else None,
    }

    return metrics

# layer decision
def make_decision(eval_metrics):
    print(eval_metrics)
    THRESHOLDS = {
        "roc": 0.7,
        "recall": 0.5,
        "f1": 0.5
    }
    report_path = config['output']['report_path']
    with open(report_path, "r") as f:
        metrics =  json.load(f)
        
    if metrics["roc"] >= THRESHOLDS["roc"] and \
       metrics["recall"] >= THRESHOLDS["recall"] and \
       metrics["f1"] >= THRESHOLDS["f1"]:
        print("Metrics are good. Deploying Flask...")

    else:
        print("Metrics below threshold. Recommend retraining or feature improvement.")

