from src.utils.config import config
from src.train import load_data, clean_data, preprocess_data, train_model, select_and_save_best_model, evaluate_model, make_decision #xgb_feature_importance, save_model, model_registry

    
def main():

    data_path = config['data']['raw_data_path']
    df = load_data(data_path)
    df_clean = clean_data(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_clean)

    results = train_model(X_train, y_train, preprocessor)

    best_model, best_score = select_and_save_best_model(results, X_test, y_test)

    eval_metrics = evaluate_model(best_model, X_test, y_test)
    print("Best Model Evaluation:", eval_metrics)

    make_decision(eval_metrics)


if __name__ == "__main__":
    main()
