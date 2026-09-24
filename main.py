from src.data_loader import load_data
from src.eda import inspect_dataset
from src.preprocess import clean_and_preprocess
from src.train import train_models, save_best_model
from src.evaluate import evaluate_models

def main():
    # 1. Load Data
    print("\n--- Step 1: Loading Dataset ---")
    df = load_data()
    
    # 2. EDA Inspection
    print("\n--- Step 2: Exploratory Data Analysis ---")
    inspect_dataset(df)
    
    # 3. Preprocessing
    print("\n--- Step 3: Preprocessing Data ---")
    X_train, X_test, y_train, y_test, feature_names = clean_and_preprocess(df)
    print(f"Features: {len(feature_names)} | Training Rows: {len(X_train)} | Test Rows: {len(X_test)}")
    
    # 4. Training
    print("\n--- Step 4: Training Classification Models ---")
    models = train_models(X_train, y_train)
    
    # 5. Evaluation
    print("\n--- Step 5: Model Evaluation ---")
    results = evaluate_models(models, X_test, y_test)
    print("\nBenchmark Results:")
    print(results.to_string(index=False))
    
    # 6. Save Top Performer
    best_model_name = results.iloc[0]['Model']
    print(f"\nBest Model by ROC-AUC: {best_model_name}")
    save_best_model(models[best_model_name])

if __name__ == "__main__":
    main()