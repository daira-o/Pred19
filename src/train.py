import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Handle local module imports
try:
    from src.features import get_pipeline_preprocessor
except ImportError:
    print("Warning: src.features not found. Ensure your folder structure is correct.")

def train_with_grid_search(X_train, y_train, X_test, y_test, model_obj, param_grid, model_name, threshold=0.5):
    """
    Trains a model using Grid Search, handles SHAP interpretability, 
    and logs all results, metrics, and plots to MLflow.
    """
    
    # 1. Pipeline Setup
    pipeline = Pipeline([
        ('preprocessor', get_pipeline_preprocessor()),
        ('classifier', model_obj)
    ])

    # 2. Start MLflow Run
    with mlflow.start_run(run_name=f"Model_{model_name}", nested=True):
        
        # Grid Search Execution
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log Best Hyperparameters
        mlflow.log_params(grid_search.best_params_)
        
        # 3. Custom Threshold Prediction
        y_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        
        # 4. Log Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "threshold": threshold
        }
        mlflow.log_metrics(metrics)

        # 5. Artifact: Confusion Matrix
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        mlflow.log_figure(plt.gcf(), f"eval/cm_{model_name}.png")
        plt.close()

        # 6. SHAP Interpretability (THE BULLETPROOF VERSION)
        try:
            preprocessor = best_model.named_steps['preprocessor']
            classifier = best_model.named_steps['classifier']
            
            # 6a. Transform and force to DENSE NUMPY ARRAY
            X_test_tx = preprocessor.transform(X_test)
            if hasattr(X_test_tx, "toarray"):
                X_test_tx = X_test_tx.toarray()
            
            feature_names = preprocessor.get_feature_names_out()

            # 6b. Logic for XGBoost / Trees
            if "XGB" in str(type(classifier)) or "Forest" in str(type(classifier)):
                explainer = shap.TreeExplainer(classifier)
                # Pass directly as numpy array to avoid string formatting issues
                shap_values = explainer.shap_values(X_test_tx)
            
            # 6c. Logic for Logistic / Linear
            else:
                # Linear explainer needs a background summary (masker)
                X_train_tx = preprocessor.transform(X_train)
                if hasattr(X_train_tx, "toarray"): X_train_tx = X_train_tx.toarray()
                
                background = shap.maskers.Independent(X_train_tx)
                explainer = shap.LinearExplainer(classifier, background)
                shap_values = explainer.shap_values(X_test_tx)

            # 6d. Plotting
            plt.figure(figsize=(10, 6))
            # Use the feature_names we extracted earlier
            shap.summary_plot(shap_values, X_test_tx, feature_names=feature_names, show=False)
            
            mlflow.log_figure(plt.gcf(), f"eval/shap_{model_name}.png")
            plt.close()
            print(f"Successfully logged SHAP for {model_name}")

        except Exception as e:
            print(f"SHAP interpretation failed for {model_name}: {e}")

        # 7. Log Model to Registry
        input_example = X_test.iloc[:5] if isinstance(X_test, pd.DataFrame) else X_test[:5]
        
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path=f"model_{model_name}",
            input_example=input_example,
            registered_model_name=f"Registered_{model_name}"
        )
        
        print(f"Successfully logged model: {model_name}")

    return best_model