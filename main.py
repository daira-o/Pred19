import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Internal imports
from config import settings
from src.preprocessing import clean_data
from src.train import train_with_grid_search

def main():
    # Set the experiment name in MLflow
    mlflow.set_experiment("MeDas_Full_System")

    # Start a parent run to group data analysis and model training
    with mlflow.start_run(run_name="Data_and_Model_Analysis"):
        
        # --- 1. DATA LOADING & CLEANING ---
        print("Loading and cleaning data...")
        df_raw = pd.read_csv(settings.RAW_DATA_PATH)
        df = clean_data(df_raw)
        
        # --- 2. CORRELATION ANALYSIS ---
        print("Performing correlation analysis...")
        pearson_corr = df.corr(method='pearson')[[settings.TARGET_COLUMN]].sort_values(by=settings.TARGET_COLUMN, ascending=False)
        spearman_corr = df.corr(method='spearman')[[settings.TARGET_COLUMN]].reindex(pearson_corr.index)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
        sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax1, cbar=False)
        ax1.set_title("Pearson Correlation\n(Linear)")
        sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax2)
        ax2.set_title("Spearman Correlation\n(Rank)")
        
        plt.tight_layout()
        mlflow.log_figure(fig, "data_analysis/correlation_comparison.png")
        plt.close()

        # --- 3. MUTUAL INFORMATION ---
        print("Calculating Mutual Information scores...")
        X_all = df.drop(columns=[settings.TARGET_COLUMN])
        y_all = df[settings.TARGET_COLUMN]
        
        # Select only numeric columns for MI and fill NaNs
        X_numeric = X_all.select_dtypes(include=[np.number])
        X_mi = X_numeric.fillna(X_numeric.median())
        
        # Identify discrete features (integers) vs continuous (floats)
        discrete_mask = X_mi.dtypes == "int64"

        mi_scores = mutual_info_classif(X_mi, y_all, discrete_features=discrete_mask, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_mi.columns).sort_values(ascending=False)

        plt.figure(figsize=(10, 10))
        sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, palette='viridis', legend=False)
        plt.title("Mutual Information Scores")
        mlflow.log_figure(plt.gcf(), "data_analysis/mutual_information.png")
        plt.close()

        # --- 4. TRAINING PREPARATION ---
        print("Splitting data for training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=settings.TEST_SIZE, 
            random_state=settings.RANDOM_STATE, stratify=y_all
        )

        # Define model experiments
        # Added max_iter for Logistic and updated XGB parameters for 2026 compatibility
        experiments = [
            {
                "name": "XGBoost", 
                "model": XGBClassifier(eval_metric='logloss', random_state=settings.RANDOM_STATE), 
                "grid": settings.XGB_GRID, 
                "threshold": 0.4
            },
            {
                "name": "Logistic_Regression", 
                "model": LogisticRegression(max_iter=2000, random_state=settings.RANDOM_STATE), 
                "grid": settings.LOGISTIC_GRID, 
                "threshold": 0.5
            }
        ]

        # --- 5. EXECUTE TRAINING ---
        for exp in experiments:
            print(f"\n>>> Starting training for: {exp['name']}")
            train_with_grid_search(
                X_train, y_train, X_test, y_test, 
                exp["model"], exp["grid"], exp["name"],
                threshold=exp["threshold"]
            )

    print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    main()