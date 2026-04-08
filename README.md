# MeDas: Clinical Decision Support System for COVID-19 Prediction

MeDas is a machine learning project designed to analyze clinical data and support COVID-19 diagnosis prediction. This project was developed as part of **Project MeDas** (Applied Data Science in Healthcare) under the academic supervision of **Prof. Dr. Christina Bartenschlager**, utilizing clinical data from the **University Hospital Augsburg**.

The workflow focuses on robust preprocessing, data exploration, and providing a comparative evaluation between linear and ensemble methods for clinical decision-making.

------------------------------------------------------------------------

## Project Objectives

-   Analyze real-world clinical data from the **University Hospital Augsburg**.
-   Handle missing values through **KNN Imputation** to maintain dataset integrity.
-   Evaluate the trade-off between model transparency and predictive power.
-   Provide clinical insights using **SHAP (SHapley Additive exPlanations)** for model interpretability.

------------------------------------------------------------------------

## Technical Highlights

### Data Imputation & Preprocessing

-   **KNN Imputer**: Implemented to handle missing clinical values, ensuring the preservation of patient samples.
-   **Feature Engineering**: Includes numerical scaling (StandardScaler), categorical encoding, and systematic **Feature Importance** analysis to identify key clinical predictors.

### Model Benchmarking

The project compares two distinct approaches to identify the optimal model for a healthcare setting:

1.  **Logistic Regression**: Serves as a transparent, linear baseline for probabilistic risk assessments.
2.  **XGBoost Classifier**: Implemented to capture complex, non-linear relationships within the patient data.

------------------------------------------------------------------------

## Model Evaluation & Interpretation

Models were evaluated to ensure clinical reliability:

-   **Performance Metrics**: Analysis based on Precision, Recall, and F1-Score (with a focus on Recall to minimize undiagnosed cases).
-   **Explainable AI (XAI)**: Implementation of **SHAP values** to provide both global insights into feature impact and local explanations for individual patient predictions.

------------------------------------------------------------------------

## Project Structure

``` text

├── medas.ipynb     # Full research notebook: EDA, Preprocessing and Modeling
├── .gitignore      # Git exclusion file
└── README.md       # Project documentation
```

------------------------------------------------------------------------

## Tech Stack

-   **Language:** Python
-   **Core Libraries:** Scikit-Learn, XGBoost, SHAP, Pandas, NumPy, Matplotlib, Seaborn
-   **Environment:** Developed in Google Colab

------------------------------------------------------------------------

## Methodology Disclosure

This repository contains the source code and methodology used during the research. To comply with data protection regulations (**GDPR** / Hospital privacy policies), the raw data is not included, and the notebook outputs have been managed to ensure no Protected Health Information (PHI) is disclosed.

------------------------------------------------------------------------

## Acknowledgments & Credits

-   **Academic Supervision:** Prof. Dr. Christina Bartenschlager
-   **Data Source:** University Hospital Augsburg
-   **Author:** Daira Orlandini
