# Pred19: Clinical Decision Support System for COVID-19 Prediction

Pred19 is a machine learning pipeline designed to analyze clinical data and support COVID-19 diagnosis prediction. This project was developed as part of **Project MeDas** (Applied Data Science in Healthcare) under the academic supervision of **Prof. Dr. Christina Bartenschlager**, utilizing clinical data from the **University Hospital Augsburg**.

The workflow focuses on robust preprocessing, model interpretability, and providing a comparative evaluation between linear and ensemble methods for clinical decision-making.

---

## Project Objectives
- Build a reproducible ML pipeline using real-world clinical data from the **University Hospital Augsburg**.
- Handle missing values through advanced **KNN Imputation** to maintain dataset integrity.
- Evaluate the trade-off between model transparency and predictive power.
- Provide clinical insights using **SHAP (SHapley Additive exPlanations)** for model transparency.

---

## Technical Highlights

### Data Imputation & Preprocessing
- **KNN Imputer**: Implemented to handle missing clinical values, ensuring the preservation of patient samples that would otherwise be lost through row deletion.
- **Feature Engineering**: Includes numerical scaling (StandardScaler), categorical encoding, and systematic **Feature Importance** analysis to identify key clinical predictors.

### Model Benchmarking & Implementation
The project compares two distinct approaches to identify the optimal model for a healthcare setting:

1. **Logistic Regression**: Serves as a transparent, linear baseline. It is highly valued in clinical contexts for its interpretability and ability to provide probabilistic risk assessments.
2. **XGBoost Classifier**: It was implemented to capture complex, non-linear relationships within the patient data that linear models might overlook.

---

## Model Evaluation & Interpretation
Models were evaluated to ensure clinical reliability:

- **Performance Metrics**: Analysis based on Precision, Recall, and F1-Score (with a focus on Recall to minimize undiagnosed cases).
- **Explainable AI (XAI)**: Implementation of **SHAP values** to provide both global insights into feature impact and local explanations for individual patient predictions.

---

## Project Evolution: From Research to Production

This repository represents a transition from research-oriented exploration to a modular production-ready architecture.

### Project Status

1. **Phase 1: Research (Notebook) — COMPLETED**  
   The `academic_notebook.ipynb` contains the full Exploratory Data Analysis (EDA), model benchmarking, and initial SHAP visualizations.

2. **Phase 2: Modularization (Python Scripts) — IN PROGRESS**  
   Refactoring core logic into decoupled Python scripts (`src/` directory) to improve maintainability and automate the pipeline execution.

3. **Phase 3: MLOps Infrastructure — IN PROGRESS**  
   Implementing `uv` for modern dependency management and preparing the environment for automated reproducibility and experiment tracking.

---

## Project Structure (Work in Progress)

```text
PRED19/
├── notebooks/              # Original academic research and experiments
├── src/                    # [WIP] Modularized source code
│   ├── preprocessing.py    # Data cleaning and imputation logic
│   ├── features.py         # Feature engineering and encoding
│   └── train.py            # Model training and serialization
├── main.py                 # [WIP] Pipeline orchestrator (Entry point)
├── pyproject.toml          # Project metadata and dependencies (managed via uv)
├── uv.lock                 # Reproducible lockfile for environment consistency
└── README.md               # Project documentation
```
---

# Tech Stack

- **Language:** Python 3.12.x
- **Dependency Management:** uv (transitioning to fully reproducible environments)
- **Core Libraries:** Scikit-Learn, XGBoost, SHAP, Pandas, NumPy
- **Environment:** Developed in Google Colab (Research phase) and refactored for local execution

---

## Methodology Disclosure
This repository contains the source code and methodology used during the research. To comply with data protection regulations (GDPR/Hospital privacy policies), the raw data is not included, and the notebook outputs have been managed to ensure no Protected Health Information (PHI) is disclosed.

---
- **Academic Supervision**: Prof. Dr. Christina Bartenschlager.
- **Data Source**: University Hospital Augsburg.
- **Author**: Daira Orlandini
