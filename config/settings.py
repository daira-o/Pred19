# config/settings.py

# =============================================================================
# DATA PATHS
# =============================================================================
RAW_DATA_PATH = "data/raw/data.csv"
PROCESSED_DATA_PATH = "data/processed/data_clean.csv"

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

# 1. Non-clinical columns to drop immediately
DROP_COLUMNS = ['Patient', 'Unnamed: 0', 'Suspect']

# 2. Special Columns
TARGET_COLUMN = "target"
SEX_COLUMN = "Sex"

# 3. All numerical clinical features
NUMERIC_COLS = [
    "Age", "CA", "CK", "CREA", "ALP", "GGT", "GLU", "AST", "ALT", "LDH", "PCR",
    "KAL", "NAT", "UREA", "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC",
    "PLT1", "NE", "LY", "MO", "EO", "BA", "NET", "LYT", "MOT", "EOT", "BAT"
]

# 4. Final selection for model training (Based on your Notebook)
SELECTED_FEATURES = ['PCR', 'LDH', 'WBC', 'CA', 'HCT', 'EO']

# =============================================================================
# IMPUTATION STRATEGY
# =============================================================================

# Columns to impute using KNN (Missing > 20%)
KNN_IMPUTE_COLS = [
    'ALP', 'GGT', 'BA', 'NET', 'LYT', 'MOT', 'NE', 
    'LY', 'BAT', 'EOT', 'MO', 'EO'
]

# Columns to impute using Mean (Missing 5-16%)
MEAN_IMPUTE_COLS = [
    'LDH', 'AST', 'GLU', 'PCR', 'ALT', 'CA', 'KAL', 'CREA',
    'NAT', 'HCT', 'HGB', 'RBC', 'WBC', 'MCV', 'MCH', 'MCHC', 
    'PLT1', 'Age'
]

# =============================================================================
# MODEL HYPERPARAMETERS (Fixed Values)
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
KNN_NEIGHBORS = 5
CLINICAL_THRESHOLD = 0.4 

XGB_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 200,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

LOGISTIC_PARAMS = {
    'C': 1.0,
    'solver': 'liblinear',
    'max_iter': 1000,
    'class_weight': 'balanced'
}

# =============================================================================
# GRID SEARCH SPACES 
# =============================================================================

# Search space for XGBoost
# Note: 'classifier__' prefix is required for Pipeline tuning
XGB_GRID = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1]
}

# Search space for Logistic Regression
LOGISTIC_GRID = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear']
}