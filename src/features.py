from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from config import settings

def get_pipeline_preprocessor():
    """
    Builds a ColumnTransformer to handle imputation and scaling simultaneously.
    
    - KNNImputer: Used for features with higher missing rates (e.g., GGT, ALP).
    - SimpleImputer (Mean): Used for standard blood count features.
    - RobustScaler: Handles outliers in clinical data.
    """
    
    # Sub-pipeline for KNN-based features
    knn_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=settings.KNN_NEIGHBORS)),
        ('scaler', RobustScaler())
    ])

    # Sub-pipeline for Mean-based features
    mean_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    # Identify which selected features belong to which group
    knn_feats = [f for f in settings.SELECTED_FEATURES if f in settings.KNN_IMPUTE_COLS]
    mean_feats = [f for f in settings.SELECTED_FEATURES if f in settings.MEAN_IMPUTE_COLS]

    # Combine branches into a single transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('knn_group', knn_pipeline, knn_feats),
            ('mean_group', mean_pipeline, mean_feats)
        ],
        remainder='drop' # Discard features not mentioned in settings
    )
    
    return preprocessor