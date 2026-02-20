import pandas as pd
import numpy as np
from config import settings

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning to ensure types are consistent.
    
    1. Removes noise columns (IDs, suspect cases).
    2. Corrects European decimal formats (commas to dots).
    3. Forces numeric types for medical blood parameters.
    4. Drops duplicate patient entries.
    """
    df = df.copy()
    
    # Drop columns defined in settings that aren't useful for modeling
    df = df.drop(columns=[c for c in settings.DROP_COLUMNS if c in df.columns], errors='ignore')
    
    # Process numeric columns: Replace commas and convert to float32
    for col in settings.NUMERIC_COLS:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("float32")

    # Ensure categorical/binary columns are integer-based
    if settings.SEX_COLUMN in df.columns:
        df[settings.SEX_COLUMN] = pd.to_numeric(df[settings.SEX_COLUMN], errors="coerce").astype("Int32")
    
    if settings.TARGET_COLUMN in df.columns:
        df[settings.TARGET_COLUMN] = pd.to_numeric(df[settings.TARGET_COLUMN], errors="coerce").astype("Int32")
        
    df.drop_duplicates(inplace=True)
    return df