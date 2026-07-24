from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_TITLE = "PRED19 Clinical Assistance Monitor"
MODEL_PATH = ROOT_DIR / "model" / "pred19_pipeline.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "model_metrics.json"
ROC_PATH = ROOT_DIR / "artifacts" / "roc_curve.csv"
DEMO_DATA_PATH = ROOT_DIR / "inference" / "demo_synthetic.csv"
MAX_UPLOAD_MB = 10
TIMESTAMP_COLUMNS = ("timestamp", "observation_timestamp", "datetime", "date_time")
