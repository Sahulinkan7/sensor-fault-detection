import os

SAVED_MODEL_DIR=os.path.join("saved_models")
# defining common constant variable for training pipeline

TARGET_COLUMN: str = "class"
PIPELINE_NAME: str = "sensor"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "sensor.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH= os.path.join("config","schema.yaml")
SCHEMA_DROP_COLUMNS="drop_columns"

# DATA INGESTION related variables starting with DATA_INGESTION variable name

DATA_INGESTION_COLLECTION_NAME : str = "aps_sensor_data"
DATA_INGESTION_DIR_NAME : str= "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float = 0.2

# DATA VALIDATION related variables starting with DATA_VALIDATION variable name

DATA_VALIDATION_DIR_NAME: str= "data_validation"
DATA_VALIDATION_VALID_DIR: str= "validated"
DATA_VALIDATION_INVALID_DIR: str= "Invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str= "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str= "report.yaml"


# Data Transformation related variables starting with DATA_TRANSFORMATION variable name

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANFORMED_OBJECT_DIR: str = "transformed_object"

# Model Training related variables starting with MODEL_TRAINER variable name

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

# Model evaluation related variables starting with MODEL_EVALUATION variable name 

MODEL_EVALUATION_DIR_NAME: str ="model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: str = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "evaluation_report.yaml"

# Model pusher related variables starting with MODEL_PUSHER variable name

MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = SAVED_MODEL_DIR
