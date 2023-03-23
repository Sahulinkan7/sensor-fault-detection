import os

SAVED_MODEL_DIR=os.path.join("saved_models")
# defining common constant variable for training pipeline

PIPELINE_NAME: str = "sensor"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "sensor.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHMA_FILE_PATH= os.path.join("config","schema.yaml")
SCHEMA_DROP_COLUMNS="drop_columns"

# DATA INGESTION related variables starting with DATA_INGESTION variable name

DATA_INGESTION_COLLECTION_NAME : str = "aps_sensor_data"
DATA_INGESTION_DIR_NAME : str= "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float = 0.2

