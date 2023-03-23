from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys

from sensor.components.data_ingestion import DataIngestion

class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info(f"{'#'*20} Starting Data Ingestion {'#'*20}")
            
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"{'#'*20} Data Ingestion Completed and artifact : {data_ingestion_artifact} {'#'*20}")
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_data_validation(self):
        try:
            pass
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_data_tansformation(self):
        try:
            pass
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def start_model_trainer(self):
        try:
            pass
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_model_evaluation(self):
        try:
            pass
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def start_model_pusher(self):
        try:
            pass
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact:DataIngestionArtifact=self.start_data_ingestion()
        except Exception as e:
            raise SensorException(e,sys) from e
    