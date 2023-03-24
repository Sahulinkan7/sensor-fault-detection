from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig
from sensor.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys

from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation

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
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info(f"{'#'*20} Starting Data Validation {'#'*20}")
            
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logging.info(f" Data Validation artifact : {data_validation_artifact} ")
            
            return data_validation_artifact
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
            data_validation_artifact: DataValidationArtifact = self.start_data_validation(
                                                data_ingestion_artifact = data_ingestion_artifact)
        except Exception as e:
            raise SensorException(e,sys) from e
    