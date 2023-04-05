from sensor.entity.config_entity import (TrainingPipelineConfig,DataIngestionConfig,
                                         DataValidationConfig,DataTransformationConfig,
                                         ModelTrainerConfig,ModelEvaluationConfig,
                                         ModelPusherConfig)
from sensor.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact,
                                           DataTransformationArtifact,ModelTrainerArtifact,
                                           ModelEvaluationArtifact,ModelPusherArtifact)
import pandas as pd
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from datetime import datetime,timedelta
from sensor.data_access.experiment_data import Experiment_save

from threading import Thread
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

from sensor.cloud_storage.s3_syncer import S3sync
import uuid

from collections import namedtuple
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher

Experiment=namedtuple("Experiment",["experiment_id","initialization_timestamp","artifact_time_stamp",
                                    "running_status","start_time","stop_time","execution_time","message",
                                    "accuracy","is_model_accepted"])

class TrainPipeline(Thread):
    experiment: Experiment = Experiment(*([None]*10))
    experiment_file_path = None
        
    def __init__(self):
        try:
            super().__init__(daemon=False,name="Train-pipeline-Thread")
            self.training_pipeline_config=TrainingPipelineConfig(timestamp=datetime.now())
            self.s3_sync=S3sync()
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:            
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:            
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()            
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_data_tansformation(self,data_validation_artifact: DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()            
            return data_transformation_artifact
        
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def start_model_trainer(self,data_transformation_artifact: DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config = model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            return model_trainer_artifact
        
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def start_model_evaluation(self,model_trainer_artifact: ModelTrainerArtifact,
                               data_validation_artifact: DataValidationArtifact)->ModelEvaluationArtifact:
        try:
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvaluation(model_trainer_artifact=model_trainer_artifact,
                                               data_validation_artifact=data_validation_artifact,
                                               model_evaluation_config=model_evaluation_config)

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            
            return model_evaluation_artifact
        
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def start_model_pusher(self,model_evaluation_artifact: ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config= self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                       model_evaluation_artifact=model_evaluation_artifact)
            model_pusher_artifact = model_pusher.initaite_model_pusher()
            
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url= f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder=SAVED_MODEL_DIR,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def run_pipeline(self):
        try:
            if TrainPipeline.experiment.running_status:
                logging.info(f"Pipeline is already Running ")
                return TrainPipeline.experiment
            logging.info(f" Starting Pipeline ")
            
            experiment_id = str(uuid.uuid4())
            
            TrainPipeline.experiment=Experiment(experiment_id = experiment_id,
                                                initialization_timestamp=self.training_pipeline_config.timestamp,
                                                artifact_time_stamp=self.training_pipeline_config.timestamp,
                                                running_status=True,
                                                start_time=datetime.now(),
                                                stop_time=None,
                                                execution_time=None,
                                                message="Pipeline has started",
                                                accuracy=None,
                                                is_model_accepted=None                                             
                                                )
            logging.info(f"Pipeline Experiment : {TrainPipeline.experiment}")
            
            exp=TrainPipeline.experiment._asdict()
            logging.info(f"{exp}")
            experiment_save=Experiment_save()
            experiment_save.save_data(exp)
            data_ingestion_artifact:DataIngestionArtifact=self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact = self.start_data_validation(
                                                data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact: DataTransformationArtifact = self.start_data_tansformation(
                                                data_validation_artifact=data_validation_artifact)
            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(
                                                data_transformation_artifact= data_transformation_artifact)
            model_evaluation_artifact: ModelEvaluationArtifact = self.start_model_evaluation(
                                                model_trainer_artifact = model_trainer_artifact,
                                                data_validation_artifact=data_validation_artifact)

            if not model_evaluation_artifact.is_model_accepted:
                stop_time=datetime.now()
                TrainPipeline.experiment=Experiment(experiment_id=TrainPipeline.experiment.experiment_id,
                                                    initialization_timestamp=self.training_pipeline_config.timestamp,
                                                    artifact_time_stamp=self.training_pipeline_config.timestamp,
                                                    running_status=False,
                                                    start_time=TrainPipeline.experiment.start_time,
                                                    stop_time=stop_time,
                                                    execution_time=timedelta.total_seconds(stop_time-TrainPipeline.experiment.start_time),
                                                    message = "Trained model not better than best Model",
                                                    accuracy = model_trainer_artifact.train_metric_artifact.f1_score,
                                                    is_model_accepted=model_evaluation_artifact.is_model_accepted
                                                    )
                exp = TrainPipeline.experiment._asdict()
                logging.info(f"{exp}")
                experiment_save.update_data(id=TrainPipeline.experiment.experiment_id,data=exp)
                logging.info(f" Pipeline Experiment : {TrainPipeline.experiment}")
                raise Exception("Trained model is not better than the best model ")
            model_pusher_artifact : ModelPusherArtifact = self.start_model_pusher(
                                                model_evaluation_artifact=model_evaluation_artifact)
            stop_time=datetime.now()
            
            TrainPipeline.experiment=Experiment(experiment_id=TrainPipeline.experiment.experiment_id,
                                                initialization_timestamp=self.training_pipeline_config.timestamp,
                                                artifact_time_stamp=self.training_pipeline_config.timestamp,
                                                running_status=False,
                                                start_time=TrainPipeline.experiment.start_time,
                                                stop_time=stop_time,
                                                execution_time=timedelta.total_seconds(stop_time-TrainPipeline.experiment.start_time),
                                                message = "Pipeline completed",
                                                accuracy=model_trainer_artifact.train_metric_artifact.f1_score,
                                                is_model_accepted=model_evaluation_artifact.is_model_accepted
                                                )
            exp = TrainPipeline.experiment._asdict()
            logging.info(f"{exp}")
            experiment_save.update_data(id=TrainPipeline.experiment.experiment_id,data=exp)
            logging.info(f" Pipeline Experiment : {TrainPipeline.experiment}")
            
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
        except Exception as e:
            self.sync_artifact_dir_to_s3()
            raise SensorException(e,sys) from e
    
    @staticmethod       
    def get_experiments_status(limit:int = 6) -> pd.DataFrame:
        try:
            df = Experiment_save().read_experiments()
            limit= -1*limit
            if len(df.columns)==0:
                return None
            df = df[limit:].drop(columns=["initialization_timestamp"],axis=1)
            return df
            
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise SensorException(e,sys) from e
    