from sensor.exception import SensorException
from sensor.logger import logging

import os,sys
import shutil

from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact

class ModelPusher:
    def __init__(self,model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def initaite_model_pusher(self)->ModelPusherArtifact:
        try:
            logging.info(f"{'#'*20} initiating Model Pusher component {'#'*20}")
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            model_file_path = self.model_pusher_config.model_file_path
            
            logging.info(" Copying trained model file into model_pusher directory ")
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path,dst=model_file_path)
            
            # saved model dir 
            logging.info(f" copying trained model file path into saved_model directory ")
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path,dst=saved_model_path)
            
            logging.info(f" model saved ")
            
            # prepare artifact 
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path,
                                                        pusher_model_path = model_file_path)
            
            logging.info(f" Model Pusher artifact : {model_pusher_artifact}")
            logging.info(f"{'#'*20} Model pusher completed {'#'*20}")
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e,sys) from e