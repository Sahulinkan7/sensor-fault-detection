from sensor.exception import SensorException
from sensor.logger import logging
import os,sys

from sensor.utils.main_utils import load_object,write_yaml_file

from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.entity.artifact_entity import DataValidationArtifact,ModelEvaluationArtifact,ModelTrainerArtifact
from sensor.ml.model.estimator import ModelResolver
import pandas as pd
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.model.estimator import TargetValueMapping
from sensor.ml.metric.classification_metric import get_classification_score


class ModelEvaluation:
    def __init__(self,model_trainer_artifact : ModelTrainerArtifact,
                 data_validation_artifact : DataValidationArtifact,
                 model_evaluation_config : ModelEvaluationConfig):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def initiate_model_evaluation(self)-> ModelEvaluationArtifact:
        try:
            valid_train_filepath = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path
            
            # valid train and test file path
            train_df = pd.read_csv(valid_train_filepath)
            test_df = pd.read_csv(valid_test_file_path)
            
            df = pd.concat([train_df,test_df])
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(),inplace=True)
            df.drop(TARGET_COLUMN,axis=1,inplace=True)
            
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True
            
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                                            is_model_accepted=is_model_accepted,
                                            improved_accuracy = None,
                                            best_model_path= None,
                                            trained_model_path= trained_model_file_path,
                                            trained_model_metric_artifact= self.model_trainer_artifact.train_metric_artifact,
                                            best_model_metric_artifact=None
                                            )
                return model_evaluation_artifact
            
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(latest_model_path)
            
            # loading trained model
            
            trained_model = load_object(file_path= trained_model_file_path)
            
            y_trained_pred = trained_model.predict(df)
            y_latest_pred = latest_model.predict(df)
            
            trained_metric = get_classification_score(y_true=y_true,y_pred=y_trained_pred)
            latest_metric = get_classification_score(y_true=y_true,y_pred=y_latest_pred)
            
            imporoved_accuracy = trained_metric.f1_score - latest_metric.f1_score
            
            if self.model_evaluation_config.changed_threshold < imporoved_accuracy:
                is_model_accepted  = True
            else:
                is_model_accepted = False
                
            model_evaluation_artifact = ModelEvaluationArtifact(
                                        is_model_accepted= is_model_accepted,
                                        improved_accuracy= imporoved_accuracy,
                                        best_model_path= latest_model_path,
                                        trained_model_path= trained_model_file_path,
                                        trained_model_metric_artifact= trained_metric,
                                        best_model_metric_artifact= latest_metric       
                                        )
            
            model_eval_report = model_evaluation_artifact.__dict__
            
            # save the report
            
            write_yaml_file(self.model_evaluation_config.evaluation_report_file_path,model_eval_report)
            logging.info(f"Model Evaluation Artifact : {model_evaluation_artifact}")
            return model_evaluation_artifact
                    
        except Exception as e:
            raise SensorException(e,sys) from e