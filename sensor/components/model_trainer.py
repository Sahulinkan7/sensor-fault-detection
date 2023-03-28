
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys

from xgboost import XGBClassifier

from sensor.utils.main_utils import load_numpy_array_data,load_object,save_object
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel

from sensor.entity.config_entity import ModelTrainerConfig
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact,ClassificationMetricArtifact

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def train_model(self,x_train,y_train):
        try:
            logging.info(f" started trainig model with input x_train and  y_train data")
            
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train,y_train)
            
            logging.info(f" model trainig completed ")
            return xgb_clf
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"{'#'*20} Initiating model trainer component {'#'*20}")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # loading training and testing array
            
            logging.info(f" Loading train and test numpy array data ")
            
            train_arr = load_numpy_array_data(file_path=train_file_path)
            test_arr = load_numpy_array_data(file_path=test_file_path)
            
            logging.info(f" splitting train and test array data into input and output array for training ")
            x_train,y_train,x_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
                        
            model = self.train_model(x_train,y_train)
            y_train_pred = model.predict(x_train)
            
            logging.info(f" getting model accuracy for the trained data ")
            classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            
            logging.info(f" model accuracy for trained data is {classification_train_metric}")
            if classification_train_metric.f1_score <= self.model_trainer_config.model_expected_accuracy:
                raise Exception(" Trained model is not good to provide expected accuracy ")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)
            logging.info(f" model accuracy for test data is {classification_test_metric}")
            
            # checking overfitting and underfitting 
            logging.info(f" checking for overfitting and underfitting of the model ")
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            
            logging.info(f" model accuracy difference for train and test data is {diff}")
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(" Model is not good try to do more experiment ")
            
            logging.info(f" loading transformation object for creating model trainer object ")
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            sensor_model = SensorModel(preprocessor= preprocessor, model= model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=sensor_model)
            logging.info(f" newly created trained model saved at file path : {self.model_trainer_config.trained_model_file_path}")
                    
            # model trainer artifact
            logging.info(f" creating model trainer artifact ")
            model_trainer_artifact = ModelTrainerArtifact(
                                        trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                        train_metric_artifact= classification_train_metric,
                                        test_metric_artifact= classification_test_metric
                                        )
            
            logging.info(f" Model trainer artifact : {model_trainer_artifact}")
            logging.info(f"{'#'*20} Model Trainer Completed {'#'*20}")
            return model_trainer_artifact       
            
        except Exception as e:
            raise SensorException(e,sys) from e