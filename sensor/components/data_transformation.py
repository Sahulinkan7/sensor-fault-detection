from sensor.exception import SensorException
from sensor.logger import logging
import os,sys

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.config_entity import DataTransformationConfig
from sensor.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from sensor.ml.model.estimator import TargetValueMapping

from sensor.utils.main_utils import save_numpy_array_data,save_object
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek

class DataTransformation:
    
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config= data_transformation_config
        except Exception as e:
            raise SensorException(e,sys) from e
        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            logging.info(" creating data transformer object ....")
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            
            logging.info(f" creating pipeline for preprocessor object ")
            # create pipeline for preprocessor
            preprocessor = Pipeline(
                steps=[
                    ("Imputer",simple_imputer), # replaces missing value with 0
                    ("RobustScaler",robust_scaler) # maintains outlier by keeping all the features value in one range
                ]
            )
            
            logging.info(f" pipeline for preprocessor object created . ")
            
            return preprocessor
        
        except Exception as e:
            raise SensorException(e,sys) from e
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            logging.info(f" reading pandas dataframe from file_path : {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"{'#'*20} Initiating data Transformation {'#'*20}")
            
            logging.info(f" reading training and testing dataframe ")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info(f" getting data transformation object ")
            preprocessor=self.get_data_transformer_object()
            
            logging.info(f" data transformation object saved ")
            
            # training dataframe
            logging.info(f" dropping target column from train dataframe ")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())
            logging.info(f" target column {TARGET_COLUMN} dropped from train dataframe")
            logging.info(f" input feature train dataframe and target feature train dataframe created separately ")
            
            # testing dataframe
            logging.info(f" dropping target column from test dataframe ")
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            logging.info(f" target column {TARGET_COLUMN} dropped from test dataframe ")
            logging.info(f" input feature test dataframe and target feature test dataframe created separately ")
            
            logging.info(f" creating preprocessor object by fitting input train dataframe with preprocessor pipeline object")            
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            logging.info(" input feature of train and test dataframe got transformed ")
            
            
            smt= SMOTETomek(sampling_strategy='minority')
            logging.info(" resampling the train and test dataframe ")
            
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            
            logging.info(f" concatinating the trasformed input individual train and test array ")
            
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # save numpy array data
            
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor_object)
            logging.info(f" saving the transformation object at {self.data_transformation_config.transformed_object_file_path}")
            
            # Data Transformation Artifact
            data_transformation_artifact = DataTransformationArtifact(
                                            transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                                            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                                            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path 
                                            )
            logging.info(f" data transformation artifact is : {data_transformation_artifact}")
            logging.info(f"{'#'*20} Data Transformation Completed {'#'*20}")
                        
            return data_transformation_artifact   
            
        except Exception as e:
            raise SensorException(e,sys) from e