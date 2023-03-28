
from sensor.exception import SensorException
from sensor.logger import logging

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH

from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig

from scipy.stats import ks_2samp
from sensor.utils.main_utils import read_yaml_file,write_yaml_file
import pandas as pd
import os,sys

class DataValidation:
    
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys)
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns= len(self._schema_config["columns"])
            
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        
        except Exception as e:
            raise SensorException(e,sys)
    
    def is_numerical_column_exists(self,dataframe : pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns=dataframe.columns
            
            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present=False
                    missing_numerical_columns.append(num_column)
            
            logging.info(f" Missing numerical columns list : {missing_numerical_columns}")
            return numerical_column_present

        except Exception as e:
            raise SensorException(e,sys) from e
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            logging.info(f" Reading the pandas dataframe present in file path {file_path}")
            dataframe=pd.read_csv(file_path)
            return dataframe
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            logging.info(" checking data drift for each column between base dataset and current dataset ")
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dict=ks_2samp(d1,d2)
                if threshold<=is_same_dict.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dict.pvalue),
                    "drift_status": is_found
                    }})
            
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            logging.info(f" Saving drift report at the file path : {drift_report_file_path}")
            
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            logging.info(f" saved report file at the location {drift_report_file_path}")
            logging.info(f" Drift checked and found to be {status}")
            return status
                
        except Exception as e:
            raise SensorException(e,sys) from e    
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info(f"{'#'*20} Initiating Data Validation {'#'*20}")
            error_message=""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            # Reading train and test file
            logging.info(f" Reading training and testing dataframe ")
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            # Validate number of columns 
            logging.info(f" Validating number of columns of train dataframe ")
            status=self.validate_number_of_columns(dataframe=train_dataframe)
            
            if not status:
                error_message=f"{error_message} Train dataframe does not contain all columns\n"
            logging.info(f" Training dataframe number of columns validated and the status is {status}")
            status=self.validate_number_of_columns(dataframe=test_dataframe)
            
            
            if not status:
                error_message=f"{error_message} Test dataframe does not contain all columns\n"
            logging.info(f" Testing dataframe number of columns validated and the status is {status}")
                
            # Validate numerical columns
            logging.info(f" Validating numerical columns of training dataframe ")            
            status = self.is_numerical_column_exists(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message} Train data frame does not contain all numerical columns\n"
            logging.info(f" Trainig dataframe numerical columns validated and status is {status}")
            
            
            status=self.is_numerical_column_exists(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message} Test data frame does not contain all numerical columns\n"
            logging.info(f" Testing dataframe numerical columns validated and the status is {status}")
            if len(error_message)>0:
                raise Exception(error_message)
            
            
            # check data drift 
            
            logging.info(f" Checking data drift in training and testing dataframe ")
            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            
            logging.info(" Data drift checked and the status found to be : {status}")
            
            data_validation_artifact = DataValidationArtifact(
                                        validation_status = status,
                                        valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                                        valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                                        invalid_train_file_parth = None,
                                        invalid_test_file_path = None,
                                        drift_report_file_path = self.data_validation_config.drift_report_file_path
                                        )   
            
            logging.info(f" Data validation artifact is : {data_validation_artifact}")       
            logging.info(f"{'#'*20} Data Validation completed {'#'*20}")  
            return data_validation_artifact          
            
        except Exception as e:
            raise SensorException(e,sys) from e