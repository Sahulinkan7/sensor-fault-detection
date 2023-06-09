from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from pandas import DataFrame

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.data_access.sensor_data import SensorData
from sklearn.model_selection import train_test_split
from sensor.utils.main_utils import read_yaml_file
class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def export_data_into_feature_store(self)->DataFrame:
        '''
        export mongodb collection data as dataframe into feature store
        '''
        try:
            logging.info(f"{'#'*10} Exporting data from mongodb collection to FeatureStore {'#'*10}")
            sensor_data=SensorData()
            dataframe=sensor_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            
            logging.info(f" Creating Feature store directory ")
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f" Feature store directory created at location {dir_path}")
            
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info(f" data frame saved at {feature_store_file_path}")
            
            return dataframe
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def split_data_as_train_test(self,dataframe:DataFrame)->None:
        '''
        Feature store data will be split into train and test file
        train.csv and test.csv
        '''
        try:
            logging.info(f" Performing train test split on the dataframe ")
            train_set,test_set=train_test_split(dataframe,
                                                test_size=self.data_ingestion_config.train_test_split_ratio)
        
            logging.info(f" Data split as train and test file done ")
            
            dir_path=os.path.dirname(self.data_ingestion_config.trainig_file_path)
            os.makedirs(dir_path,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.trainig_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            
            logging.info(f" Exported train and test file path ")
            
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info(f"{'#'*20} Initiating Data Ingestion {'#'*20}")
            dataframe=self.export_data_into_feature_store()
            dataframe=dataframe.drop(self._schema_config['drop_columns'],axis=1)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.trainig_file_path,
                                                          test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f" Data ingestion artifact is : {data_ingestion_artifact}")
            logging.info(f"{'#'*20} Data Ingestion completed {'#'*20}")
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys) from e