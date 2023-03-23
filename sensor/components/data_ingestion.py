from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from pandas import DataFrame
from sensor.data_access.sensor_data import SensorData
from sklearn.model_selection import train_test_split
class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
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
            
            #creating folder
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def split_data_as_train_test(self,dataframe:DataFrame)->None:
        '''
        Feature store data will be split into train and test file
        train.csv and test.csv
        '''
        try:
            logging.info(f"{'#'*10} Performing train test split on the dataframe {'#'*10}")
            train_set,test_set=train_test_split(dataframe,
                                                test_size=self.data_ingestion_config.train_test_split_ratio)
        
            logging.info(f"{'#'*10} Data split as train and test file done {'#'*10}")
            
            dir_path=os.path.dirname(self.data_ingestion_config.trainig_file_path)
            os.makedirs(dir_path,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.trainig_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            
            logging.info(f"{'#'*10} Exported train and test file path {'#'*10}")
            
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            dataframe=self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.trainig_file_path,
                                                          test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys) from e