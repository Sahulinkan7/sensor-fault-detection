# from sensor.configuration.mongo_db_connection import MongoDBClient

# if __name__=='__main__':
#     mongodb_client=MongoDBClient("scania_aps_db")
#     print(mongodb_client.database.list_collection_names())
import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from sensor.pipeline.training_pipeline import TrainPipeline

t=TrainPipeline()
t.run_pipeline()