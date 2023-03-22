# from sensor.configuration.mongo_db_connection import MongoDBClient

# if __name__=='__main__':
#     mongodb_client=MongoDBClient("scania_aps_db")
#     print(mongodb_client.database.list_collection_names())
import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
def div():
    try:
        logging.info("logging this message")
        print(9/0)
    except Exception as e:
        raise SensorException(e,sys) from e
    
div()