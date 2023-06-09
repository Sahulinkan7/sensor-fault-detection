import pandas as pd
import numpy as np
from typing import Optional
import json
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.exception import SensorException
from sensor.logger import logging
import sys
class SensorData:
    '''
    this class helps to extract entire record of a mongo db database collection as dataframe
    '''
    def __init__(self):
        try:
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise SensorException(e,sys) from e
    
    def export_collection_as_dataframe(self,collection_name :str,database_name: Optional[str]=None)->pd.DataFrame:
        '''
        exports entire collection as dataframe
        return pandas dataframe
        '''
        try:
            if database_name is None:
                collection=self.mongo_client.database[collection_name]
            else:
                collection=self.mongo_client[database_name][collection_name]
            logging.info(f" getting data from the collection named : {collection}")
            df=pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise SensorException(e,sys) from e