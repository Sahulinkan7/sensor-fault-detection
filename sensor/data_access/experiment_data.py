from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException

from sensor.constant.database import DATABASE_NAME

from typing import Optional

import os,sys
import pandas as pd
class Experiment_save:
    def __init__(self):
        self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

    def save_data(self,dictd : dict):
        self.mongo_client.database['experiments'].insert_many([dictd])
        
    def update_data(self,id,data:dict):
        self.mongo_client.database['experiments'].update_one({"experiment_id":id},{"$set":data})
    
    def read_experiments(self):
        try:
            collection = self.mongo_client.database['experiments']
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            return df
        except Exception as e:
            raise SensorException(e,sys) from e