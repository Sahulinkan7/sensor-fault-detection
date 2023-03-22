import logging

from datetime import datetime

import os

CURRENT_TIME_STAMP=datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
LOG_DIR='Sensor_log'
LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"

os.makedirs(LOG_DIR,exist_ok=True)
LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE_NAME)
logging.basicConfig(filemode='w',filename=LOG_FILE_PATH,level=logging.INFO,
                    format="[ %(asctime)s ] file_name : %(filename)s at line number : %(lineno)d %(name)s - %(levelname)s - log_message : %(message)s")