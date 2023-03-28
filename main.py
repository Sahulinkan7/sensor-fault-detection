import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.pipeline.training_pipeline import TrainPipeline
from app import app


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.info(f"{e}")
        print(str(e))
