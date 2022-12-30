from sensor.logger import logging
from sensor.exception import SensorException
from sensor.pipeline.training_pipeline import TrainingPipeline
from sensor.entity.config_entity import TrainingPipelineConfig
import sys
from fastapi import FastAPI
from sensor.constants import APP_HOST,APP_PORT



if __name__=="__main__":
     try:
          training_pipeline_config = TrainingPipelineConfig()
          training_pipeline = TrainingPipeline(training_pipeline_config)
          training_pipeline.run_pipeline()
     except Exception as e:
          raise SensorException(e, sys)

     