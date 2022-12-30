from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
from sensor.components.data_ingestion import DataIngestionComponent
from sensor.components.data_validation import DataValidationComponent
from sensor.components.data_transformation import DataTranformationComponent
from sensor.components.model_trainer import ModelTrainerComponent
from sensor.components.model_evaluation import ModelEvaluationComponent
from sensor.components.model_pusher import ModelPusher
import os
import sys


class TrainingPipeline:
    is_pipeline_running = False
    def __init__(self, trainin_pipeline_config: TrainingPipelineConfig):
        training_pipeline_config = TrainingPipelineConfig()
        self.training_pipeline_config = trainin_pipeline_config

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(
                self.training_pipeline_config)
            self.data_ingestion_config = data_ingestion_config
            data_ingestion = DataIngestionComponent(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_data_validation(self, data_ingestion_artifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                self.training_pipeline_config)
            data_validation = DataValidationComponent(data_validation_config=data_validation_config,
                                                    data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_data_transformation(self,data_validation_artifact)->DataTransformationArtifact:
        try:
            data_tranformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTranformationComponent(data_validation_artifact=data_validation_artifact, data_transformation_config=data_tranformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                self.training_pipeline_config)
            model_trainer = ModelTrainerComponent(
                data_transformation_artifact=data_transformation_artifact, model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
    def start_model_evaluation(self,model_trainer_artifact:ModelTrainerArtifact,data_validation_artifact:DataValidationArtifact)->ModelEvaluationArtifact:
        try:

            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config = self.training_pipeline_config)
            model_evaluation = ModelEvaluationComponent(model_evaluation_config = model_evaluation_config,model_trainer_artifact = model_trainer_artifact,data_validation_artifact = data_validation_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config,model_eval_artifact=model_evaluation_artifact)
            model_evaluation_artifact = model_pusher.initiate_model_pusher()
            return model_evaluation_artifact
            
        except Exception as e:
            raise SensorException(e,sys)

    def run_pipeline(self):
        try:
            TrainingPipeline.is_pipeline_running = True
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact: DataTransformationArtifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact)

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,data_validation_artifact = data_validation_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            self.start_model_pusher(model_evaluation_artifact = model_evaluation_artifact)
            TrainingPipeline.is_pipeline_running = False
        except Exception as e:
            raise SensorException(e, sys)
        

