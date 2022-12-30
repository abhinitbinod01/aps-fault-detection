import os,sys
from datetime import datetime
from sensor.constants.database import DATABASE_NAME,COLLECTION_NAME
from sensor.constants.training_pipeline import *

class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name:str = PIPELINE_NAME
        self.timestamp:str = TIMESTAMP
        self.artifact_dir:str = os.path.join(self.pipeline_name,ARTIFACT_DIR,self.timestamp)
       

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.database_name:str = DATABASE_NAME
        self.collection_name:str = DATA_INGESTION_COLLECTION_NAME
        self.data_ingestion_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
            )
        self.feature_store_file_path:str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME)
        self.training_file_path:str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TRAIN_FILE_NAME)
        self.testing_file_path:str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TEST_FILE_NAME)
        self.train_test_split_ratio:float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME
            )
        self.valid_train_file_path:str = os.path.join(
            self.data_validation_dir,DATA_VALIDATION_VALID_DIR,TRAIN_FILE_NAME
            )
        self.valid_test_file_path:str = os.path.join(
            self.data_validation_dir,DATA_VALIDATION_VALID_DIR,TEST_FILE_NAME
            )
        self.invalid_train_file_path:str = os.path.join(
            self.data_validation_dir,DATA_VALIDATION_INVALID_DIR,TRAIN_FILE_NAME
            )
        self.invalid_test_file_path:str = os.path.join(
            self.data_validation_dir,DATA_VALIDATION_INVALID_DIR,TEST_FILE_NAME
            )
        self.drift_report_file_path:str = os.path.join(
            self.data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
            )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                         DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir,
                                                             DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                             TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,
                                                            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                            TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir,
                                                         DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                         PREPROCESSING_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.model_trainer_dir:str = os.path.join(self.training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path:str = os.path.join(self.model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.model_expected_accuracy:float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_uderfitting_threshold:float = MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD

class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.model_evaluation_dir_name:str = os.path.join(self.training_pipeline_config.artifact_dir,MODEL_EVALUATION_DIR_NAME)
        self.model_evaluation_file_path:str = os.path.join(self.model_evaluation_dir_name,MODEL_EVALUATION_REPORT_NAME)
        self.change_threshold :float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

class ModelPusherConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.model_evaluation_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path = os.path.join(self.model_evaluation_dir,MODEL_FILE_NAME)
        timestamp = round(datetime.now().timestamp())
        self.saved_model_path=os.path.join(
            SAVED_MODEL_DIR,
            f"{timestamp}",
            MODEL_FILE_NAME)