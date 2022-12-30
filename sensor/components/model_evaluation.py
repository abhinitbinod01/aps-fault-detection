from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.entity.artifact_entity import ModelEvaluationArtifact,ModelTrainerArtifact,DataValidationArtifact
import os,sys
import pandas as pd
from sensor.utils import load_object,write_yaml_file
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.constants.training_pipeline import TARGET_COLUMN
import numpy as np


class ModelEvaluationComponent:
    def __init__(self,model_evaluation_config : ModelEvaluationConfig,model_trainer_artifact:ModelEvaluationArtifact,data_validation_artifact:DataValidationArtifact):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_validation_artifact = data_validation_artifact

    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path,index_col=0)
            test_df = pd.read_csv(valid_test_file_path,index_col=0)

            df = pd.concat([train_df,test_df],axis=0)
            df = df.replace('na',np.nan)
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path

            model_resolver = ModelResolver()
            is_model_accepted = True
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted = True,
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=trained_model_file_path, 
                    train_model_metric_artifact = self.model_trainer_artifact.test_metric_arifatct, 
                    best_model_metric_artifact = None)
                return model_evaluation_artifact

            latest_model_file_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_file_path)
            train_model = load_object(file_path = trained_model_file_path)
            y_true = df[TARGET_COLUMN]
            y_true = y_true.replace(TargetValueMapping().to_dict())

            df = df.drop(TARGET_COLUMN,axis=1)            

            y_trained_predict = train_model.predict(df)
            y_latest_predict = latest_model.predict(df)

            trained_model_metric = get_classification_score(y_true, y_pred=y_trained_predict)
            latest_model_metric = get_classification_score(y_true, y_pred=y_latest_predict)

            improved_accuracy = trained_model_metric.f1_score - latest_model_metric.f1_score

            if improved_accuracy > self.model_evaluation_config.change_threshold:
                is_model_accepted = True
            else:
                is_model_accepted = False

            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted = True,
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_file_path, 
                    trained_model_path=trained_model_file_path, 
                    train_model_metric_artifact = trained_model_metric, 
                    best_model_metric_artifact = latest_model_metric)
            model_eval_report = model_evaluation_artifact.__dict__

            write_yaml_file(file_path=self.model_evaluation_config.model_evaluation_file_path, content=model_eval_report)

            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)

