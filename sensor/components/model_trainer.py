from sensor.entity.config_entity import ModelTrainerConfig
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_numpy_array
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.utils import save_object,load_object
from sensor.ml.model.estimator import SensorModel
import sys



class ModelTrainerComponent:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys) 


    def train_model(self,x_train,y_train):
        try:
            xgb_classifier = XGBClassifier()
            xgb_classifier.fit(x_train,y_train)
            return xgb_classifier

        except Exception as e:
            raise SensorException(e, sys) 

    def initiate_model_trainer(self):
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            logging.info(f"train_array = {train_arr[0]}")
            logging.info(f"test_arr = {test_arr[0]}")


            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logging.info(f"X_train = {X_train[0]}")
            logging.info(f"y_train = {y_train[0]}")
            logging.info(f"X_test = {X_test[0]}")
            logging.info(f"y_test = {y_test[0]}")

            model = self.train_model(X_train, y_train)

            y_train_pred = model.predict(X_train)
            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)
            if classification_train_metric.f1_score <= self.model_trainer_config.model_expected_accuracy:
                raise Exception("Trained Model is not good to provide expected accuracy")

            y_test_pred = model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            #check overfitting and underfitting
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)

            if diff> self.model_trainer_config.overfitting_uderfitting_threshold:
                raise Exception("Model is not good try to do some more experimentation")
            
            #create Sensor Model and save the model to pickl file
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=sensor_model)

            #return model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          train_metric_artifact=classification_train_metric, test_metric_arifatct=classification_test_metric)
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e,sys)