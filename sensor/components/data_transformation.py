from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import DataTransformationConfig
from sensor.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from sensor.utils import save_numpy_array,load_numpy_array,save_object,load_object
import os,sys
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sensor.constants.training_pipeline import *
from sensor.ml.model.estimator import TargetValueMapping
from imblearn.combine import SMOTETomek
import numpy as np



class DataTranformationComponent:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            df = pd.read_csv(file_path,index_col=0)
            logging.info(f"{df.dtypes}")
            return df
        except Exception as e:
            raise SensorException(e, sys)
    
    def get_data_transformer_object(self)->Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy='constant',fill_value=0)
            preprocessor = Pipeline(steps=[
                ('Imputer',simple_imputer),
                ('RobustScaler',robust_scaler),
            ])
            return preprocessor
        except Exception as e:
            raise SensorException(e,sys)


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_df = DataTranformationComponent.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTranformationComponent.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor_object = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_train_df = input_feature_train_df.replace('na',np.nan)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())

            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = input_feature_test_df.replace('na',np.nan)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())

            logging.info(f"{input_feature_train_df.head()}")

            preprocessor_object.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy = "not majority")
            input_feature_train_final,target_feature_train_final = smt.fit_resample(transformed_input_train_feature, target_feature_train_df)
            input_feature_test_final,target_feature_test_final = smt.fit_resample(transformed_input_test_feature, target_feature_test_df)

            train_arr = np.c_[input_feature_train_final,target_feature_train_final]
            test_arr = np.c_[input_feature_test_final,target_feature_test_final]

            save_numpy_array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, test_arr)

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            return DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise SensorException(e,sys)

    
