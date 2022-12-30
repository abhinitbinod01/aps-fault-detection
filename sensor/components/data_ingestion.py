from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.exception import SensorException
import os
import sys
from sensor.entity.config_entity import TrainingPipelineConfig
from sensor.utils import get_collection_as_dataframe
import pandas as pd
import yaml
from sensor.constants.training_pipeline import *
from sklearn.model_selection import train_test_split
from sensor.logger import logging
from sensor.utils import load_yaml_file


class DataIngestionComponent:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    @staticmethod
    def get_data_from_db(database_name, collection_name) -> pd.DataFrame:
        try:
            logging.info("Extracting data from DB")
            return get_collection_as_dataframe(database_name, collection_name)
        except Exception as e:
            raise SensorException(e, sys)

    def export_data_to_feature_store(self, dataframe: pd.DataFrame):
        try:
            logging.info("Exporting dataframe to Feature Store Dir")
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            dataframe.to_csv(
                self.data_ingestion_config.feature_store_file_path)
            logging.info("Exported dataframe to Feature Store Dir Successfully")
        except Exception as e:
            raise SensorException(e, sys)

    def drop_columns_from_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Dropping Columns from df")
            logging.info(f"columns before dropping: {dataframe.shape}")
            schema = load_yaml_file(SCHEMA_FILE_PATH)
            drop_columns = list(schema["drop_columns"])
            dataframe.drop(drop_columns,inplace=True,axis=1)
            logging.info(f"columns after dropping: {dataframe.shape}")
            logging.info("Dropped Columns from df successfully")
            return dataframe
        except Exception as e:
            raise SensorException(e, sys)

    def train_test_split(self, df: pd.DataFrame):
        try:
            logging.info("Splitting train test data")
            X = df.drop(TARGET_COLUMN,axis=1)
            y = df[TARGET_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            train_dir = os.path.dirname(
                self.data_ingestion_config.training_file_path)
            os.makedirs(train_dir, exist_ok=True)
            df.to_csv(self.data_ingestion_config.training_file_path)
            test_dir = os.path.dirname(
                self.data_ingestion_config.testing_file_path)
            os.makedirs(test_dir, exist_ok=True)
            df.to_csv(self.data_ingestion_config.testing_file_path)
            logging.info("Splitted train test data and exported successfully to ingested data dir")
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(f"{'='*60} Data Ingestion Started {'='*60}")
            df = DataIngestionComponent.get_data_from_db(
                database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)
            print(df.shape)
            self.export_data_to_feature_store(df)
            df_new = self.drop_columns_from_df(df)
            self.train_test_split(df_new)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info("data_ingestion_artifact=    {}".format(data_ingestion_artifact))
            logging.info(f"{'='*60} Data Ingestion Done Successfully {'='*60}")

            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)
