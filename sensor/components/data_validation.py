from sensor.entity.config_entity import DataValidationConfig
from sensor.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from sensor.exception import SensorException
import os,sys
import pandas as pd
from sensor.utils import load_yaml_file,write_yaml_file
from sensor.constants.training_pipeline import SCHEMA_FILE_PATH
from sensor.logger import logging
from scipy.stats import ks_2samp


class DataValidationComponent:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self._schema_config = load_yaml_file(SCHEMA_FILE_PATH)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path,index_col=0)
        except Exception as e:
            raise SensorException(e, sys)

    def validate_number_of_columns(self,dataframe)->bool:
        try:
            column_list = list(self._schema_config['columns'])
            if len(dataframe.columns) == len(column_list):
                return True
            return False
        except Exception as e:
            raise SensorException(e, sys)
    
    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_column_list = list(self._schema_config['numerical_columns'])
            numerical_colum_present = True
            missing_numerical_columns = []
            for num_col in numerical_column_list:
                if num_col not in dataframe.columns:
                    numerical_colum_present = False
                    missing_numerical_columns.append(num_col)
            logging.info(f"Missing Numerical Columns: {missing_numerical_columns}")  
            return numerical_colum_present

        except Exception as e:
            raise SensorException(e, sys)

    def detect_dataset_drift(self,base_df,current_df,threshold:float=0.5)->bool:
        try:
            report = {}
            status = True
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({
                    column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status":is_found
                    }                    
                })
                drift_report_file_path = self.data_validation_config.drift_report_file_path
                drift_dir = os.path.dirname(drift_report_file_path)
                os.makedirs(drift_dir,exist_ok=True)
                if is_found:
                    write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
                else:
                    write_yaml_file(file_path=drift_report_file_path, content=report, replace=False)
                return status
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            error_message=""
            train_df = DataValidationComponent.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidationComponent.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            #validate number of columns
            status = self.validate_number_of_columns(dataframe = train_df)
            if not status:
                error_message = f"{error_message} Columns are missing in train dataframe"
            status = self.validate_number_of_columns(dataframe = test_df)
            if not status:
                error_message = f"{error_message} Columns are missing in test dataframe"

            #validate numerical columns
            status = self.is_numerical_column_exist(dataframe=train_df)
            if not status:
                error_message = f"{error_message} Numerical Columns are missing in train dataframe"
            status = self.is_numerical_column_exist(dataframe=test_df)
            if not status:
                error_message = f"{error_message} Numerical Columns are missing in test dataframe"
            
            if len(error_message) > 0:
                raise Exception(error_message)
            
            #check data drift
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            data_validation_artifact = DataValidationArtifact(validation_status=status,
                                                              valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                                                              valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                                                              invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                                                              invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                                                              drift_report_file_path=self.data_validation_config.drift_report_file_path)
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)
        