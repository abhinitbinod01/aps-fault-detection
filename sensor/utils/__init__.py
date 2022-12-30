import pandas as pd
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.configuration.mongo_db_connection import MongoDbClient
import os,sys
from sensor.constants.database import DATABASE_NAME
import yaml
import dill
import numpy as np


def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    try:
        logging.info(f"Reading data from database : {database_name} and collection_name :{collection_name}")
        mongo_client = MongoDbClient(DATABASE_NAME)
        print(mongo_client)
        df = pd.DataFrame(list(mongo_client.client[database_name][collection_name].find()))
        logging.info(f"dataframe shape before dropping _id is {df.shape}")
        logging.info(f"dataframe size is {df.shape[0]}")
        if "_id" in df.columns:
            df.drop("_id",axis=1,inplace=True)
        logging.info(f"dataframe shape after dropping _id is {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e, sys)

def load_yaml_file(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(e, sys)

def write_yaml_file(file_path:str,content:object,replace:bool):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise SensorException(e, sys)

def save_numpy_array(file_path:str,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_object:
            np.save(file_object,array)
    except Exception as e:
        raise SensorException(e, sys)


def load_numpy_array(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,'rb') as file_object:
            return np.load(file_object)
    except Exception as e:
        raise SensorException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise SensorException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SensorException(e, sys) from e