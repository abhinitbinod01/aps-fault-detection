import pymongo
import os,sys
from sensor.exception import SensorException
from sensor.constants.database import DATABASE_NAME,MONGO_DB_URL
class MongoDbClient:
    client  = None

    def __init__(self,database_name:str=DATABASE_NAME):
        try:
            if MongoDbClient.client == None:
                connection_url = os.getenv(MONGO_DB_URL)
                MongoDbClient.client = pymongo.MongoClient(connection_url)
            self.client = MongoDbClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise SensorException(e, sys)


