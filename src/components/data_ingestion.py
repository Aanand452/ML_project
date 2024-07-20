import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import get_data_transformation
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion","train.csv")
    test_data_path = os.path.join("artifacts/data_ingestion","test.csv")
    raw_data_path = os.path.join("artifacts/data_ingestion","raw.csv")

# notebook\data\income_cleandata.csv
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info("data ingestion initiated")
            data = pd.read_csv(os.path.join("/Users/apple/Downloads/Ml_Project/Ml_Project/notebook/data","income_cleandata.csv"))
            logging.info("Entered the data ingestion method or component")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Train test split initiated")
            
            train_set,test_set = train_test_split(data,test_size=0.3,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error occured in the data ingestion stage")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = get_data_transformation()
    train_array,test_array, _=  data_transformation.initialize_data_transformation(train_data_path,test_data_path)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array,test_array))
    
    
        
    
