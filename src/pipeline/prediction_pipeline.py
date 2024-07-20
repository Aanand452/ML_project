# create prediction pipeline
# create function for load a object
# create custom class based upon our dataset
# create function convert data into dataframe with help of dict


import os
import sys 
from src.logger import logging 
from src.exception import CustomException 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass 
from src.utils import load_object

class PreductionPipelineConfig:
    def __init__(self):
        pass 
    
    def predict(self,features):
        preprocessor_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
        model_path = os.path.join("artifacts/model_trainer","model.pkl")
        
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        
        scaled_feature = preprocessor.transform(features)
        
        pred = model.predict(scaled_feature)
        return pred
    
class Customclass:
    def __init__(self,
                 age:int, 
                 capital_gain:int, 
                 capital_loss:int, 
                 hours_per_week: int,
                 workclass:int, 
                 education_num:int,
                 marital_status:int, 
                 occupation:int,
                 relationship:int,
                 race:int, 
                 sex:int, 
                 native_country:int):
        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.native_country = native_country
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        
    def get_data_as_dataframe(self):
        try:
            customclass_input = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education_num": [self.education_num],
                "marital_status": [self.marital_status],
                "occupation":[self.occupation],
                "relationship":[self.relationship],
                "race":[self.race],
                "sex":[self.sex],
                "native_country":[self.native_country],
                "capital_gain":[self.capital_gain],
                "capital_loss":[self.capital_loss],
                "hours_per_week":[self.hours_per_week]
            }
            data = pd.DataFrame(customclass_input)
            return data
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
        
        
        
