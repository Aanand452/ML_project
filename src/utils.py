from src.logger import logging
from src.exception import CustomException 
import os 
import sys 
import pickle

def save_object(file_path,obj):
    try:
        dir_path = os.path_join(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        return CustomException(e,sys)