# from dataclasses import dataclass
# import os
# import sys
# import pandas as pd
# import numpy as np
# from src.logger import logging
# from src.exception import CustomException
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from src.utils import save_object
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from src.utils import evaluate_model


# class ModelTrainerConfig:
#     train_model_file_path = os.path.join("artifacts/model_trainer","model.pkl")
    
    
# class modelTrainer:
#     def __init__(self):
#         self.model_Trainer_config = ModelTrainerConfig()
    
#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             X_train,y_train,X_test,y_test = (
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
            
#             models = {
                
#                 "Random Forest":RandomForestClassifier(),
#                 "Decision Tree":DecisionTreeClassifier(),
#                 "Logistic Regression":LogisticRegression()
#             }
            
#             params = {
#                 "RandomForest":{  
#                     "class_weight":["balanced"],
#                     "n_estimators":[20,50,30],
#                     "max_depth":[10,8,5],
#                     "min_samples_split":[2,5,10],
#                     },
                
#                 "DecisionTree":{
#                     "class_weight":["balanced"],
#                     "criterion":["gini","entropy","log_loss"],
#                     "splitter":["best","random"],
#                     "max_depth":[3,4,5,6],
#                     "min_samples_split":[2,3,4,5],
#                     "min_samples_leaf":[1,2,3],
#                     "max_features":["auto","sqrt","log2"]
#                     },
                
                
#                 "Logistic":{
#                     "class_weight":["balanced"],
#                     "penalty":["l1","l2"],
#                     "C":[0.001,0.01,0.1,1,10,100],
#                     "solver":["liblinear", "saga"]
#                 }
#             }
                    
#             model_report:dict = evaluate_model(X_train = X_train,y_train = y_train,X_test = X_test,y_test = y_test,models = models,params = params)
            
#             best_model_score = max(sorted(model_report.values()))
            
            
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
            
#             best_model = models[best_model_name]
            
#             logging.info("Best model found, Model Name is {}".format(best_model_name),"Best model score is {}.format(best_model_score))")
            
#             save_object(file_path = self.model_Trainer_config.train_model_file_path,obj = best_model)
            
#             return best_model            
#         except Exception as e:
#             raise CustomException(e,sys)
            
            
from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.utils import evaluate_model


class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_Trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression()
            }

            params = {
                "Random Forest": {  
                    "class_weight": ["balanced"],
                    "n_estimators": [20, 50, 30],
                    "max_depth": [10, 8, 5],
                    "min_samples_split": [2, 5, 10],
                },
                
                "Decision Tree": {
                    "class_weight": ["balanced"],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "splitter": ["best", "random"],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_split": [2, 3, 4, 5],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": [None,"sqrt", "log2",10,20]
                },
                
                "Logistic Regression": {
                    "class_weight": ["balanced"],
                    "penalty": ["l1", "l2"],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "saga"],
                    "max_iter":[100,200,500]
                }
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            logging.info(f"Best model found, Model Name is {best_model_name}, Best model score is {best_model_score}")
            
            save_object(file_path=self.model_Trainer_config.train_model_file_path, obj=best_model)
            
            return best_model
        except Exception as e:
            raise CustomException(e, sys)
