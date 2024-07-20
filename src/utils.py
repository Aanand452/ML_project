from src.logger import logging
from src.exception import CustomException 
import os 
import sys 
import pickle
from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score,accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path_join(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        return CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for model_name, model in models.items():
            para = params[model_name]
            GS = GridSearchCV(model,para,cv=3,scoring="accuracy")
            GS.fit(X_train,y_train)
            
            model.set_params(**GS.best_params_)
            model.fit(X_train,y_train)
            
            y_pred = model.predict(X_test) 
            test_model_accuracy = accuracy_score(y_test,y_pred)
            
            report[model_name] = test_model_accuracy
            
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
