import os
import pickle
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)        

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    except:
        pass

                
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        
        report = {}

       # for i in range (len(list(models))):
        for model_name, (model, params) in models.items():
            #model = list(models.values())[i]
            #hypere parameter tuning for each model in the list.
            #para = param[list(models.keys())[i]]
            print(f"Hyper tuning the model :{model_name}")
            

            gs = GridSearchCV(model,params,cv=3)
            gs.fit(X_train,y_train)
            best_params = gs.best_params_
            best_score = gs.best_score_
            print(f"Best parameters for {model_name}: {best_params}")
            print(f"Best score for {model_name}: {best_score}\n")

             # Train the model with best hyperparameters
            best_model = model.set_params(**best_params)
            best_model.fit(X_train, y_train)

           # model.set_params(**gs.best_params_)
             #Train the model
            #model.fit(X_train,y_train)

            #Evaluate the model
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            print(f"Test model score for {model_name}: {test_model_score}")

            report[model_name] = test_model_score

            #report[list(models.keys())[i]] = test_model_score


        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)