import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data.")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            #Define models and hyperparameter tuning params
            models = {
                "Decision Tree": (DecisionTreeRegressor(),{
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10]
                }),
                "Random Forest": (RandomForestRegressor(),{
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10]
                }),
                "Gradient Boosting": (GradientBoostingRegressor(),{
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 10],
                    'learning_rate': [0.1, 0.05, 0.01]
                }),
                "Linear Regression": (LinearRegression(),{
                    #does not have any hyperparameters to tune.
                }),
                "XGBRegressor": (XGBRegressor(),{
                    'max_depth': [3, 5, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200, 300]
                }),
                "CatBoosting Regressor": (CatBoostRegressor(verbose=False),{
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8]
                }),
                "AdaBoost Regressor":(AdaBoostRegressor(),{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.05, 0.01]
                })
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,
                                          X_test=X_test,y_test=y_test,models=models,param=models.items())
            
            ## To get the best model score from dict

            best_model_score = max(sorted(model_report.values()))
            print(f"Best model score :{best_model_score}")

            ## To get the best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            idx_best_model = list(model_report.keys()).index(best_model_name)
            print(f"Best Model Name - {best_model_name}")

            best_model = models[best_model_name]
            print(f"Best Model - {best_model} at {idx_best_model}")


            if best_model_score < 0.6:
                raise CustomException("No Best Model Found!")
            
            logging.info("Best Model found on both training and test dataset")
            
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                
            )

            predicted = best_model[0].predict(X_test)
            r2_square = r2_score(y_test,predicted)

           
            return r2_square
      

        except Exception as e:
            raise CustomException(e,sys)