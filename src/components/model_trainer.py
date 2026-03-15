import pandas as pd
import numpy as np
import os
import sys

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from dataclasses import dataclass

from catboost import CatBoostRegressor

from src.utils import save_object,evaluate_model
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_path: str= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_transformer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "DecisionTree":DecisionTreeRegressor(random_state=42),
                "XGBRegressor":XGBRegressor(random_state=42),
                "CatBoost":CatBoostRegressor(random_state=42,verbose=False)
            }
            params = {
                "LinearRegression": {
                    "fit_intercept": [True, False]
                },
                "Ridge": {
                    "alpha": [0.01, 0.1, 1.0, 10, 100],
                    "solver": ['auto', 'svd', 'cholesky', 'sparse_cg', 'saga'],
                    "max_iter": [None, 1000, 5000, 10000]
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10],
                    "max_iter": [1000, 5000, 10000],
                    "selection": ['cyclic', 'random']
                },
                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ['uniform', 'distance'],
                    "metric": ['euclidean', 'manhattan', 'minkowski']
                },
                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0],
                    "loss": ['linear', 'square', 'exponential']
                },
                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['auto', 'sqrt', 'log2']
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "DecisionTree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['auto', 'sqrt', 'log2', None]
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.2, 0.5],
                    "reg_alpha": [0, 0.01, 0.1, 1],
                    "reg_lambda": [1, 1.5, 2]
                },
                "CatBoost": {
                    "iterations": [500, 1000, 2000],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "depth": [3, 5, 7, 10],
                    "l2_leaf_reg": [1, 3, 5, 7, 9],
                    "bagging_temperature": [0, 1, 2, 5],
                    "border_count": [32, 50, 100]
                }
            }

            model_results = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_results['scores'].values())
            best_model_name = max(model_results['scores'], key=model_results['scores'].get)
            best_model = model_results['best_models'][best_model_name]  # <-- now it's tuned!

            if best_model_score < 0.60:
                raise CustomException("No best model found")

            save_object(best_model, self.model_trainer_config.trained_model_path)

            return {"best_model_name": best_model_name, "best_model_score": best_model_score}

        except Exception as e:
            raise CustomException(e,sys)