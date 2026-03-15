import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(obj,path):
    try:
        dir_name=os.path.dirname(path)

        os.makedirs(dir_name,exist_ok=True)

        with open(path,'wb')as f:
            pickle.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params, n_iter=20):
    try:
        report = {}
        best_models = {}  # to store tuned models

        for name, model in models.items():
            print(f"Training and tuning {name}...")

            if name in params:
                rs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params[name],
                    n_iter=n_iter,
                    cv=5,
                    n_jobs=-1,
                    scoring='r2',
                    random_state=42
                )
                rs.fit(X_train, y_train)
                best_model = rs.best_estimator_
                print(f"Best params for {name}: {rs.best_params_}")

                # Already trained on training folds; refit on full training data to be sure
                best_model.fit(X_train, y_train)
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            best_models[name] = best_model  # store tuned model

        return {"scores": report, "best_models": best_models}

    except Exception as e:
        raise CustomException(e, sys)