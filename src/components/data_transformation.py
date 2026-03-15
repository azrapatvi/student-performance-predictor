# the aim of this file is to create featuers , transform fetures , data cleaning ,data transformation

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer # columntransformer is basically used to create pipeline
from sklearn.pipeline import Pipeline # thsi is to define steps to perfom in sequence
from sklearn.impute import SimpleImputer # this is to fill the missing values in data with mean median, most_frequent values

from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str =os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transfomerobj(self):
        try:
            categorical_cols=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_cols=['reading_score', 'writing_score']

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoding", OneHotEncoder())
                ]
            )
            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("StandardScaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_cols),
                    ("cat_pipeline",cat_pipeline,categorical_cols)
                ]
                
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            print("read train data and test data into dataframe")

            preprocessing_obj=self.get_data_transfomerobj() # get_data_transformerobj returns preprocessor object
            target_col='math_score'
        
            X_train=train_df.drop(target_col,axis=1)
            y_train=train_df[target_col]

            X_test=test_df.drop(target_col,axis=1)
            y_test=test_df[target_col]

            X_train_transformed=preprocessing_obj.fit_transform(X_train)
            X_test_transformed=preprocessing_obj.transform(X_test)

            #np.c_ joins the arrays column wise
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr  = np.c_[X_test_transformed, np.array(y_test)]

            save_object(preprocessing_obj,self.data_transformation_config.preprocessor_obj_file_path) # save_object funciton  is in utils.py which can now take any object and create .pkl file in artifacts
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            raise CustomException(e,sys)

