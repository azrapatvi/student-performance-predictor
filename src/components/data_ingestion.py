# to bringing the data , means this will contain all the code for reading the data
import os 
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#decorator
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig() # the three variables get stored in this self.ingestionconfig (train_data_path,test_data_path,raw_data_path)

    def get_data(self):
        print("entered into data ingestion")
        try:
            df=pd.read_csv("notebook\data\stud.csv") # as for now our data is at this path notebook\data\stud.csv we used read_csv later on we can specify the code for getting the data from different locations also like mongodb ,mysql and all
            print("read the dataset as datafram")

            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok=True) #os.path.dirname return the dir name of sself.ingestionconfig.train_data_path and it (self.ingestionconfig.train_data_path) has path "artifacts\train.csv" so indirectly this will give artifacts 

            df.to_csv(self.ingestionconfig.raw_data_path,index=False,header=True)# storing the fetched data in artifacts\data.csv

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)
            print(f"stored train data at :{self.ingestionconfig.train_data_path} and test data at :{self.ingestionconfig.test_data_path}")

            print("ingestion of data is completed")

            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.get_data()

    datatransformation=DataTransformation()
    datatransformation.initiate_data_transformation(train_data,test_data)
        


