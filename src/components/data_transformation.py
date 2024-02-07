import sys 
import os 
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_transformer_object(self):
        try:
            num_features = ['writing score','reading score']
            cat_features = ['gender',
                            'lunch',
                            'parental level of education',
                            'race/ethnicity',
                            'test preparation course'
                            ]
            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Numerical columns standardization completed')
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohEncoder', OneHotEncoder()),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical columns encoding completed')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_features',num_pipeline,num_features),
                    ('cat_features',cat_pipeline,cat_features)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def init_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info('Reading train and test data completed')
            
            preprocessing = self.get_transformer_object()
            target_column = "math score"
            num_columns = ['writing score', 'reading score']
            
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.transformation_config.preprocessor_obj_file,
                obj=preprocessing

            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
