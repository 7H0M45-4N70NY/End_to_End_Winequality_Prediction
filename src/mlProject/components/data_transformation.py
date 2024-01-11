import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from imblearn.over_sampling import SMOTE
import numpy as np


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        target_column="quality"
        smote=SMOTE(random_state=1)
        X = train.drop(columns=target_column,axis=1)
        y=train[[target_column]]
        smote_X,smote_y=smote.fit_resample(X,y)
        train_new=pd.concat([smote_X,smote_y],axis=1)
        logger.info("oversampling to balance classes")
        logger.info("Oversampled train data shape")
        logger.info(train_new.shape)
        train_new.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        

        print(train.shape)
        print(test.shape)
        
