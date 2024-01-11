import os
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from mlProject.utils.common import save_json
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvalConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self,config:ModelEvalConfig):
        self.config=config
    def eval_metrics(self,actual,pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred,average='weighted')
   
        return accuracy,f1
        
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        (accuracy,f1) = self.eval_metrics(test_y, predicted_qualities)   
        # Saving metrics as local
        scores = {"Accuracy":accuracy,"f1_score":f1}
        save_json(path=Path(self.config.metric_file_name), data=scores)

