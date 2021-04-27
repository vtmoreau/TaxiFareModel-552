# Python Imports
import numpy as np
import pandas as pd
import os
from termcolor import cprint
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
# MlFlow Imports
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
# Package Imports
from TaxiFareModel.data import get_data, clean_data, split_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import sk_rmse, sk_neg_rmse

mlf_uri = os.getenv("ML_FLOW_URI")

class Trainer():
    def __init__(self,  X, y, local=True):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.local = local
        self.experiment_name =f"taxifare-vtmoreau-552"

    @memoized_property
    def mlflow_client(self):
        if self.local:
            return MlflowClient()
        mlflow.set_tracking_uri(mlf_uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def set_pipeline(self, estimator):
        """defines the pipeline as a class attribute"""
        # Features: Distance
        feat_distance = ['pickup_latitude',
                        'pickup_longitude',
                        'dropoff_latitude',
                        'dropoff_longitude']

        pipe_distance = Pipeline([
            ('to_distance', DistanceTransformer()),
            ('std_scale', StandardScaler())
        ])
        # Features: Time
        feat_time = ['pickup_datetime']
        pipe_time = Pipeline([
            ('to_time_feat', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe_encode', OneHotEncoder(handle_unknown="ignore"))
        ])

        # Preprocessing
        pipe_cols = ColumnTransformer([
            ('pipe_distance', pipe_distance, feat_distance),
            ('pipe_time', pipe_time, feat_time)
        ])
        pipe_preproc = Pipeline([('preproc', pipe_cols)]) 
        
        # Model
        self.pipeline = Pipeline([
            ('preproc', pipe_preproc),
            ('model', estimator)
        ])
        
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)       
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
            # predict on test
        y_pred = self.pipeline.predict(X_test)
        
        score = cross_val_score(self.pipeline,
                                X_test, y_test,
                                scoring=sk_rmse,
                                n_jobs=-1)
        
        self.mlflow_log_param('estimator', str(self.pipeline.get_params()['model'])
                              .strip('()'))
        self.mlflow_log_metric('rmse', score.mean())
        
        return score.mean()
        

if __name__ == "__main__":
    # get data
    data = get_data()
    # clean data
    clean_data = clean_data(data)
    # set X and y
    # hold out
    X_train, X_test, y_train, y_test = split_data(clean_data)
    # train
    for model in [Ridge(), Lasso(), SGDRegressor()]:
    
        trainer = Trainer(X_train, y_train)
        trainer.set_pipeline(model)
        trainer.run()
        # evaluate
        score = trainer.evaluate(X_test, y_test)
    
        cprint(f"Estimator: {model}", "blue")
        cprint(f"RMSE:{round(score, 3)}", "green")
        cprint("-----------------------------------------------------")
