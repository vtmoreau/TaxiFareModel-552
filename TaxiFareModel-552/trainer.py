# imports
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from data import get_data, clean_data, split_data
from encoders import TimeFeaturesEncoder, DistanceTransformer
from utils import sk_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
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
            ('model', Ridge())
        ])
        
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
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
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    score = trainer.evaluate(X_test, y_test)
    
    print(f"Score is {score}")
