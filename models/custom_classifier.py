import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 output_categories=None, 
                 drop_cols=None, 
                 subestimator = None):
         
        self.subestimator = subestimator
        self.drop_cols = drop_cols
        self.output_categories = output_categories
        self.keep_cols = [x for x in output_categories if x not in drop_cols]
        
    def fit(self, X, y=None):
        y_red = y[self.keep_cols]
        self.subestimator.fit(X, y_red)
        
    def predict(self, X, y=None):
        y_pred_red = self.subestimator.predict(X)
        df = pd.DataFrame(y_pred_red, index=np.arange(X.shape[0]), columns = self.keep_cols)
        zero_df = pd.DataFrame(0, index=np.arange(X.shape[0]), columns=self.drop_cols)
        full_pred = pd.concat([df, zero_df], axis=1)
        full_pred = full_pred[self.output_categories]
        return full_pred