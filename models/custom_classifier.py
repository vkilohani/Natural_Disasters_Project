import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class MyClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom estimator class for multioutput binary classifiers
    that skips training on labels where there are no positive samples.
    The output on such labels will be the default label (assumed to be 0).
    
    Attributes
    ----------
    output_categories: list
        list of all output categories
    drop_cols: list
        list of categories which the classifier will not fit to
    subestimator: sklearn estimator 
    
    Methods
    -------
    fit(X, y=None):
        Performs the fit on all labels except those in self.drop_cols.
        
    predict(X, y=None):
        Predicts the results of the fitted classifier. Prediction on 
        self.drop_cols will be set to the default value (assumed to be 0).
    """
    
    def __init__(self, 
                 output_categories=None, 
                 drop_cols=None, 
                 subestimator = None):
        """
        Constructs the class object.
        
            Args:
            -----
                output_categories: default = None, list
                    list of all output categories
                drop_cols: default = None, list
                    list of categories which the classifier will not fit to
                subestimator: default = None, sklearn estimator 
                    sklearn estimator to perform multioutput classification
        """
         
        self.subestimator = subestimator
        self.drop_cols = drop_cols
        self.output_categories = output_categories
        self.keep_cols = [x for x in output_categories if x not in drop_cols]
        
    def fit(self, X, y=None):
        """Performs the fit on all labels except those in self.drop_cols.
        """
        
        y_red = y[self.keep_cols]
        self.subestimator.fit(X, y_red)
        
    def predict(self, X, y=None):
        """Predicts the results of the fitted classifier. Prediction on 
        self.drop_cols will be set to the default value (assumed to be 0).
        """
        
        y_pred_red = self.subestimator.predict(X)
        df = pd.DataFrame(y_pred_red, index=np.arange(X.shape[0]), columns = self.keep_cols)
        zero_df = pd.DataFrame(0, index=np.arange(X.shape[0]), columns=self.drop_cols)
        full_pred = pd.concat([df, zero_df], axis=1)
        full_pred = full_pred[self.output_categories]
        return full_pred