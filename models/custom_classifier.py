
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import CountVectorizer, \
TfidfTransformer, TfidfVectorizer

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score

import nltk
from nltk.tokenize import word_tokenize
#nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from nltk.stem import WordNetLemmatizer

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, output_categories, drop_cols, estimators):
         
        self.estimators=estimators
        self.my_clf = MultiOutputClassifier(VotingClassifier(
                             estimators=self.estimators), n_jobs = 1)
        self.drop_cols = drop_cols
        self.output_categories = output_categories
        self.keep_cols = [x for x in output_categories if x not in drop_cols]
        
    def fit(self, X, y=None):
        y_red = y[self.keep_cols]
        self.my_clf.fit(X, y_red)
        
    def predict(self, X, y=None):
        y_pred_red = self.my_clf.predict(X)
        df = pd.DataFrame(y_pred_red, index=np.arange(X.shape[0]), columns = self.keep_cols)
        zero_df = pd.DataFrame(0, index=np.arange(X.shape[0]), columns=self.drop_cols)
        full_pred = pd.concat([df, zero_df], axis=1)
        full_pred = full_pred[self.output_categories]
        return full_pred 

def tokenize(text):
    """
    Tokenizes the input string
    args:
    - string
    returns:
    - tokenized string
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model(output_categories, drop_cols):
    """Function to build a model pipeline ending up in a gridsearch object
    args: None
    returns:
    - grid search pipeline object
    """

    f1_scorer = make_scorer(fbeta_score, beta=1, greater_is_better=True, labels = ['1'], average = 'samples', zero_division = 0)

    
    estimators = [('clf1', LogisticRegression(class_weight='balanced',
                                       max_iter=10000)),
              ('clf2', LinearSVC(class_weight='balanced')),
              ('clf3', AdaBoostClassifier())
              ]

    clf = MyClassifier(output_categories, drop_cols, estimators)
    
    my_pipeline = Pipeline([
                        ('vect', TfidfVectorizer(tokenizer = tokenize)),    
                        ('clf', clf)
    ])
    
    
    
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        #'clf__estimator__clf3__n_estimators': [50, 100, 200],
        #'clf__estimator__estimators__clf3__min_samples_split': [2, 3, 4],
        #'clf__estimator__estimators__clf1__C': [0.1, 1.0]
    }
    
    

    cv = GridSearchCV(
                    estimator = my_pipeline, 
                    param_grid=parameters,
                    scoring = 'f1_samples', #f1_scorer 
                    cv=4,
                    refit=True,
                    n_jobs=-1
                    )
    
    return cv