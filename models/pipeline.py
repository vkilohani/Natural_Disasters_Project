from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from models.custom_classifier import MyClassifier
from models.pipeline_utils import trivial, len_words, tokenize    
from models.clean_utils import custom_cleanup

def build_model(output_categories, drop_cols):
    """
    Function to build a model pipeline ending up in a gridsearch object.
    
        Args:
        ----
            No args.
        Returns:
        --------
            cv: sklearn.model_selection.GridSearchCV 
                gridsearch pipeline object
    """
    
    estimators = [('clf1', LogisticRegression(class_weight='balanced',
                                       max_iter=10000)),
              ('clf2', LinearSVC(class_weight='balanced', max_iter=10000)),
              #('clf3', AdaBoostClassifier())
              ]
    
    subestimator = MultiOutputClassifier(VotingClassifier(
                             estimators=estimators), n_jobs = 1)

    clf = MyClassifier(output_categories, drop_cols, subestimator)
    
    col_tfr = ColumnTransformer(
                                [("vect", 
                                  TfidfVectorizer(tokenizer=tokenize), 0)], 
                                remainder='passthrough'
                                )

    my_pipeline = Pipeline([
                #('cleanup', FunctionTransformer(custom_cleanup)),
                ('f_union', 
                    FeatureUnion([
                        ('trivial', 
                     FunctionTransformer(trivial)),
                        ('wc_norm', 
                            Pipeline([
                            ('wc', FunctionTransformer(len_words)),
                            ('scaler', MinMaxScaler())
                            ])
                        )
                    ])
                ),
                ('col_tfr', col_tfr),
                ('clf', clf)
            ])
      
    parameters = {
        'col_tfr__vect__ngram_range': [(1, 1)],
        #'clf__subestimator__estimator__clf2__C': [0.1, 1.0, 10.0],
    }
    
    cv = GridSearchCV(
                    estimator = my_pipeline, 
                    param_grid=parameters,
                    scoring = 'f1_samples',
                    cv=4,
                    refit=True,
                    n_jobs=-1
                    )
    
    return cv