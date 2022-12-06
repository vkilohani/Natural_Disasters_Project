import sys, os, time, joblib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path_utils import full_data_path, full_models_path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import string
from models.pipeline import build_model

import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])


def load_data(database_filepath):
    """Loads data from database_filepath.
    
        Args:
        ----- 
            database_filepath: str
                name of disaster database
        
        Returns:
        --------
            (X, y): a 2-tuple of dataframes
                feature, labels from the disaster database
    """
    engine = create_engine('sqlite:///'
                           + full_data_path(database_filepath)
                           )
    df = pd.read_sql(full_data_path(database_filepath), engine)
    
    #Convert all column names to string so we can pass to the 
    #ColumnTransformer() that requires string column names
    df.columns = [str(x) for x in df.columns.values]    
    
    X_df = df["message"]
    y_df = df.drop(["id", "message", "original", "genre"], axis=1)
    
    return X_df, y_df
    


def create_categories(y_df):
    """Gets a tuple consisting of a list of all categories and a list of 
    categories with the missing classes from the labels dataset.
    
        Args:
        ----- 
            y_df: dataframe
                labels dataframe
        Returns:
        --------
            (output_categories, drop_cols): (list, list)
                (all categories, labels with missing classes)
    """
    unique_vals = y_df.nunique()
    output_categories = y_df.columns.values
    drop_cols = unique_vals[unique_vals==1].index.values.tolist()
    
    return output_categories, drop_cols
    
    

def evaluate_model(model, X_df, y_df):
    """
    Evaluates and pints the evaluation of the model on a test set.
    
        Args:
        -----
            model: sklearn.model_selection.GridSearchCV
                trained model
            X_df: pd.DataFrame
                feature dataframe
            y_df: pd.DataFrame
                labels dataframe
        Returns:
        --------
        No return value        
    """
    
    y_pred = model.predict(X_df).values
    y_actual = y_df.values
    output_categories = y_df.columns.values
    
    for i in range(y_actual.shape[1]):
        cf_report = classification_report(y_actual[:,i] , y_pred[:,i])
        metrics_col = cf_report
        print(output_categories[i], ":\n")
        print(cf_report.split('\n')[3].split()[1:4])


def save_model(model, filename):
    """Exports the model to a classifier pickle file with a specified name.
    
        Args:
        -----
            model: sklearn.model_selection.GridSearchCV
                trained model
            filename: str
                name of the pickle file
                
        Returns:
        --------
            No return value        

    """
    joblib.dump(model, full_models_path(filename))



def main():
    """Loads the dataset. Builds the model. Trains and evaluates the 
    model, and then saves it."
    
        Args:
        -----
            No args
        
        Returns:
        --------
            No return value           
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_df, y_df = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X_df, 
                                                            y_df,
                                                            test_size=0.2,
                                                            random_state=42)
        
        output_categories, drop_cols = create_categories(y_df)
        
        print('Building model...')
        model = build_model(output_categories, drop_cols)
        
        start = time.time()
        print('Training model...')
        model.fit(X_train, y_train)
        end = time.time()
        
        duration = end-start
        print("Time taken to train the model: {}m and {:.1f}s".format(
            duration//60, duration%60))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()