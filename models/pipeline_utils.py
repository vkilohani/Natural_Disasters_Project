import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
"""
Uncomment the following two lines if the packages are absent
"""
#import nltk
#nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from nltk.stem import WordNetLemmatizer

def len_words_text(text):
    "Returns the number of words in the input string."
    return len(tokenize(text))

def len_words(text_collection):
    """Returns an array, reshaped as a two dimensional array, containing number of words in the entries of a list, numpy array or pandas series.
    """
    if isinstance(text_collection, list):
        len_list = [len_words_text(x) for x in text_collection]
        return np.array(len_list).reshape(-1, 1)
    elif isinstance(text_collection, pd.Series):
        len_series = text_collection.apply(len_words_text).rename("len")
        return np.array(len_series).reshape(-1, 1)
    elif isinstance(text_collection, np.ndarray):
        len_array = len_words_text(text_collection)
        return len_array.reshape(-1, 1)
    
def trivial(text_series):
    """Returns text_series (pandas.Series) as a reshaped two dimensional 
    array.
    """
    return np.array(text_series).reshape(-1, 1)

def tokenize(text):
    """
    Tokenizes the input string into words and converts it to lowercase.
    
        Args:
        -----
            text: str
                Text to be tokenized
                
        Returns:
        --------
            clean_tokens: list 
                tokenized string
    """
    tokens = word_tokenize(text)
    clean_tokens = []
    for token in tokens:
        clean_token = token.lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens
