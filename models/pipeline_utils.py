import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
"""
Uncomment the following line if the packages are absent
"""
#nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from nltk.stem import WordNetLemmatizer

def len_words_text(text):
   return len(tokenize(text))

def len_words(text_collection):
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
    return np.array(text_series).reshape(-1, 1)

def tokenize(text):
    """
    Tokenizes the input string
    args:
    - string
    returns:
    - tokenized string
    """
    tokens = word_tokenize(text)
    clean_tokens = []
    for token in tokens:
        clean_token = token.lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens
