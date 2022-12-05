from deep_translator import GoogleTranslator
import re
import spacy
import pandas as pd
import string

from utils.custom_utils import full_data_path

class Dict_Substitute(dict):
    """
    A class inhertied from dict to translate text using a custom dictionary.
    
    Attributes
    ----------
    dict: a dictionary
        Custom dictionary to translate text
    
    Methods
    -------
    regex_compiler():
        Returns a compiled regular expression based on the keys of the dictionary
        
    regex_translate(text):
        Translates text using the dictionary that was used to instantiate
        the class
    """
    def regex_compiler(self):
        """
        Returns a compiled regular expression based on the keys of the dictionary.
        """
        return re.compile(
                        r'\b'+r'\b|\b'.join(list(self.keys())) + r'\b',
                        re.IGNORECASE)
    
    def regex_translate(self, text):
        """
        Translates text using the dictionary that was used to instantiate
        the class.
        
            Args:
            ----
                text: str
                    text to translate
            
            Returns:
            -------
                    Translated text (str)
        """
        return re.sub(
            self.regex_compiler(), lambda match: self[match.group(0)], text)

def remove_repeated(text):
    """Reduces repated letters to a maximum of 2 in text (str)."""
    
    pattern = re.compile(r'(.)\1{2,}')
    return re.sub(pattern, lambda match: match.group(1)+match.group(1), text)

def remove_repeated_series(pd_series):
    """Reduces repated letters to a maximum of 2 in text entries of pd_series
    (pandas.Series).
    """
    
    return pd_series.apply(lambda x: remove_repeated(x))

def remove_singles(text):
    """Removes single letter words from text (str)."""
    
    pattern = re.compile(r'\b(\w)\b')
    return re.sub(pattern, '', text)

def remove_singles_series(pd_series):
    """Removes single letter words from text entries of pd_series (pandas.Series)."""
    
    return pd_series.apply(lambda x: remove_singles(x))

def remove_punct(text):
    """Removes punctuation from text (str).
    """
    
    #return text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r"[',.:;@#?!&#%-]+\ *"," ", text, flags=re.VERBOSE)
    return re.sub(r'\s+',' ', clean_text)

def remove_punct_series(pd_series):
    """Removes punctuation from text entries of pd_series (pandas.Series).
    """
    
    return pd_series.apply(remove_punct)

def remove_digits(text):
    """Removes digits from text (str).
    """
    
    pattern = re.compile(r'\b\d+\b')
    return re.sub(pattern,'', text)

def remove_digits_series(pd_series):
    """Removes digits from text entries of pd_series (pandas.Series).
    """
    pattern = re.compile(r'\b\d+\b')
    return pd_series.apply(remove_digits)

def replace_url(text):
    """Replaces url from text (str) by urlplaceholder.
    """
    
    pattern = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
    return re.sub(pattern, 'urlplaceholder', text)

def replace_url_series(pd_series):
    """Removes url from text entries in pd_series (pandas.Series) by urlplaceholder.
    """
    
    return pd_series.apply(replace_url)

def translate_to_english(text):
    """Translates text (str) to english."""
    
    translator = GoogleTranslator(source='auto', target='en')    
    return translator.translate(text)

def translate_to_english_series(pd_series):
    """Translates text entries in pd_series (pandas.Series) to english."""
    
    return pd_series.map(translate_to_english)

def translate_chat_abbv_series(pd_series):
    """Translates chat abbreviations in text entries (str) of a pandas series to formal english using data/chat_acronyms_list.csv.
    """
    
    chat_csv_file = full_data_path('chat_acronyms_list.csv')
    chat_df = pd.read_csv(chat_csv_file, delimiter=';')
    
    chat_dict = dict(zip(chat_df["slang"], chat_df["full form"]))
    #Special cases - adding by hand
    chat_dict["\'ve"] = ""
    chat_dict["S.O.S."] = "SOS"
    chat_dict = Dict_Substitute(chat_dict)
    
    return pd_series.map(chat_dict.regex_translate)
    

def spacy_lemmatize_series(pd_series):
    """Uses spacy to lemmatize text entries in pd_series (pandas.Series).
    """
    
    nlp = spacy.load("en_core_web_sm")
    
    def spacy_lemmatize(text):
        """Uses spacy to lemmatize text (str).
        """
        return ' '.join([token.lemma_ for token in nlp(text)])
    
    
    return pd_series.apply(spacy_lemmatize) 
        

def custom_cleanup(pd_series):
    """
    Performs custom cleanup of pd_series (pandas.Series).
        - URLSs will be replaced by urlplaceholder.
        - Repeated consecutive letters will be reduced to 2.
        - Spacy Lemmatization will be performed.
        - Single letter words will be removed.
        - Chat Abbreviations will be translated to formal english.
        - Punctuations will be removed.
        - Digits in the text will be removed.
    
        Args:
        ----
            pd_series: pandas.Series
                pandas series to be cleaned up
        
        Returns:
        --------
            clean_series: pandas.Series
                cleaned up pandas series
    """
    
    #Perform cleanup
    
    #Step 1: Replace URLs & translate to english
    clean_series = replace_url_series(pd_series)
    #clean_series = translate_to_english_series(clean_series)
    
    #Step 2: Remove repeated letters from words if repeated>=3 times 
    # to 2 times
    clean_series = remove_repeated_series(clean_series)
    
    #Step 3: Spacy_Lemmatize
    clean_series = spacy_lemmatize_series(clean_series)
    
    #Step 4: Remove single-letter words
    clean_series = remove_singles_series(clean_series)
    
    #Step 5: Translate chat_abbreviations
    clean_series = translate_chat_abbv_series(clean_series)
    
    #Step 6: Remove punctuation
    clean_series = remove_punct_series(clean_series)
    
    #Step 7: Remove numeric data
    clean_series = remove_digits_series(clean_series)
    
    return clean_series