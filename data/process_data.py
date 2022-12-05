import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.custom_utils import full_data_path
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from categories_filepath
        Args:
        ---- 
            messages_filepath: str
                name of csv file with messages
            categories_filepath: str 
                name of csv file with categories
        Returns:
        --------
            df: pandas.DataFrame
                dataframe obtained from merging the two input csv files
    """
    print("\nUsing", full_data_path(messages_filepath), 
          "and", 
          full_data_path(full_data_path(categories_filepath)),
          '.',
          "\n\nTo be merged and cleaned up.\n"
          )
    
    messages_df = pd.read_csv(full_data_path(messages_filepath))
    categories_df = pd.read_csv(full_data_path(categories_filepath))
    
    merge_columns = 'id' #Column(s) used for merging
    
    df = messages_df.merge(categories_df, on = [merge_columns], how = 'left') #Merge the dataframes
    return df


def clean_data(df):
    """Cleans the dataframe loaded by merging of messages and 
    categories files
    args:
    - dataframe
    returns:
    - cleaned dataframe
    """
    categories_df = df['categories'].str.split(pat = ';', expand=True)
    row0 = categories_df.iloc[0, :].values
    category_column_names = [x[:-2] for x in row0]
    categories_df.columns = category_column_names
    
    for column in categories_df.columns:
        categories_df[column] = categories_df[column].map(lambda x: x[-1])
        categories_df[column] = pd.to_numeric(categories_df[column])
        
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories_df], axis=1)
    
    df = df.drop_duplicates() #Drop duplicates
    
    #Final Cleaning
    repeat_ids = df['id'].value_counts()[df['id'].value_counts()>1].index
    df = df[~df['id'].isin(repeat_ids)]
  
  
    for column in category_column_names:
        df.drop(df[~df[column].isin([0, 1])].index, inplace=True) #Drop observations that are not properly binary classified
        
    print("Number of observations in the cleaned up data:", len(df), "\n")
        
    return df



def save_data(df, database_filename):
    """saves the dataframe to a sequel file in the ../data folder
    args:
    - dataframe
    - name of the database file
    
    If the database file already exists, it will be overwritten
    """
    engine = create_engine('sqlite:///'
                           + full_data_path(database_filename)
                           )
    
    df.to_sql(full_data_path(database_filename),
              engine,
              index = False,
              if_exists="replace")
    
    print("\nCleaned the data.",
        "\nSaved the cleaned database to: " 
        + full_data_path(database_filename),
        "\n"
        )  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()