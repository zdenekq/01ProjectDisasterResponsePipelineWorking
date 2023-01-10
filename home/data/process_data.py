import sys
import os
from os import path

execution_filepath = os.path.dirname(__file__)
print(r'folder path of the file we are executing ='+' {}'.format( execution_filepath ))
messages_filepath = execution_filepath# os.path.realpath(os.path.join(execution_filepath, '..',  'data'))
categories_filepath = execution_filepath
print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))


# from config.definitions import *


import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    print("Item messages_filepath is a file: " + str(path.isfile(messages_filepath)))
    print("Item categories_filepath is a file: " + str(path.isfile(categories_filepath)))
    print("Item process_data.py is a file: " + str(path.isfile("process_data.py")))

    messages = pd.read_csv(messages_filepath,dtype=str,index_col=False) 

    categories = pd.read_csv(categories_filepath,dtype=str,index_col=False)
    

    # merge datasets
    df = pd.merge(messages,categories, how='inner', on='id')
    df.reset_index(drop=True, inplace=True)
    return df

    pass




def clean_data(df):


    pass


def save_data(df, database_filename = 'InsertDatabaseName.db'):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine,if_exists='replace', index=False)
    return
    pass  


def main():
    # if len(sys.argv) == 4: # former with parameters
  
    
    if len(sys.argv) == 1: # my for testing

        '''
        script arguments: 
        sys.argv[1] is messages_filepath = 'messages.csv'
        sys.argv[2] is categories_filepath = 'categories.csv'
        sys.argv[3] is database_filepath = 'InsertDatabaseName.db'
        where sys.argv[0] is the name of the script
        '''
        
        # sys.argv is a list in Python that contains all the command-line arguments passed to the script. 
        # https://www.knowledgehut.com/blog/programming/sys-argv-python-examples
        '''
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # sys.argv[0] is the name of the script

        print ("Number of arguments:", len(sys.argv), "arguments") # my added
        print ("Argument List:", str(sys.argv)) # my added

       # print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        '''
       

        
        
        df = load_data(messages_filepath = 'disaster_messages.csv', categories_filepath = 'disaster_categories.csv')
        # df = load_data(messages_filepath, categories_filepath) # původní

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath = 'InsertDatabaseName.db')
        
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