
'''
ETL Pipeline Preparation
Creates  ETL pipeline from disaster_messages.csv , disaster_categories.csv
and stores data in database DisasterResponse.db

cd home\data
execute on command line:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
# coding utf-8

'''

# imported libraries
import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
'''


'''
def load_data(messages_filepath, categories_filepath):
    '''
    def load_data(messages_filepath, categories_filepath):
    pass

    input messages_filepath.csv and  categories_filepath.csv
    output in df are merged messages and categories
    
    '''  

    # print(type(messages_filepath))
    # cwd = os.getcwd()

    # load messages dataset 
    messages = pd.read_csv(messages_filepath,dtype=str,index_col=False) 
    messages.head()

    # load categories dataset
    categories = pd.read_csv(categories_filepath,dtype=str,index_col=False)
    categories.head()
    

    # merge datasets
    # Merge the messages and categories datasets using the common id
    df = pd.merge(messages,categories, how='inner', on='id')
    df.reset_index(drop=True, inplace=True)
    return df

    




def clean_data(df):
    '''
    input is dataframe df
    output is cleaned df

    '''

    '''
    ## 1. Split `categories` into separate category columns.
    - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
    - Use the first row of categories dataframe to create column names for the categories data.
    - Rename columns of `categories` with new column names.
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True) 
    # categories.head()


    # select the first row of the categories dataframe
    row = categories.iloc[1]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    categories.columns = row.str.split(pat='-',expand=True)[0]
    # print(categories.columns)


    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str.split(pat='-',expand=True)[1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #categories.head()



    '''
    All columns have values 0 or 1 only column related has some values 2,
     which are evidently wrong. We are going to drop such raws.

     ### 5. Replace `categories` column in `df` with new category columns.
    - Drop the categories column from the df dataframe since it is no longer needed.
    - Concatenate df and categories data frames. 
    '''
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # df.head()

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],join="inner",axis=1)
    # df.head()

    # drop rows with bad data quality
    df = df[df.related != 2]
    df.shape


    '''
    ### 6. Remove duplicates.
    - Check how many duplicates are in this dataset.
    - Drop the duplicates.
    - Confirm duplicates were removed.
    
    
    '''

    # drop duplicates
    df = df.drop_duplicates(subset=['id'])
    # df shape before uploading to database
    df.shape

    # check number of duplicates
    # df.shape[0]- df['id'].nunique()
    

    return df





def save_data(df, database_filename):
    '''
    Save the clean dataset df into an sqlite database wirh database_filename.
    You can do this with pandas 
    [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) 
    combined with the SQLAlchemy library. 
    Remember to import SQLAlchemy's `create_engine` 
    in the first cell of this notebook to use it below.

    def save_data(df, database_filename):
    pass  
    '''
    # InsertDatabaseName.db
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine,if_exists='replace', index=False)
    return
     




def main():
    # if len(sys.argv) == 4: # former with parameters
    
    
    if len(sys.argv) == 4: # my for testing
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
       
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
        '''

        '''
        # if i need absolute path
        cwd = os.getcwd()
        messages_filepath = f'{cwd}/home/data/'+ messages_filepath
        categories_filepath = f'{cwd}/home/data/'+ categories_filepath
        database_filepath = f'{cwd}/home/data/'+ database_filepath
        '''
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
        
        print('wrong end with sys.argv[1:] = {}'.format(sys.argv[1:]))
        print('\n wrong end')


if __name__ == '__main__':
    main()