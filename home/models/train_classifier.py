
'''
# ML Pipeline Preparation
Follow the instructions below to help you create your ML pipeline.
### 1. Import libraries and load data from database.
- Import Python libraries
- Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
- Define feature and target variables X and Y

https://towardsdatascience.com/building-a-disaster-response-web-application-4066e6f90072

# prikazova radka # 
# cd home\models
# python train_classifier.py ../data/DisasterResponse.db classifier.pkl
# python -m debugpy --connect ./train_classifier.py ../data/DisasterResponse.db classifier.pkl

'''
# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

# import for machine learning

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

# for nlp
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download(['punkt', 'wordnet','stopwords'])


from sklearn.datasets import make_multilabel_classification

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
##########################################################################
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    load data function

    Input:
    database_filepath is path to Lite db

    Output:
    X is numpy feature
    Y is numpy label
    category_names is used for data figures   
    
    
    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath).connect()  
    df = pd.read_sql_table('InsertTableName', engine)
    # df.columns

    # removes rows which are null in message column
    df.dropna(subset=['message'],axis=0,inplace=True)

    # This machine pipeline should take in the message column as input 
    X_pandas = df.message
    # creates numpy
    X = X_pandas.values

    Y_pandas= df.drop(columns = ['id','message','original','genre'], axis = 1)
    Y = Y_pandas.values

    categories_columns = Y_pandas.columns
    category_names = categories_columns

    return X, Y, category_names


def tokenize(text):
    '''
    together normalisation + tokenization + lemmatization of text string
    
    Input:
    text string for processing

    Output:
    clean_tokens is a list containing normalised and lemmatised tokens from words

    '''
    
    """ Origin 


"""       
    # get rid nonalphanumeric characters
    # https://regexr.com/3a8p3 - testing possible     
    not_alphanumeric_reg = r"[^a-zA-Z0-9]"
    text = re.sub(not_alphanumeric_reg, " ", text)
    
    #tokenize 
    tokens = word_tokenize(text)    
    
    # init a remove stopwords
    stop_words = stopwords.words("english")
    #vystup = [lemmatizer.lemmatize(p) for p in tokens if not p in stop_words]
    
    #lematizer instance
    lemmatizer = WordNetLemmatizer()
    
    #stemmer instance 
    stemmer = PorterStemmer()       
    

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            #only low case
            # clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            noun_lemmatize_tok = lemmatizer.lemmatize(tok)
            verb_lemmatize_tok = lemmatizer.lemmatize(noun_lemmatize_tok,pos="v")
            stem_lem_token = stemmer.stem(verb_lemmatize_tok)

            new_tok = stem_lem_token.lower().strip() 
            clean_tokens.append(new_tok)

    return clean_tokens
    
       
def tokenize2(text):
    # alternative to tokenize1 function giving same outputs
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
   
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens 
    
    


def build_model():
    '''
    ### 3. Build a machine learning pipeline
    This machine pipeline should take in the `message` column as 
    input and output classification results on the other 36 categories
    in the dataset. You may find the
    [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) 
    helpful for predicting multiple target variables.

    Output: GridSearchCV containing  a sklearn estimator
    '''

    # TfidfVectorizer is the combination of CountVectorizer + TfidfTransformer.
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # n_estimators = 10 is very low, if computer allows should be more 
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10)))
                        ])
    
    
    # Hyperparameters grid
    parameters = {'tfidf__use_idf': [True, False],
                    'clf__estimator__n_estimators': [5]} 




    # initialize 
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1) # cv=3, scoring='f1_weighted'

    model = cv 
    
    return model    


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Test your model
    Show the accuracy, precision, and recall of the tuned model.  

    Since this project focuses on code quality, process, and  pipelines, 
    there is no minimum performance metric needed to pass. 
    However, make sure to fine tune your models for accuracy,
    precision and recall to make your project stand out

    Input:
        model: sklearn.model_selection.GridSearchCV with a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each message
        category_names: Disaster category names

    Output:
    Evaluated model
    '''

    
    Y_pred = model.predict(X_test)
    
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    Y_test_df = pd.DataFrame(Y_test, columns = category_names)
    
    for i in range(36):
        
         print('Category: {}'
            "\n\n\n classification_report: {}".format(category_names[i],
            classification_report(Y_test_df.iloc[:,i], Y_pred_df.iloc[:,i])))


    
    print(classification_report(Y_test_df.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = category_names))
    # fit
    # model.fit(X_train, Y_train)
    # model.best_params_  
    
    
    return 



def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)





def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # vypisuje database_filepath = ../data/DisasterResponse.db
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("""
        \t X.shape: {}
        \t Y.shape: {}
        \t X_train.shape: {}
        \t X_test.shape: {}
        \t Y_train.shape: {}
        \t Y_test: {}""".format(X.shape,Y.shape,X_train.shape, X_test.shape, Y_train.shape , Y_test.shape)
            )



        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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