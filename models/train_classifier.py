# import libraries
import sys
import pandas as pd 
import numpy as np 
import nltk
import sqlite3
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt','wordnet','stopwords'])
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


    

def load_data(database_filepath, table_name='Categorized_Messages'):
    '''
    - Loads the data from the SQL Database.
    - define the feature(X) and labels(Y)
    
    Args:
        database_filepath : the filepath into which the datbase has been saved.
        
    Returns:
        X : The features dataframe.
        Y : The target(labels) dataframe.
        category_names : The categories to which a message can belong ( it will be used for data visualization (app)).
    '''    
    # load the data from the SQL database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    

     # extract the database file name from the file path
     
     #db_filename = database_filepath.split("/")[-1]
     #database_filename = db_filename.split(".")[0]

    # run a query to read the sql database
    df = pd.read_sql_table(table_name,engine)
    # df = pd.read_sql('SELECT * FROM {}'.format(database_filename), engine)
    #df = pd.read_sql('SELECT * FROM {}'.format(table_name), engine)
    
    # define the feature(X) and labels(Y)
    X = df['message']
    Y = df.drop(['id','message', 'original', 'genre'], axis = 1)
    
    # getting the category names
    category_names = Y.columns
    
    return X, Y, category_names 

def tokenize(text):
    '''
    Tokenizes the given text.
    
    Args:
        text: The text string given by the user through the app.
    
    Returns:
        A list of clean tokens.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        # normalizing by lowercaseing all the words and removing punctuation
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        # add the clean_tok to the clean_tokens list
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    '''
    - Creates a pipeline
    - Builds the classification model with the halp of Grid Search
    
    Returns:
    The built model after performing grid search
    '''
    # model pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # hyper-parameter grid
    parameters = {
                'vect__ngram_range': ((1, 1), (1, 2)),
                'tfidf__use_idf': (True, False),
                'clf__estimator__min_samples_split': [2, 4]
                }

    # create the model
    cv = GridSearchCV(pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3,
            n_jobs=4)
    model = cv
    return model
    

def evaluate_model(model, X_test, Y_test):
    '''
    Evaluates the model by generateing a Classification Report on the model
    
    Args: 
    Model : Our trained model
    X_test: Test features
    Y_test: Test labels
    category_names: String array of category names
    
    Returns: 
    The Classification Report related with the model.
    '''
    # get the predictions for the test data 
    Y_pred = model.predict(X_test)
    
    # iterate through the columns and call the sklearn's classification_report on each column to get 
    # the evaluation of the model's performance for each column.
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
        print('Accuracy: {}'.format(np.mean(Y_test[col].values == Y_pred[:, i])))
    

def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle
    
    Args:
        model: Ourtrained model
        model_filepath: The filepath into which we want to save the model
        
    Returns:
        Nothing
    """
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
