import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    - Takes two file paths as input.
    - Loads the CSV files related with the given filepaths as two dataframes.
    - Merges the two datasframes into a single combined dataframe based on their common column 'id'
    
    Args:
        messages_filepath: The Filepath of the messages CSV Files
        categories_filepath: The Filepath of the categories CSV File
        
    Returns:
        df : The dataframe that results from merging the messages and categories dataframes.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath,engine='python', encoding='utf-8', error_bad_lines=False)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets together based on their common column 'id'
    df = messages.merge(categories, on= 'id')
    return df


def clean_data(df):
    '''
    Cleans the merged dataframe for future use by our ML model through performing the following : 
    - Split the values in the 'categories' column into separate category columns.
    - Use the first row of categories dataframe to create column names for the categories data.
    - Rename columns of categories with new column names.
    - Convert category values to just numbers 0 or 1.
    - Replace categories column in df with new category columns by : 
        - Drop the categories column from the df dataframe since it is no longer needed.
        - Concatenate df and categories data frames.
        - Remove duplicates.
    
    Args:
        df : The Merged pandas dataframe returned from load_data() function.
        
    Returns:
        df : The Cleaned dataframe that is ready to be used by our ML model.
    '''
    # create a dataframe of the 36 individual category columns and split categories into separate category columns
    categories = df['categories'].str.split(pat= ';',expand = True)
    
    # select the first row of the categories dataframe
    row_values = categories.iloc[0].values

    # extract a list of new column names for categories using the 
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [] 
    for value in row_values:
        category_colnames.append(value[:-2])
    categories.columns = category_colnames
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # replace categories column in df with new category columns by dropping the original categories column from `df` and # concatenating the original dataframe with the new `categories` dataframe
    df.drop( columns = 'categories', axis = 1, inplace = True)
    df[categories.columns] = categories
    
    # remove duplicates by dropping them
    df.drop_duplicates(inplace = True)
    
    return df 


def save_data(df, database_filename = 'Disaster_Response', table_name='Categorized_Messages'):
    '''
    - Saves the clean dataset into an sqlite database.
    
    Args:
        - df: The cleaned dataframe that is ready to be used by our ML model.
        - database_filename: The database filename of to be given to the  cleaned dataframe when saving it as a SQL Database.
        
    Returns:
        None
    '''
    # use the provided database_filename to create the path into which the cleaned data is to be saved
    engine = create_engine('sqlite:///{}'.format(database_filename))
 
    # Save the clean dataset into an sqlite database    
    df.to_sql(table_name, engine, index=False, if_exists='replace', chunksize=600)

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