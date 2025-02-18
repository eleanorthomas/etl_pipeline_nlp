import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load message data and category data and return merged dataframe"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """Convert category data into numerical flags and remove duplicates"""
    
    # Set category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Clean category values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories in original dataframe
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)    

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """Save cleaned data into SQLite database"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename.split('/')[-1].replace('.db',''), engine, index=False)

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
