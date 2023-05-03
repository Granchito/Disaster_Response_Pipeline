import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Filepath of the messages dataset.
        categories_filepath (str): Filepath of the categories dataset.

    Returns:
        df (pandas.DataFrame): Merged Pandas DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Cleans the input DataFrame by splitting the 'categories' column into separate category columns,
    converting their values to binary, and dropping duplicates.

    Args:
        df (pandas.DataFrame): Input DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = categories.iloc[0, :].apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to binary
    for column in categories:
        categories[column] = categories[column].apply(lambda x: 1 if int(x.split('-')[1]) > 0 else 0)

    # drop original categories column and concatenate the new binary ones
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the clean disaster response data in a SQLite database.

    Args:
    df (pandas.DataFrame): Cleaned disaster response data.
    database_filename (str): Path to SQLite database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CleanDisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories datasets as the first and second argument '
              'respectively, as well as the filepath of the database to save the cleaned data to as the third argument.'
              '\n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
