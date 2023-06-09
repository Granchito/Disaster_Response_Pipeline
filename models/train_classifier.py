import sys
import pandas as pd
import nltk
nltk.download('omw')
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Load cleaned data from a SQLite database into pandas DataFrame and extract message text, target labels, and category names.

    Args:
    database_filepath: str. Filepath of the SQLite database.

    Returns:
    X: pandas DataFrame. Message text.
    Y: pandas DataFrame. Target labels.
    category_names: list. Category names of the target labels.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('CleanDisasterResponse', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text by converting all text to lowercase, removing English stopwords, and lemmatizing words.

    Args:
    text (str): Text to be tokenized

    Returns:
    tokens (list): A list of cleaned and lemmatized tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Made N_Jobs=1 as I was unable to get the file to pickle using -1
def build_model():
    """
    Build a machine learning pipeline with GridSearchCV to classify the messages
    into multiple categories.

    Returns:
    cv: GridSearchCV object, the model with the best parameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [10]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=1, cv=2)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the trained model on the test set.

    Parameters:
    model (object): A trained scikit-learn pipeline model.
    X_test (array-like): A numpy array or Pandas DataFrame of the test feature data.
    Y_test (array-like): A numpy array or Pandas DataFrame of the test target data.
    category_names (list): A list of the target categories.

    Returns:
    None

    Prints out a classification report for each target category, including precision, recall, and f1-score.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"Category: {col}\n")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("-" * 80)


def save_model(model, model_filepath):
    """
    Save the trained model to a file in binary format.

    Parameters
    ----------
    model : object
        The trained model to be saved.
    model_filepath : str
        The path of the file to save the model to.

    Returns
    -------
    None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)



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
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
