# Disaster Response Pipeline Project
This project is part of Udacity's Data Scientist Nanodegree. The aim of this project is to build a web app that can classify messages during disasters. This application will be able to help thousands in the event of a disaster. The ability to categorize responses on the fly will allow quicker and more accurate responses.

## Project Structure
The project consists of three main components:

- ETL Pipeline: This component takes raw data from CSV files, cleans it up, and organizes it in a SQLite database. The cleaned data can then be used by a machine learning model to train and make predictions.

- Machine Learning Pipeline: This component takes cleaned data from a SQLite database, trains a machine learning model on the data, and saves the trained model as a file that can be used later.

- Web Application: This component allows users to enter new messages and see the model's classification results immediately.


## How to Run the Project
### Set-up
Install the full repository to you machine and open in your IDE

Install the required Libraries:
- Pandas
- Numpy
- NLTK
- sqlalchemy
- pickle
- sys
- Plotly
- Flask
- Joblib

Follow the instructions within the README within the Models Folder.

## Files in the Repository
app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- DisasterResponse.db # database to save clean data to

models

|- train_classifier.py

|- classifier.pkl # saved model

|- test.py # a file to test small adjustments in and use as a notes file

|- README.md # additional instructions to run scripts once open in IDE

README.md

### Acknowledgements
Thank you to Rajat for all of the assistance on my questions posted to the knowledge forums.
