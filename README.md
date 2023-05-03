# Disaster Response Pipeline Project
This project is part of Udacity's Data Scientist Nanodegree. The aim of this project is to build a web app that can classify messages during disasters.

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
- Ideas folder: Not neccecary to run project
- App folder: Contains files for web application
- Data folder: Contains files for the data and data cleansing
- Models: Contains machine learning model and pkl file

### Acknowledgements
Thank you to Rajat for all of the assistance on my questions posted to the knowledge forums.
