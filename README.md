# Disaster Response Pipeline Project

## Overview:
In this Portfolio Project, A data set containing real messages that were sent during disaster events was provided.  Then, a machine learning pipeline to categorize these events was created so that we can send the messages to an appropriate disaster relief agency.
In addition, the project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data ( two visualizations). 

This project shows-off my Software and Data Engineering Skills, including the ability to create basic data pipeline, such as ETL and ML Pipelines and my ability to write a clean and organized code.


## Instructions for Running the Python Scripts:
1. Run the following commands in the project's root directory to set up your database and model.
o	To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
o	To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py
3. Go to http://0.0.0.0:3001/


## The Repository’s File:
#### Three important components in this repository are:
1- **process_data.py**, which includes the ETL Pipeline that process and clean data in preparation for model building by performing the following steps:
    •	Loads the messages and categories datasets
    •	Merges the two datasets
    •	Cleans the data
    •	Stores it in a SQLite database

2- **train_classifier.py**,  which includes the ML Pipeline used to fit, tune, evaluate, and export the model to a Python pickle (pickle is not uploaded to the repository).
This is done by performing the following steps:
    •	Loads data from the SQLite database
    •	Splits the dataset into training and test sets
    •	Builds a text processing and machine learning pipeline
    •	Trains and tunes a model using GridSearchCV
    •	Outputs results on the test set
    •	Exports the final model as a pickle file 

3- **run.py**, which includes the code of the Flask Web App. The code included starts the Python server for the web app and prepare visualizations.

 
