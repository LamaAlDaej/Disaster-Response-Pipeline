# Disaster Response Pipeline Project

## Summary of the Project
This is my second project in Udacity's Data Scientist Nanodegree. It analyzes disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. It also displays visualizations of the data. Below is a screenshot of the web app's master page.
![Web App Master Page Screenshot][(web-app-screenshot.png)]


### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run the web app: `python run.py`

## Description of the Repository's Files
- 'data': a folder contains the CSV data files 'disaster_messages.csv' and 'disaster_categories.csv', database 'DisasterResponse.db', and ETL pipeline 'process_data.py'
- 'models': a folder contains the Machine Learning pipeline 'train_classifier.py' and an exported classifier as a Pickle file 'classifier.pkl'
- 'app': a folder contains the web app 'run.py' and its HTML files 'master.html' and 'go.html' in the 'templates' folder
- 'notebooks': a folder contains the Jupyter Notebooks of ETL and ML preparations 'ETL Pipeline Preparation.ipynb' and 'ML Pipeline Preparation.ipynb'
- 'web-app-screenshot.png': a screenshot of the web app's master page with my additional visual

## Acknowledgement
I applied what I learned from Udacity's Data Scientist Nanodegree's lessons, the template codes are from Udacity, and all references are cited within the scripts.
