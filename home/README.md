# Disaster Response Pipeline Project

### Summary
During disasters emergency authorities are overloaded by messages, so we have to be able to classify messages into different categories. This automatic preread allow us to better understand situation and organise help in more effective way.

The project was split to:
1. ETL pipeline processing. Two .csv files were read, cleaned,wrangled, merged and uploaded to SQLLite database
2. ML pipeline read data from database and analysed them via NLTK, scikit-learn pipeline with multioutput clasification and GridSearchCV.
3. New message can be clasified via trained model in a Flask web


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage



