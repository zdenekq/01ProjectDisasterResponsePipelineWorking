# Disaster Response Pipeline Project

### Summary
During disasters emergency authorities are overloaded by messages, so we have to be able to classify messages into different categories. This automatic preread allow us to better understand situation and organise help in more effective way.

The project was split to:
1. ETL pipeline processing via process_data.py. Two .csv files disaster_categories.csv,disaster_messages.csv  were read, cleaned,wrangled, merged and uploaded to SQLLite database DisasterResponse.db
2. ML pipeline via launched train_classifier.py reads data from DisasterResponse.db database and analysed them via NLTK, scikit-learn pipeline with multioutput clasification and GridSearchCV. The output is stored in Pickle classifier.pkl.
3. New message is clasified via trained model in a Flask web after launching  run.py . Model is uploaded from classifier.pkl and business logic is at run.py. For figures are used data from DisasterResponse.db.


•	app    
| - template   
| |- master.html # main page of web app   
| |- go.html # classification result page of web app     
|- run.py # Flask file that runs app   
•	data    
|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process   
|- process_data.py     
|- DisasterResponse.db # database to save clean data to   
•	models      
|- train_classifier.py    
|- classifier.pkl # saved model      
•	README.md          



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage



