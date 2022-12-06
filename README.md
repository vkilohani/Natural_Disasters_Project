# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py DisasterResponse.db classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. In the address bar in a browser type: 'localhost:3000'

5. Click the `Disaster Response Project` button to open the homepage. Type in a query for message classification and click 'Classify Message' button to see the classification results.
