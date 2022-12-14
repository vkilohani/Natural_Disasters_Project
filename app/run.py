import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


# load data
database_filepath = os.path.abspath("../data/DisasterResponse.db")
engine = create_engine('sqlite:///'+database_filepath)
df = pd.read_sql(database_filepath, engine)

# load model
model_filepath = os.path.abspath("../models/classifier.pkl")
model = joblib.load(model_filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Function renders the /templates/master.html template and shows some 
    statistics of the data used for training in the flask app.
        
        Args:
        -----
            No args
        Returns:
        --------
            No return value
        """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    counts = df.iloc[:, 4:-1].sum().sort_values(ascending=True)
    names = counts.index
    numbers = counts.values
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
                {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },   
        {
            'data': [
                Bar(
                    x=numbers,
                    y=names,
                    orientation = 'h',
                    width = 0.8
                )
            ],

            'layout': {
                'title': 'Incidence Count Bar Chart',
                'yaxis': {
                    'title': {'text': "Disaster category",
                              'standoff': 200
                    }           
                },
                'xaxis': {
                    'title': "Incidences"
                },
                'bargap': 1,
                'height': 900,
                'width': 1200,
                'margin': {'l':150, 'r': 100, 'b':200, 't':100},
                'separators': ','
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Function renders the /templates/master.html template in the flask 
    app which takens in a user query and classifies the message into one 
    of the 36 disaster categories.
    
        Args:
        -----
            No args
            
        Returns:
        --------
            No return value
    """

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict(pd.Series([query])).values[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Runs the flask app.
    
        Args:
        -----
            No args
        
        Returns:
        --------
            No return value
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()