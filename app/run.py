import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Categorized_Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    ## extract data needed for The First Visual
    # calculate the proportion of messages of each category
    cat_props = df[df.columns[4:]].sum()/len(df)
    
    # get the percentage from the proportions calculated
    cat_percent = cat_props * 100
    
    # round the percentages to two decimals only
    cat_percent = round(cat_percent,2)
    
    # re-ordering the percentages in a descending order ( This will cause the category with the largest proportion to appear on the left of the chart)
    cat_percent = cat_percent.sort_values(ascending = False) 
    
    # get a categories names list (ordered based on the previous sorting method)
    cat_names = list(cat_percent.index)
    
    
    ## extract data needed for The Second Visual to show the top ten categories in term of the messages count 
    # get the count of the messages in each category
    categories = df.iloc[:,4:]
    
    # calculate the count of messages in each category re-ordering the percentages in a descending order
    message_counts = categories.sum().sort_values(ascending=False)
    
    # get the top ten most frequent categories
    message_counts = message_counts[1:11]
    
    # get a categories names list (ordered based on the previous sorting method
    top_ten_cat = list(message_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_percent
                )
            ],

            'layout': {
                'title': 'Percentages of Messages per Category',
                'yaxis': {
                    'title': "Percentage(%)"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_ten_cat,
                    y=message_counts
                )
            ],

            'layout': {
                'title': 'Top Ten Categories in term of Count of Messages',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
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
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()