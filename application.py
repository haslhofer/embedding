from flask import Flask, jsonify, request, make_response, current_app, render_template, session, redirect, url_for

from flask_bootstrap import Bootstrap
from flask_moment import Moment

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist



#todolist0 = ['Buy flour and go shopping and then go swimming']
#todolist1 = ['read book about neural networks and machine learning']

#todolists = [todolist0, todolist1]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fhhuiwhksn'

bootstrap = Bootstrap(app)
moment = Moment(app)

model = SentenceTransformer('bert-base-nli-mean-tokens')

class SetupForm(FlaskForm):
    todo1 = StringField('Enter the first todo of the first list', validators=[DataRequired()])
    todo2 = StringField('Enter the first todo of the second list', validators=[DataRequired()])
    submit = SubmitField('Save list')

class NameForm(FlaskForm):
    name = StringField('Enter a new todo. I will guess which list it belongs to', validators=[DataRequired()])
    submit = SubmitField('Guess which list it belongs to')


@app.before_request
def before_request_func():
    print("before_request completed!")


@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('setupitems'))


@app.route('/setupitems', methods=['GET', 'POST'])
def setupitems():

    form = SetupForm()
    if form.validate_on_submit():
         # we have a new string
        todo1 = form.todo1.data 
        todo2 = form.todo2.data

        session['todo1'] = [todo1]
        session['todo2'] = [todo2]
        
        return redirect(url_for('additems'))

    return render_template('setup.html', form=form)


@app.route('/additems', methods=['GET', 'POST'])
def additems():

    
    form = NameForm()
    if form.validate_on_submit():
        todolist1 = session['todo1']
        todolist2 = session['todo2']
         # we have a new string
        todolist = [getFullString(todolist1), getFullString(todolist2) ]
        idx, distance, distances = getClosestIndex(form.name.data, todolist)
        session['confidence1'] = 1-distances[0]
        session['confidence2'] = 1-distances[1]
        if (idx == 0):
            session['todo1'].append(form.name.data)  
        else:
            session['todo2'].append(form.name.data)
         
        session['name'] = form.name.data
        session['idx'] = idx + 1   # bucket starts at 0
        
        return redirect(url_for('additems'))

    return render_template('index.html', form=form, name=session.get('name'), bucket = session.get('idx'), list1 = session['todo1'], list2 = session['todo2'], confidence1 = session.get('confidence1'), confidence2 = session.get('confidence2'))

# General pages

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Test routines

@app.route("/test", methods=['GET', 'POST'])
def indexold():
    
    #response = make_response('<h1>Hello World2</h1>')
    #response.set_cookie('state','41')
    return render_template('index.html', todo1 = line1)
    return response
    #return jsonify({'message': 'Hello world!'})


@app.route('/api', methods=['GET'])
def api2():
    myjson = request.get_json()
    idx, distance = getClosestIndex(myjson[0], myjson[1:])
    
    rv = jsonify([idx, distance])
    return rv


# Utilities
def getFullString(x):
    res = ''
    for y in x:
        res = res + y + '. '
    return res




def getClosestIndex(query, sentences):
    

    # sentences = ['Buy milk, meat, groceries', 
    #            'Book airplane, reserve seats, check in and remember frequent flyer miles']

    # Each sentence is encoded as a 1-D vector with 78 columns
    sentence_embeddings = model.encode(sentences)

    # print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))

    # print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])

    # code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py

    # query = 'Get flour' #@param {type: 'string'}

    queries = [query]
    query_embeddings = model.encode(queries)

    # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    number_top_matches = 3 #@param {type: "number"}

    # print("Semantic Search Results")

    for query, query_embedding in zip(queries, query_embeddings):
        distances = cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        #print("\n\n======================\n\n")
        #print("Query:", query)
        #print("\nTop 5 most similar sentences in corpus:")

        for idx, distance in results[0:number_top_matches]:
            return idx, distance, distances
        #    print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)