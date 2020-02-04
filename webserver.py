from flask import Flask, jsonify, request

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return jsonify({'message': 'Hello world!'})



@app.route('/api', methods=['GET'])
def api2():
    myjson = request.get_json()
    idx, distance = getClosestIndex(myjson[0], myjson[1:])
    sentences = ['Buy milk, meat, groceries', 
                'Book airplane, reserve seats, check in and remember frequent flyer miles']

    rv = jsonify(idx)
    return rv


def getClosestIndex(query, sentences):
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # sentences = ['Buy milk, meat, groceries', 
    #            'Book airplane, reserve seats, check in and remember frequent flyer miles']

    # Each sentence is encoded as a 1-D vector with 78 columns
    sentence_embeddings = model.encode(sentences)

    print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))

    print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])

    # code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py

    # query = 'Get flour' #@param {type: 'string'}

    queries = [query]
    query_embeddings = model.encode(queries)

    # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    number_top_matches = 3 #@param {type: "number"}

    print("Semantic Search Results")

    for query, query_embedding in zip(queries, query_embeddings):
        distances = cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx, distance in results[0:number_top_matches]:
            return idx, distance
            print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))

app.run(host='0.0.0.0', port=81)