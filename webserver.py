from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return jsonify({'message': 'Hello world!'})


@app.route('/similarity', methods=['POST'])
def add_income():
  myjson = request.get_json()
  sentences = ['Buy milk, meat, groceries', 
             'Book airplane, reserve seats, check in and remember frequent flyer miles']

  rv = jsonify(myjson)
  print(rv)
  return rv, 204


app.run(host='0.0.0.0', port=81)