from flask import Flask, jsonify, request
app = Flask(__name__)


incomes = [
    {'description': 'salary', 'amount': 5000}
]


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route('/incomes')
def get_incomes():
    return jsonify(incomes)


@app.route('/incomes', methods=['POST'])
def add_income():
    incomes.append(request.get_json())
    return request.get_json(), 200
