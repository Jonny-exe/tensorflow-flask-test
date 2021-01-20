# index.py
from flask import Flask, jsonify, request
import csvFuncs as funcs
# from csvFuncs import add_row
app = Flask(__name__)


incomes = [
    {'description': 'salary', 'amount': 5000}
]


@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route('/add_row', methods=['POST'])
def add_income():
    json = request.get_json()
    funcs.add_row(dict(json))
    return json, 200

    
