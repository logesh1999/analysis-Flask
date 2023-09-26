import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask("Forecasting Analysis")

model = pickle.load(open("model(pkl).pkl", 'rb'))

@app.route("/")
def home():
    return render_template('hello.html')

@app.route("/api", method=['POST'])
def predict():

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']

    data = request.get_json(force=True)
    predict = [[data[inputQuery1, inputQuery2]]]
    request1 = np.array(predict)
    print(request)
    prediction = model.get_prediction(predict)
    pred = prediction[0]
    print(pred)

    return jsonify(int(pred))



if __name__ == "__main__":
    app.run(debug = True)