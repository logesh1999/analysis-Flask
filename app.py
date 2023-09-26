from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask('Forecast Analysis')

model = pickle.load(open('model(pkl).pkl', 'rb'))
@app.route("/")
def index():
    return render_template('hello.html')


@app.route("/predict", methods = ["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('hello.html')


@app.route('/results', methods = ['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug = True)