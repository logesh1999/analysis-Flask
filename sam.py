from flask import Flask, request, render_template
import pandas as pd
import joblib



# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------

# Running the app



@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier
        clf = joblib.load("model(pkl).pkl")

        # Get values through input bars
        start_date  = request.form.get("start_date")
        end_date = request.form.get("end_date")

        # Put inputs to dataframe
        X = pd.DataFrame([["start_date","end_date"]], columns=[["start_date","end_date"]])

        # Get prediction
        prediction = s.predict(X)[0]

    else:
        prediction = ""

    return render_template("website.html", output=prediction)

if __name__ == '__main__':
    app.run(debug = True)