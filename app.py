import pickle
import pickle
import sqlite3
import string
from urllib.parse import urlparse

import numpy as np
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
from tensorflow import keras

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/": {"origins": ""}})
# Load the saved model
loaded_dnn = keras.models.load_model('dnn_model.h5')
# Load the saved models
# Load the model from the file
with open('rbm_model.pkl', 'rb') as file:
    loaded_rbm = pickle.load(file)

@app.route('/ping', methods=['GET'])  # using post method to data send
def ping():
    return "running"

@app.route('/predict', methods=['POST','GET'])  # using post method to data send
def feedModel():
    # url = request.json.get('url')
    data = request.get_json()
    url = data['url']
    print("url is", url)

    result = getData(url)
    if result == "phishing":
        return jsonify({
                "status": "phishing",  # return the value within a dictionary json object
                "phishing_rate": "100"
            })
    elif result == "legitimate":
        return jsonify({
            "status": "safe",  # return the value within a dictionary json object
            "phishing_rate": "100"
        })
    else:
        encoded_input = np.array([preprocess_url(url)])
        rbm_transformed_input = loaded_rbm.transform(encoded_input)
        prediction_proba = loaded_dnn.predict(rbm_transformed_input)
        prediction = (prediction_proba > 0.5).astype(int).flatten()  # Apply threshold and flatten the array

        phishing_rate=prediction_proba*100

        print("Predicted by model: ", prediction, "Phishing Rate: ",phishing_rate[0][0])
        if prediction[0] == 0:
            return jsonify({
                "status": "safe",  # return the value within a dictionary json object
                "phishing_rate": str(phishing_rate[0][0])
            })
        else:
            return jsonify({
                "status": "phishing",  # return the value within a dictionary json object
                "phishing_rate": str(phishing_rate[0][0])
            })

# Function to preprocess URLs
def tokenize_url(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path


def preprocess_url(url):
    tokenized_url = tokenize_url(url)
    characters = string.ascii_letters + string.digits + string.punctuation
    encoding = {c: 0 for c in characters}
    for char in tokenized_url:
        if char in encoding:
            encoding[char] = 1
    return list(encoding.values())


def getData(url):
    # Connect to the SQLite database
    conn = sqlite3.connect('url_database.db')
    c = conn.cursor()

    # Query the data from the table
    c.execute("SELECT url_type FROM url_data WHERE url = ?", (url,))

    # Fetch and print the data
    rows = c.fetchall()

    if not rows:
        return "not-found"
    for row in rows:
        print(row[0])
        return row[0]

        # Close the cursor and connection
    c.close()
    conn.close()


if __name__ == '__main__':
    print("Running server")
    app.run(host='0.0.0.0',debug=True)
