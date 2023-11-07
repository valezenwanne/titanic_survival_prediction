# import libraries
from flask import Flask, request, render_template
import numpy as np
import pickle


# define the app
app = Flask(__name__)


# import the model and Standardscaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))


# define the home page
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    int_feat = [int(x) for x in request.form.values()]
    print(int_feat)
    feat = [np.array(int_feat)]
    print(feat)
    scaled_feat = scaler.transform(feat)
    prediction = model.predict(scaled_feat)
    output = []
    if prediction == 0:
        output = 'NOT SURVIVE'
    elif prediction == 1:
        output = 'SURVIVE'
    return render_template('index.html', prediction_text="This passenger will {}".format(output))


# start the app
if __name__ == '__main__':
    app.run(debug=True)

