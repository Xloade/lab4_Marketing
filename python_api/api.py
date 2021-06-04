#%%
import flask
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import *
import json
#%%
encoder = joblib.load("../data_preprocessing/encoder.pkl")
scaler = joblib.load("../data_preprocessing/scaler.pkl")
model = joblib.load("../python MLPclassifier/MLPClassifier.pkl")
#%%
def MLPCmodel(inputs):
    scaled = scaler.transform([inputs])
    y = model.predict(scaled)
    y_prob = model.predict_proba(scaled)
    return y[0],y_prob[0][y[0]-1]
# %% test
test = [58,2,3,2,0,2143,1,0,1,5,2,261,1,-1,0,0]
answer = MLPCmodel(test)
#%%
app = Flask(__name__)
@app.route('/model')
def model_call(): 
    class_index, prob = MLPCmodel(list(request.args.values()))
    return json.dumps({'result_class': str(class_index),
                       'result_probability': str(prob)})
@app.route("/")
def home():
    return render_template("api.html")
app.run()

# %%
