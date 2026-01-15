from fastapi import FastAPI
import numpy as np

data = np.load("model_data.npz", allow_pickle=True)
w = data["w"]
b = data["b"]
mean = data["mean"]
std = data["std"]
columns = data["columns"]

def sigmoid(x, w, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))

def predictml(x, w, b):
    y1 = sigmoid(x, w, b)
    return (y1 >= 0.7).astype(int)

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message':'Productivity model API'}

@app.post('/predict')
def predict(data: dict):
    """
    Docstring for predict
    
    :param data: Description
    :type data: dict
     {"features"=[,,,,]}
    """
    features = (features - mean[:-1]) / std[:-1]

    features=np.array(data['features']).reshape(1,-1)
    pred = "Yes" if predictml(features,w,b) == 1 else "No"

    return {'will_you_be_productive':pred}


