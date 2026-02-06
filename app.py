from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np

data = np.load("model_data.npz", allow_pickle=True)
w = data["w"]                 
b = data["b"]
mean = data["mean"][:7]        
std = data["std"][:7]          
columns = data["columns"]


def sigmoid(x, w, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))

def predictml(x, w, b):
    y1 = sigmoid(x, w, b)
    return (y1 >= 0.7).astype(int)

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get('/')
def reed_root(request: Request):
   return templates.TemplateResponse(
        "index.html",
        {"request": request, "columns": columns.tolist()}
    )
@app.post('/predict')
def predict(data: dict):
    """
    Docstring for predict
    
    :param data: Description
    :type data: dict
     {"features":[,,,,]}
    """
    if 'features' not in data:
        return {"error": "Missing 'features' in request body"}
    
    features=np.array(data['features']).reshape(1,-1)
    if features.shape[1] != w.shape[0]:
        return {
            "error": f"Expected {w.shape[0]} features, got {features.shape[1]}",
            "expected_order": columns.tolist()
        }
    
    features = (features - mean) / std

    pred = "Yes" if predictml(features,w,b) == 1 else "No"

    return {'will_you_be_productive':pred}


