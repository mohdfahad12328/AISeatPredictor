import joblib
from fastapi import FastAPI 
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

model =joblib.load('aiseatpredictor_decisionTree.joblib')

@app.get("/{rank}/{caste}")
def read_root(rank: int, caste: int):
    global model
    predictions = model.predict([[rank, caste]])
    return {"college": predictions[0][0], "branch": predictions[0][1]}
