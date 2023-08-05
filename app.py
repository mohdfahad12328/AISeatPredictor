import joblib
from fastapi import FastAPI 
from sklearn.tree import DecisionTreeClassifier
import uvicorn

app = FastAPI()

model =joblib.load('aiseatpredictor_decisionTree.joblib')

@app.get("/{rank}/{caste}")
def read_root(rank: int, caste: int):
    global model
    predictions = model.predict([[rank, caste]])
    return {"college": predictions[0][0], "branch": predictions[0][1]}

if __name__ == "__main__":
    uvicorn.run("app:app", port=3000, log_level="info")