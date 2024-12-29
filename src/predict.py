import pandas as pd
from cleaning_name import CleaningName
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

def predict(model_path: str, data_path: str):



    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    data = json.load(open(data_path, "r"))
    data = pd.Series(data).to_frame().T

    return model.predict_proba(data)

if __name__ == "__main__":
    model_path = "final_model.pickle"
    data_path = "test.json"
    print(predict(model_path, data_path))
