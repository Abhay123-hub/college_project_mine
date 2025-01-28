from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Load processor and model pickle files
try:
    with open("artifacts/processor.pkl", "rb") as processor_file:
        processor = pickle.load(processor_file)
    with open("artifacts/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise Exception(f"Error loading pickle files: {str(e)}")


# Define the function to convert JSON to Pandas DataFrame
def json_to_df(json_data):
    try:
        # Convert JSON to a Pandas DataFrame
        df = pd.DataFrame(json_data)
        return df
    except Exception as e:
        raise ValueError(f"Error converting JSON to DataFrame: {str(e)}")


# Define Pydantic model for input validation
class InputData(BaseModel):
    data: list  # JSON input must be a list of dictionaries


@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Step 1: Convert JSON to DataFrame
        input_df = json_to_df(input_data.data)

        # Step 2: Process the DataFrame using the processor
        processed_data = processor.transform(input_df)

        # Step 3: Make predictions using the model
        predictions = model.predict(processed_data)

        # Step 4: Return the predictions as JSON
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
