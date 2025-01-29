from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from fastapi.responses import JSONResponse

app = FastAPI()

# Add a simple root route for testing
@app.get("/")
def read_root():
    return {"message": "API is working!"}


@app.head("/")
def head_root():
    return JSONResponse(status_code=200)

# Load processor and model pickle files
try:
    if not os.path.exists("artifacts/processor.pkl") or not os.path.exists("artifacts/model.pkl"):
        raise FileNotFoundError("Processor or model pickle file is missing.")
    
    with open("artifacts/processor.pkl", "rb") as processor_file:
        processor = pickle.load(processor_file)
    with open("artifacts/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
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

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
