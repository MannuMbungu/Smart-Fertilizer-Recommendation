
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import get_crop_model,get_fertilizer_model,get_input


description = """
### Crop Recommendation JSON Input 
    { "array": [N,P,K,temperature,humidity,ph,rainfall] }
### Fertilizer Recommendation JSON Input 
    { "array": [Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous,Soil Type,Crop Type] }
"""

app = FastAPI(description=description)

# ------------------------------------------

# Enabling CORS policy

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models
crop_model = get_crop_model()
fertilizer_model = get_fertilizer_model()

#--------------------------------------------------------------------

class InputArray(BaseModel):
    array: List[Union[float, str]] = Field(..., description="An array of floats or strings")

# @app.post("/crop_recommend")
# def array_endpoint(request: Request, input: InputArray):
    
#     print("Making Crop Prediction...")
#     # Make a prediction using the input array
#     prediction = crop_model.predict([input.array])
#     prediction = prediction.tolist()[0]
    
#     print("Returning Response...")    
#     # Return the prediction in the response
#     return prediction

@app.post("/crop_recommend")
def array_endpoint(request: Request, input: InputArray):
    try:
        print("Making Crop Prediction...")

        # Convert the input array to numeric values
        numeric_input = [float(val) if isinstance(val, (int, float, str)) else 0.0 for val in input.array]

        # Make a prediction using the numeric input array
        prediction = crop_model.predict([numeric_input])
        prediction = prediction.tolist()[0]

        print("Returning Response...")
        # Manually include CORS header
        response = JSONResponse(content=prediction)
        response.headers["Access-Control-Allow-Origin"] = origins[0]
        return response

    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}




@app.post("/fertilizer_recommend")
def array_endpoint(request: Request, input: InputArray):
    
    # process the input array
    x = get_input(input.array)
    
    print("Making Fertilizer Prediction...")
    # Make a prediction using the input array
    prediction = fertilizer_model.predict([x])
    prediction = prediction[0]
    
    print("Returning Response...")    
    # Return the prediction in the response
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5080)
    
    # changed 127.0.0.1 to 0.0.0.0 for railway.app deployment
    # you can go to "/docs" or "/redoc" endpoint to get the API documentation
    
    # CLI command
    # uvicorn app:app --host 0.0.0.0 --port 8080

