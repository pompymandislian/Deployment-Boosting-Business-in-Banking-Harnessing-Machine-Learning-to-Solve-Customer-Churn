import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Open data train
with open('D:/BOOTCAMP/project/Project Pribadi/ml churn/X_train_fix.pkl', 'rb') as file:
    # Call load method to deserialize
    X_train = pickle.load(file)

# Open model svm
with open('D:/BOOTCAMP/project/Project Pribadi/ml churn/model_svm.pkl', 'rb') as file:
    # Call load method to deserialize
    model_svm = pickle.load(file)

app = FastAPI()

class DataInput(BaseModel):
    gender_Male: int
    country_France: int
    Age_young: int
    gender_Female: int
    country_Spain: int
    country_Germany: int
    Age_mature: int
    active_member: int
    tenure: int
    balance: float
    credit_score: float

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API'}

origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/prediction')
async def get_data(data: DataInput):
    try:
        input_list = [
            data.gender_Male, data.country_France, data.Age_young,
            data.gender_Female, data.country_Spain, data.country_Germany,
            data.Age_mature, data.active_member, data.tenure,
            data.balance, data.credit_score
        ]
        # Reshape input data to match the shape used during training
        input_data = [input_list]

        result_prediction = model_svm.predict(input_data)[0]

        if result_prediction == 0:
            # Jika hasil prediksi adalah 0
            print("Hasil customer adalah unchurn")
            return {'prediction is': 'unchurn'}
        else:
            # Jika hasil prediksi adalah 1
            print("Hasil customer adalah churn")
            return {'prediction is': 'churn'}

    except ZeroDivisionError:
        # Tangani ZeroDivisionError jika terjadi
        return {"error": "Cannot divide by zero"}
