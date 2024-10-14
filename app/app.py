from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes.predict import prediction_router
from routes.data import data

app = FastAPI()

# CORS configuration
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(data)