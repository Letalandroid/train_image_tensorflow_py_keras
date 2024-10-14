from DepthwiseConv2D import CustomDepthwiseConv2D
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf

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

# Load model and labels once
model = tf.keras.models.load_model(
    "./model/keras_model.h5",
    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
    compile=False
)

with open("./model/labels.txt", "r") as f:
    class_names = f.readlines()

@app.post("/get-predict")
async def predict(img_test: UploadFile = File(...)):
    if img_test.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are allowed.")

    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(img_test.file).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
        normalized_image_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].split(' ')[1].strip()
        confidence_score = str(round(prediction[0][index] * 100, 2)) + "%"

        return class_name
        # return JSONResponse(content={"class_name": class_name, "confidence_score": confidence_score})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
