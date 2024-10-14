from fastapi import APIRouter
from PIL import Image, ImageOps
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.DepthwiseConv2D import CustomDepthwiseConv2D
from utils import imgur
from sqlalchemy import func, select
from model.predict import predict as predict_table
from config.db import engine
import numpy as np
import tensorflow as tf
import os

prediction_router = APIRouter()

# Load model and labels once
model = tf.keras.models.load_model(
    "./routes/model/keras_model.h5",
    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
    compile=False
)

with open("./routes/model/labels.txt", "r") as f:
    class_names = f.readlines()

@prediction_router.post("/get-predict")
async def predict(img_test: UploadFile = File(...)):
    if img_test.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are allowed.")

    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(img_test.file).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
        image.save('temp_image.jpg')
        normalized_image_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].split(' ')[1].strip()
        confidence_score = str(round(prediction[0][index] * 100, 2)) + "%"

        with engine.connect() as conn:
            try:
                #count_query = select([func.count()]).select_from(predict_table)
                #count_result = conn.execute(count_query).scalar()  # Get the scalar result (count)

                data_insert = {
                    "tipo": class_name,
                    "image": imgur.upload_image(4, class_name, 'temp_image.jpg')
                }
                res = conn.execute(predict_table.insert().values(data_insert))
                os.remove('temp_image.jpg')
                conn.commit()
                print(res.lastrowid)
            except Exception as e:
                print(f"Error occurred: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return class_name
        # return JSONResponse(content={"class_name": class_name, "confidence_score": confidence_score})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
