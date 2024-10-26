from fastapi import APIRouter
from PIL import Image, ImageOps
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.DepthwiseConv2D import CustomDepthwiseConv2D
from utils import imgur
from sklearn.metrics import classification_report
from model.predict import predict as predict_table
from sqlalchemy import func, select
from config.db import engine
import numpy as np
import tensorflow as tf
import os
from time import time

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

        start_predict = time()
        prediction = model.predict(data)
        index = np.argmax(prediction)
        end_predict = time()
        print(f'Tiempo de prediccion: {end_predict - start_predict:.2f} segundos')
        class_name = class_names[index].split(' ')[1].strip()
        confidence_score = str(round(prediction[0][index] * 100, 2)) + "%"

        # Imprimir la predicción
        print(f"La predicción favorece a {class_name} con un porcentaje de {confidence_score}\n")

        # Obtener las probabilidades de predicción
        pred_probs = prediction[0]

        # Imprimir otras coincidencias
        print("Otras coincidencias:")
        pred_indices = np.argsort(pred_probs)[::-1]  # Obtener índices ordenados de mayor a menor
        others_predicts = []

        # Omitir el índice de la clase más probable (la primera)
        for i in range(1, len(pred_indices)):  # Comenzar desde 1 para omitir la mejor predicción
            class_label = class_names[pred_indices[i]].split(' ')[1].strip()
            others_predicts.append({
                'class_label': class_label,
                'pred_probs': float(pred_probs[pred_indices[i]] * 100)
            })

            print(f"{class_label} : {pred_probs[pred_indices[i]] * 100:.2f}%")

        with engine.connect() as conn:
            try:
                # count_query = select([func.count()]).select_from(predict_table)
                # count_result = conn.execute(count_query).scalar()  # Get the scalar result (count)

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

        return JSONResponse(content={"class_name": class_name, "confidence_score": confidence_score, 'others': others_predicts})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
