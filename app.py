from DepthwiseConv2D import CustomDepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost:3000",  # Permite solicitudes desde este origen
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los encabezados
)

@app.post("/get-predict")
async def predict(img_test: UploadFile = File(...)):
    model = tf.keras.models.load_model(
        f"./model/keras_model.h5",
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
        compile=False
    )

    # Cargar las etiquetas
    with open(f"./model/labels.txt", "r") as f:
        class_names = f.readlines()

    # Crear el array con la forma correcta
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Cargar la imagen
    image = Image.open(img_test.file).convert("RGB")

    # Redimensionar la imagen
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # Convertir la imagen a un array de numpy
    image_array = np.asarray(image)

    # Normalizar la imagen
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Cargar la imagen en el array
    data[0] = normalized_image_array

    print("\n[!] Esperando predicción")
    # Predecir con el modelo
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].split(' ')[1].strip()
    confidence_score = str(round(prediction[0][index] * 100, 2)) + "%"

    print(f"La predicción favorece a {class_name} con un porcentaje de {confidence_score}\n")

    return class_name
