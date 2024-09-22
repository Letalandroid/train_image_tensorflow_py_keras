from DepthwiseConv2D import CustomDepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

options = ["animales", "deportes", "expresiones", "lugares turisticos"]
print("""
> Por favor elija una de las opciones:
 0.  Animales
 1.  Deportes
 2.  Expresiones
 3.  Lugares turísticos
      """)
print(">> ", end="")

try:
    opt = int(input(""))
except ValueError:
    print("Eso no es un número entero válido.")
    exit()

if not (opt  >= 0 and opt <= 3 or opt):
    print("Opción no válida")
    exit()

print()
opt_select = options[opt].lower().replace(' ', '_')
print(opt_select)

# Cargar el modelo con la clase personalizada
model = tf.keras.models.load_model(
    f"./models/{opt_select}/keras_model.h5",
    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
    compile=False
)


# Cargar las etiquetas
with open(f"./models/{opt_select}/labels.txt", "r") as f:
    class_names = f.readlines()

# Crear el array con la forma correcta
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Cargar la imagen
image = Image.open(f"images/{opt_select}.jpg").convert("RGB")

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

# Imprimir la predicción
print(f"La predicción favorece a {class_name} con un porcentaje de {confidence_score}\n")

# Obtener las probabilidades de predicción
pred_probs = prediction[0]

# Imprimir otras coincidencias
print("Otras coincidencias:")
pred_indices = np.argsort(pred_probs)[::-1]  # Obtener índices ordenados de mayor a menor

# Omitir el índice de la clase más probable (la primera)
for i in range(1, len(pred_indices)):  # Comenzar desde 1 para omitir la mejor predicción
    class_label = class_names[pred_indices[i]].split(' ')[1].strip()
    print(f"{class_label} : {pred_probs[pred_indices[i]] * 100:.2f}%")