from keras.backend import reverse
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("1.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.LANCZOS)  # Updated here

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = str(round(prediction[0][index] * 100, 2)) + "%"

# Print prediction and confidence score
print(f"La predicción favorece a {class_name[2:]} con un porcentaje de {confidence_score}\n")
prediction.sort()
print("Otras conincidencias:")
pred_reverse = sorted(prediction[0], reverse=True)

for i in range(len(pred_reverse)):
    if pred_reverse[i] != prediction[0][len(prediction[0]) - 1]:
        print(f"{class_names[i][2:]} :{pred_reverse[i] * 100}")
