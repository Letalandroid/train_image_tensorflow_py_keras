import requests, json
import pyimgur

im = pyimgur.Imgur('fcf38cce9d8cc9b')

def upload_image(id, type, image_path):
    uploaded_image = im.upload_image(image_path, title=f"{type} - #{id}")
    return uploaded_image.link