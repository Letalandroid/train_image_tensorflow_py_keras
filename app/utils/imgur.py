import requests, json
import pyimgur

im = pyimgur.Imgur('de498b815e1ac5d')

def upload_image(id, type, image_path):
    uploaded_image = im.upload_image(image_path, title=f"{type} - #{id}")
    return uploaded_image.link