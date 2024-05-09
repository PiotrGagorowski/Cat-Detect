import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Wczytanie pre-trenowanego modelu MobileNetV2
model = MobileNetV2(weights='imagenet')

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])

# Testowanie klasyfikatora

classify_image('C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/Datasets/00000001_000.jpg')
