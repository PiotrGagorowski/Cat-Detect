import cv2
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions

# Wczytanie pre-trenowanego modelu MobileNetV3Small
model = MobileNetV3Small(weights='imagenet')

# Wczytanie klasyfikatora Haar Cascade do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

def classify_image(image_path):
    img = cv2.imread(image_path)
    
    # Konwersja koloru obrazu z BGR na RGB (OpenCV wczytuje obrazy w formacie BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Wykrywanie twarzy na obrazie
    faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        # Rysowanie prostokąta wokół wykrytej twarzy
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Konwersja obrazu do rozmiaru 224x224 pikseli, wymaganego przez MobileNetV3
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Przetwarzanie obrazu tak, aby pasował do wejścia modelu MobileNetV3
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)

    # Przetwarzanie obrazu i predykcja
    preds = model.predict(x)
    predictions = decode_predictions(preds, top=1)[0]

    # Sprawdzenie, czy przynajmniej jedna twarz została wykryta
    if len(faces) > 0:
        # Pobranie predykcji dla pierwszego obiektu
        predicted_class = predictions[0][1]  # Pobierz nazwę klasy (rasę kota)
        probability = predictions[0][2]  # Pobierz prawdopodobieństwo

        # Umieszczenie informacji o predykcji nad ostatnim prostokątem (jeśli istnieje)
        text = f'Predicted class: {predicted_class}, Probability: {probability:.2f}'
        cv2.putText(img, text, (faces[-1][0], faces[-1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    else: 
        predicted_class = "Nie wykryto kota"
        probability = 0
    
    print('Predicted class:', predicted_class)
    print('Probability:', probability)

    # Wyświetlanie obrazu z zaznaczonym kwadratem na twarzy kota
    cv2.imshow('Image with Detected Cat Face', img)
    key = cv2.waitKey(0)
    if key == 27:  # Naciśnięcie klawisza ESC zamyka okno
        cv2.destroyAllWindows()
        return False
    return True

# Przetwarzanie wszystkich obrazów z folderu
folder_path = 'C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/Datasets'
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

for image_file in image_files:
    if not classify_image(image_file):
        break




