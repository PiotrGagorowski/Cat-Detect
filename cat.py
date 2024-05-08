import cv2
import os
import time
import threading

# Wczytanie wytrenowanego modelu kaskadowego
cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# Ścieżka do folderu zawierającego zdjęcia kotów
folder_path = 'C:/Users/Piotr G/Cat-Detect/Datasets'

# Lista plików w folderze
image_files = os.listdir(folder_path)

# Funkcja do przetwarzania obrazów
def process_image(image_path):
    img = cv2.imread(image_path)
    max_dimension = 600
    scale_ratio = max_dimension / max(img.shape[0], img.shape[1])
    img_resized = cv2.resize(img, (int(img.shape[1] * scale_ratio), int(img.shape[0] * scale_ratio)))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Detekcja kotów za pomocą kaskady Haar
    faces = cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_resized, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img_resized, 'Kot', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    # Wyświetlenie obrazu
    cv2.imshow('Detektor', img_resized)
    cv2.waitKey(5000)  # Wyświetlanie przez 5 sekund

# Funkcja do trenowania modelu w tle
def train_model_background():
    train_model()

# Funkcja do trenowania modelu
def train_model():
    # Tu umieść kod trenowania modelu na danych treningowych
    pass

# Wątek do trenowania modelu w tle
train_thread = threading.Thread(target=train_model_background)
train_thread.daemon = True  # Wątek zostanie zakończony po zakończeniu programu
train_thread.start()

# Wyświetlanie obrazów i wyników detekcji
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    process_image(image_path)

# Zamykanie okien po zakończeniu wyświetlania
cv2.destroyAllWindows()
