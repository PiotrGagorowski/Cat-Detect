import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,  precision_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Wczytanie modelu
model = load_model("fine_tuned_model.h5")

# Ścieżka do folderu z testowymi obrazami
test_images_dir = 'C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/test'

# Funkcja do wczytywania i przetwarzania obrazów
def load_and_process_image(image_path, target_size=(224, 224)):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  # Normalizacja
        return img_array
    except Exception as e:
        print(f"Failed to load image: {image_path}, Error: {e}")
        return None

# Lista testowych obrazów
cat_images = [os.path.join(test_images_dir, 'koty', f) for f in os.listdir(os.path.join(test_images_dir, 'koty')) if os.path.isfile(os.path.join(test_images_dir, 'koty', f))]
dog_images = [os.path.join(test_images_dir, 'psy', f) for f in os.listdir(os.path.join(test_images_dir, 'psy')) if os.path.isfile(os.path.join(test_images_dir, 'psy', f))]
test_images = cat_images + dog_images


# Losowo wybierane kilka obrazów z listy
selected_images = np.random.choice(test_images, min(5, len(test_images)), replace=True)

# Wyświetlenie obrazów i predykcji
predictions = []
for image_path in selected_images:
    img = load_and_process_image(image_path)
    if img is not None:
        prediction = model.predict(img)
        predictions.append(prediction)
        predicted_class = "Kot" if prediction < 0.5 else "Pies" 
        plt.imshow(image.load_img(image_path))
        plt.title(f"Przewidywana klasa: {predicted_class}")
        plt.axis('off')
        plt.show()

# Iteruja po całym zbiorze testowym w celu wyliczenia mertryki recall
for image_path in test_images:  
     img = load_and_process_image(image_path)
     if img is not None:
         prediction = model.predict(img)
         predictions.append(prediction)

# Wyliczenie precision i recall
true_labels = [0] * len(cat_images) + [1] * len(dog_images)  # 0 = kot, 1 = psies

# Usunięcie nadmiarowych przewidywanych etykiet
predicted_labels = [0 if pred < 0.5 else 1 for pred in predictions][:len(true_labels)]

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

print(f"Matryka precision: {precision:.3f}")
print(f"Metryka recall: {recall:.3f}")

