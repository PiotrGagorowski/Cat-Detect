import cv2
import os

cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# Ścieżka do folderu zawierającego zdjęcia kotów
folder_path = 'C:/Users/kakit/Documents/GitHub/Cat-Detect/Datasets'

# Lista plików w folderze
image_files = os.listdir(folder_path)


# Funkcja do przetwarzania zdjęć
def process_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(img_gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
    cv2.imshow('Detektor', img)
    key = cv2.waitKey(0)
    if key == 27:  # Naciśnięcie ESC zakończy wyświetlanie zdjęć
        return False
    return True

# Wyświetlanie zdjęć po kolei
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    if not process_image(image_path):
        break

# Zamykanie okien po zakończeniu wyświetlania
cv2.destroyAllWindows()
