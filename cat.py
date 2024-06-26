import cv2
import os

cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cascade_dog = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Ścieżka do folderu zawierającego zdjęcia kotów
folder_path = 'C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/Datasets'

# Lista plików w folderze
image_files = os.listdir(folder_path)


# Funkcja do przetwarzania zdjęć
def process_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Wykrywanie kotów
    cats = cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in cats:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img, 'Kot', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        
    # Wykrywanie psów
    dogs = cascade_dog.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in dogs:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, 'Pies', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)
        
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
