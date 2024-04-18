import cv2


cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml') 

# Wczytujemy obraz z pliku
img = cv2.imread('C:/Users/Piotr G/Desktop/koty/Cat-Detect/Psy4000/archive/dataset/test_set/cats/cat.4010.jpg')


# Konwertujemy obraz do odcieni szarości
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# Wykrywamy twarze na obrazie
faces = cascade.detectMultiScale(img_gray, 1.5, 5) 
  
# Rysujemy prostokąt wokół wykrytych twarzy
for (x,y,w,h) in faces: 
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        
# Wyświetlamy obraz z zaznaczonymi twarzami
cv2.imshow('img',img) 

# Oczekiwanie na naciśnięcie klawisza klawiatury
cv2.waitKey(0)

# Zamykamy wszystkie otwarte okna
cv2.destroyAllWindows() 

