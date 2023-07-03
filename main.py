import cv2
import matplotlib.pyplot as plt

imgPath = 'i.jpg'
img = cv2.imread(imgPath)

#convert to grayscale
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#load the haar cascade classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    g_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

#drawing a line around the detected faces
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

#convert cv image to RGB in order to display to user
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.imshow(rgb_img)
plt.show()
# plt.axis('off')
