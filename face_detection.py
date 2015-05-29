import cv2
import numpy as np
import os
from sklearn.externals import joblib

CASCPATH = "data/haarcascades/haarcascade_frontalface_alt.xml"

TRAINING_SET_FOLDER = "training_data/"
FACE_DETECTION_RESULTS_DIR = ""
FACE_DETECTION_PREFIX = "face_"
EXTENSION = ".jpg"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(CASCPATH)

clf = joblib.load("svm.pkl")
pca_obj = joblib.load("pca.pkl")

classes = {}
class_labels = []
for line in open("classes.in", "r"):
    class_number, class_name = line.split(" ")
    classes[class_number] = class_name
    class_labels.append(class_name)

images = []
for dirname, dirnames, filenames in os.walk('test_data'):
    # print path to all filenames.
    for filename in filenames:
        images.append(os.path.join(dirname, filename))


for i in range(len(images)):
    imagePath = images[i]

    print ("processing %s" % imagePath)

    name, extension = os.path.splitext(imagePath)
    print ("processing %s" % (imagePath))

    if extension != ".jpg" and extension != ".JPG":
        continue

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            )

    print "Found {0} faces!".format(len(faces))
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        crop_img = gray[y : y + h, x : x + w]
        resized = np.array(cv2.resize(crop_img, (100, 100))).flatten()

        x_val = list()
        x_val.append(resized)
        x_test_arr = np.array(x_val)

        x_test_arr = pca_obj.transform(x_test_arr)

        print class_labels[clf.predict(x_test_arr)]
