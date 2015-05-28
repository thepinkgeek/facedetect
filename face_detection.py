import cv2
import sys
import traceback
import numpy
import os
from neural_network import MyNeuralNetwork

CASCPATH = "data/haarcascades/haarcascade_frontalface_alt.xml"

TRAINING_SET_FOLDER = "training_data/"
FACE_DETECTION_RESULTS_DIR = ""
FACE_DETECTION_PREFIX = "face_"
EXTENSION = ".jpg"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(CASCPATH)

classes = {}
class_labels = []
for line in open("classes.in", "r"):
    class_number, class_name = line.split(" ")
    classes[class_number] = class_name
    class_labels.append(class_name)


total_num_inputs = 180 * 256
total_num_outputs = len(class_labels) # Note: still stubbed. corresponds to total number of people to recognize
total_num_hidden_layers = 100 # Note: still needs to be experimented.
print "loading neural network"
neuralNetwork = MyNeuralNetwork(total_num_inputs, total_num_hidden_layers, total_num_outputs)
print "loading done."

data_file = open("data.out", "w")
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
        crop_img = image[y : y + h, x : x + w]
        hsv_base = cv2.cvtColor( crop_img, cv2.COLOR_BGR2HSV )
        hist = cv2.calcHist([hsv_base], [0, 1], None, [180, 256], [0, 180, 0, 256])

        print class_labels[neuralNetwork.activate(hist.ravel())]
