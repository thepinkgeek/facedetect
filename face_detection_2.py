import cv2
import sys
import traceback
import numpy
import os
from neural_network import MyNeuralNetwork

# Get user supplied values
cascPath = sys.argv[1]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


data_file = open("data.out", "w")
images = []

total_num_inputs = 180 * 256
total_num_outputs = 4 # Note: still stubbed. corresponds to total number of people to recognize
total_num_hidden_layers = 5 # Note: still needs to be experimented.
neuralNetwork = MyNeuralNetwork(total_num_inputs, total_num_hidden_layers, total_num_outputs)

for dirname, dirnames, filenames in os.walk('Downloads/testpicsfolder/'):
    # print path to all filenames.
    for filename in filenames:
        images.append(os.path.join(dirname, filename))


for i in range(len(images)):
    imagePath = images[i]

    print ("processing %s" % imagePath)
    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )

    print "Found {0} faces!".format(len(faces))
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        crop_img = image[y : y + h, x : x + w]
        hsv_base = cv2.cvtColor( crop_img, cv2.COLOR_BGR2HSV );
        cv2.imshow("face", crop_img)
        cv2.waitKey(0)

        face_name = raw_input("whose face is this?")

        if face_name != "-1":
            face_file_name = "results/face_" + repr(i) + ".jpg"
            cv2.imwrite(face_file_name, crop_img)
            data_file.write("%s %s" % (face_file_name, face_name))