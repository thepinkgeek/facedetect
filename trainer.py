import cv2
import traceback
import os
from neural_network import MyNeuralNetwork

CASCPATH = "data/haarcascades/haarcascade_frontalface_alt.xml"

TRAINING_SET_FOLDER = "training_data/"
FACE_DETECTION_RESULTS_DIR = "results"
FACE_DETECTION_PREFIX = "face_"
EXTENSION = ".jpg"

classes = {}
class_labels = []
for line in open("classes.in", "r"):
    class_number, class_name = line.split(" ")
    classes[class_number] = class_name
    class_labels.append(class_name)


total_num_inputs = 180 * 256
total_num_outputs = len(class_labels) # Note: still stubbed. corresponds to total number of people to recognize
total_num_hidden_layers = 100 # Note: still needs to be experimented.
neuralNetwork = MyNeuralNetwork(total_num_inputs, total_num_hidden_layers, total_num_outputs)

# Get user supplied values
cascPath = CASCPATH
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

"""
data_file = open("data.out", "w")
images = []

for dirname, dirnames, filenames in os.walk(TRAINING_SET_FOLDER):
    # print path to all filenames.
    for filename in filenames:
        images.append(os.path.join(dirname, filename))


for i in range(len(images)):
    imagePath = images[i]

    print ("processing %s" % imagePath)

    name, extension = os.path.splitext(imagePath)

    print name
    print extension
    if extension != ".jpg" and extension != ".JPG":
        continue

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print "finding faces..."
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            )

    print ("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        crop_img = image[y : y + h, x : x + w]
        cv2.imshow("face", crop_img)
        cv2.waitKey(0)

        face_name = raw_input("whose face is this?")

        if face_name != "-1":
            face_file_name = FACE_DETECTION_RESULTS_DIR + "/" + FACE_DETECTION_PREFIX + repr(i) + EXTENSION
            cv2.imwrite(face_file_name, crop_img)
            data_file.write("%s %s\n" % (face_file_name, face_name))

data_file.close()
"""

x_train = []
y_train = []

for line in open("data.out", "r"):
        fileName, face_id = line.split(" ")
        imagePath = fileName
        name, extension = os.path.splitext(imagePath)

        print ("processing %s" % (imagePath))
        print extension

        if extension != ".jpg" and extension != ".JPG":
            continue

        try:
            image = cv2.imread(imagePath)
            hsv_base = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_base], [0, 1], None, [180, 256], [0, 180, 0, 256])

            x_train.append(hist.ravel())
            y_train.append(int(face_id) - 1)

        except:
            traceback.print_exc()
            continue

neuralNetwork.train(x_train, y_train, len(x_train[0]), class_labels)
