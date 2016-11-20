# -*- coding: utf-8 -*-

'''
USAGE:
    python test_network.py --model output/cifar10_shallownet.hdf5 --test-images test_images
    python test_network.py --model output/cifar10_lenet.hdf5 --test-images test_images
'''

from __future__ import print_function
from keras.models import load_model
from keras.datasets import cifar10
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model file")
ap.add_argument("-t", "--test-images", required=True, help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of mini-batches passed to network")
args = vars(ap.parse_args())

# initialize the ground-truth labels for the CIFAR-10 dataset
gtLabels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
    "ship", "truck"]

print("[INFO] loading network architecture and weights...")
# 載入之前已訓練過的 network(shallowNet, LeNet, KarpathyNet, MiniVGGNet)
model = load_model(args["model"])


print("[INFO] sampling CIFAR-10...")
# ((X_train, y_train), (X_test, y_test)) = cifar10.load_data(), return 2 tuples
# X_train, X_test (Data): uint8 array of RGB image data with shape (nb_samples, 3, 32, 32).
# y_train, y_test (Label): uint8 array of category labels (integers in range 0-9) with shape (nb_samples,).
(testData, testLabels) = cifar10.load_data()[1]
testData = testData.astype("float") / 255.0
np.random.seed(42)
# random pick 15 indexes
idxs = np.random.choice(testData.shape[0], size=(10,), replace=False)
print("testData.shap: {}".format(testData.shape))
# print("idxs:{}".format(idxs))
# pick test sample by idxs
(testData, testLabels) = (testData[idxs], testLabels[idxs])
# print("testLabels:{}".format(testLabels))
testLabels = testLabels.flatten()
# print("testLabels:{}".format(testLabels))

print("[INFO] prediction on testing data...")
probs = model.predict(testData, batch_size=args["batch_size"])
predictions = probs.argmax(axis=1)

# prediction on the sample of testing data
for (i, prediction) in enumerate(predictions):
    # (R, G, B) = testData[i] # the pixel order of testData is RGB
    (R, G, B) = cv2.split(testData[i])

    # we need to convert pixel order to CV2's order
    image = np.dstack((B, G, R))

    # resize test image to large size
    image = imutils.resize(image, width=128, inter=cv2.INTER_CUBIC)

    print("[INFO] predicted: {}, actual: {}".format(gtLabels[prediction], gtLabels[testLabels[i]]))

    cv2.imshow("Image", image)
    cv2.waitKey(0)


# Close all open windows
cv2.destroyAllWindows()

print("[INFO] testing on images NOT part of CIFAR-10")

for imagePath in paths.list_images(args["test_images"]):
    print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1:]))
    image = cv2.imread(imagePath)

    # revrse Color channel from BGR(OpenCV) to RGB(Keras) then resize to 32x32
    kerasImage = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (32, 32)).astype("float") / 255.0

    # add an extra dimension to the image so we can pass it through the network,
    # then make a prediction on the image (normally we would make predictions on
    # an *array* of images instead one at a time)
    kerasImage = kerasImage[np.newaxis, ...]
    probs = model.predict(kerasImage, batch_size=args["batch_size"])
    prediction = probs.argmax(axis=1)[0]

    cv2.putText(image, gtLabels[prediction], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

