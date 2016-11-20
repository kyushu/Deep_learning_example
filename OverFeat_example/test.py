# -*- coding:utf-8 -*-

'''
USAGE:
    python test.py --conf conf/ct101.json
'''

from __future__ import print_function
from mtImageSearch.utils import Conf
import numpy as np
import argparse
import cPickle
import h5py
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

print("[INFO] loading model...")
conf = Conf(args["conf"])
# sklearn.preprocessing.LabelEncoder
le = cPickle.loads(open(conf["label_encoder_path"]).read())
# Linear SVM classifier
model = cPickle.loads(open(conf["classifier_path"]).read())


print("[INFO] gathering test data...")
db = h5py.File(conf["features_path"])
split = int(db["image_ids"].shape[0] * conf["training_size"])
(testData, testLabels) = (db["features"][split:], db["image_ids"][split:])

# pick up 10 images from testing data for test
for i in np.random.choice(np.arange(0, len(testData)), size=(10,), replace=False):
    (trueLabel, filename) = testLabels[i].split(":")
    vector = testData[i]

    # prediction by Feature Vector
    prediction = model.predict(np.atleast_2d(vector))[0]
    prediction = le.inverse_transform(prediction)
    print("[INFO] predicted: {}, actual:{}".format(prediction, trueLabel))

    # load the image for display
    path = "{}/{}/{}".format(conf["dataset_path"], trueLabel, filename)
    image = cv2.imread(path)
    cv2.putText(image, prediction, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

db.close()

