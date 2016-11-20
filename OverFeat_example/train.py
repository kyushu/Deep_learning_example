# -*- coding:utf-8 -*-

'''
USAGE:
    python train.py --conf conf/ct101.json
'''

from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from mtImageSearch.utils import Conf
from mtImageSearch.utils import dataset
import numpy as np
import argparse
import cPickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# Initialize Configuration Helper Object
conf = Conf(args["conf"])
# Load the Class Name
le = cPickle.loads(open(conf["label_encoder_path"]).read())



print("[INFO] gathering train/test splits...")
# Load the extracted feature from hdf5 file
db = h5py.File(conf["features_path"])
# split = train_data_set = total data * training_size ratio
split = int(db["image_ids"].shape[0] * conf["training_size"])
# Split data into training and testing dataset
(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
(testData, testLabels)   = (db["features"][split:], db["image_ids"][split:])

# use the label encoder to encode the training and testing labels
# trainLabels = [le.transform(l.split(":")[0]) for l in trainLabels]
# testLabels = [le.transform(l.split(":")[0]) for l in testLabels]
trainLabels = le.transform( np.array([l.split(":")[0] for l in trainLabels]) )
testLabels = le.transform( np.array([l.split(":")[0] for l in testLabels]) )



# Use GridSearch Cross Validation to find the best parameter
# we evaluate a Linear SVM for each value of C
# write the training result to file (results_path)
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

print("[INFO] evaluating...")
f = open(conf["results_path"], "w")
rank1 = 0
rank5 = 0

for (label, features) in zip(testLabels, testData):
    # predict the probability of each class label and grab the top-5 labels
    preds = model.predict_proba(np.atleast_2d(features))[0]
    # grab the top 5 of high probability
    preds = np.argsort(preds)[::-1][:5]

    # if the correct label is the first entry in the pedicted labels list,
    # increment the number of correct rank-1 predictions
    if label == preds[0]:
        rank1 += 1

    # if the correct label is in the top-5 predicted labels,
    # increment the number of correct rank-5 predictions
    if label in preds:
        rank5 += 1

# convert the accuracies to percents
rank1 = (rank1 / float(len(testLabels))) * 100
rank5 = (rank5 / float(len(testLabels))) * 100
# write rank result to the result file
f.write("rank-1: {:.2f}%\n".format(rank1))
f.write("rank-5: {:.2f}%\n\n".format(rank5))
# write training report to the result file
predictions = model.predict(testData)
f.write("{}\n".format(classification_report(testLabels, predictions, target_names=le.classes_)))
f.close()

# Dump classifier to file
print("[INFO] dumping classifier...")
f = open(conf["classifier_path"], "w")
f.write(cPickle.dumps(model))
f.close()

# close the database
db.close()
