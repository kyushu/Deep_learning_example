# -*- coding:utf-8 -*-

'''
USAGE:
python index_features.py --conf conf/ct101.json
'''

from __future__ import print_function
import warnings
# suppress any FutureWarning from Theano
warnings.simplefilter(action="ignore", category=FutureWarning)

from mtImageSearch.overfeat import OverfeatExtractor
from mtImageSearch.indexer import OverfeatIndexer
from mtImageSearch.utils import Conf
from mtImageSearch.utils import dataset
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import argparse
import cPickle
import random

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and grab all image paths in the dataset
conf = Conf(args["conf"])
# paths.list_images() return a iterator object of file path name
# then use "list" to get all of file path name
imagePaths = list(paths.list_images(conf["dataset_path"]))

# random image paths for training and testing  spliting
random.seed(42)
random.shuffle(imagePaths)

# determine the set of possible class labels from the image dataset assuming
# that the images are in {directory}/{filename} structure and create the
# label encoder
print("[INFO] encoding labels...")
le = LabelEncoder()
# exampl of file path name is  "./dogs_and_cats/cat/cat.10897.jpg"
le.fit([p.split("/")[-2] for p in imagePaths])


#
# Initialize the Overfeat extractor and the Overfeat indexer
#
print("[INFO] initializing network...")
# Initialize the Overfeat neural network and user third from last layer as Featrue-Extraction layer
oe = OverfeatExtractor(conf["overfeat_layer_num"])

# Initialize the Overfeat indexer used for store fetrue with corresponding name(class)
oi = OverfeatIndexer(conf["features_path"], estNumImages=len(imagePaths))
print("[INFO] starting feature extraction...")

#
# Start Featrue-Extraction by looping image paths in batches
#
for (i, paths) in enumerate(dataset.chunk(imagePaths, conf["overfeat_batch_size"])):
    # load the set of images from disk and describe them
    (labels, images) = dataset.build_batch(paths, conf["overfeat_fixed_size"])
    features = oe.describe(images)
    # loop over each set of (label, vector) pair and add them to the indexer
    for (label, vector) in zip(labels, features):
        oi.add(label, vector)

    # check to see if progress should be displayed
    if i > 0:
        oi._debug("processed {} images".format((i + 1) * conf["overfeat_batch_size"],
            msgType="[PROGRESS]"))


# finish the indexing process (store feature to conf["features_path"])
oi.finish()

# dump the label encoder to file
print("[INFO] dumping labels to file...")
f = open(conf["label_encoder_path"], "w")
f.write(cPickle.dumps(le))
f.close()

