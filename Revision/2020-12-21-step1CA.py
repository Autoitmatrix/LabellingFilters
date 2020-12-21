# -*- coding: utf-8 -*-
"""Step1 Collecting Activations
"""

# Imports
import logging
import numpy as np
import os
import pandas as pd
import pickle
#import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
#from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from warnings import warn

"""## Inputs"""

# Input
network = "VGG16"
images_dir = r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Data" # "/home/ws/ns1888/FilterLabel/ILSVRC2012_img_train"
save_dir = r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Pickles" # "/home/ws/ns1888/FilterLabel/codes/resultstest4"
seed_rn = 123
method = "average"  # either "average" or "max" for merging image activations
aggregation = "filterfirst"  # "imagesfirst"
split = 0.2  # data taken for validation

# debugging and testing input
loglevel = "INFO"
res_classes = None  # restrict number of classes

"""## Preparations"""

# create ID and paths for files
now = datetime.utcnow() + timedelta(hours=2)
runid = now.strftime("%Y-%m-%d-%H%M")
if not save_dir:
    save_dir = os.getcwd()
save_vars = save_dir + "/" + runid + "-" + network + "-CA-vars"
save_vars_names = save_dir + "/" + runid + "-" + network + "-CA-vars-names"
save_vars_list = []
save_vars_info = save_vars + "-info.txt"

# create info file
with open(save_vars_info, "a") as q:
    q.write("Vars Info CA\n----------\n\n")
    q.write("This file gives an overview of the stored variables in pickle file " + save_vars + "\n")
    q.write("To load them use e.g.:\n")
    q.write("with open(\"" + save_vars_names + "\",\"rb\") as r:\n")
    q.write("  var_names=pickle.load(r)\n")
    q.write("with open(\"" + save_vars + "\",\"rb\") as p:\n")
    q.write("  for var in var_names:\n")
    q.write("    globals()[var]=pickle.load(p)\n\n")
    q.write("stored variables are:\n")

# logging
logging.captureWarnings(True)
logger = logging.getLogger('logger1')
logger.setLevel(loglevel)
if (logger.hasHandlers()):
    logger.handlers.clear()
info_handler = logging.FileHandler(save_dir + "/" + runid + '-CA-info.log')
info_handler.setLevel(logging.INFO)
infoformat = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d.%m.%Y %H:%M:%S')
info_handler.setFormatter(infoformat)
logger.addHandler(info_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(infoformat)
logger.addHandler(stream_handler)
warnings_logger = logging.getLogger("py.warnings")
warnings_handler = logging.FileHandler(save_dir + "/" + runid + '-CA-warnings.log')
warnings_logger.addHandler(warnings_handler)
warnings_stream_handler = logging.StreamHandler()
warnings_logger.addHandler(warnings_stream_handler)
if logger.getEffectiveLevel() == 10:
    debug_handler = logging.FileHandler(save_dir + "/" + runid + '-CA-debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debugformat = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                                    datefmt='%d.%m.%Y %H:%M:%S')
    debug_handler.setFormatter(debugformat)
    logger.addHandler(debug_handler)
logger.propagate = False

# images
if "ception" in network:  # Inception networks have other image resolution as input
    img_rows = 299
    img_cols = 299
else:
    img_rows = 224
    img_cols = 224

# network
network_mod = {"VGG16": "vgg16", "VGG19": "vgg19", "InceptionV3": "inception_v3", "ResNet50": "resnet",
               "ResNet101": "resnet", "ResNet152": "resnet", "ResNet50V2": "resnet_v2", "ResNet101V2": "resnet_v2",
               "ResNet152V2": "resnet_v2", "InceptionResNetV2": "inception_resnet_v2"}
logger.info("importing network")
cnn = getattr(keras.applications, network)()  # import network
layers = []  # interesting layers, omitting pooling layer and fc
for i, layer in enumerate(cnn.layers):
    if 'conv' not in layer.name:
        continue
    layers.append(i)
logger.debug("conv layers are at positions " + str(layers))

"""## functions"""

def preprocess_images_from_dir(images_dir, img_rows, img_cols, network, seed_rn, split, numclasses=None):
    '''
    prepares the images in images_dir as input for the networks.

    input:
    - images_dir: path to the directory with image data, the folders in it should be named after the image class of the images they contain.
    - img_rows: height of images
    - img_cols: width of images
    - network: name of the cnn
    - seed_rn: random seed
    - split: fraction used for test set, 1-split is used for train set
    - numclasses: restriction of the number of classes

    returns:
    - images_train: train set
    - images_test: test set

    remark:
    uses tensorflow.keras as keras and ImageDataGenerator from tensorflow.keras.preprocessing.image
    '''
    logger.info("loading images in " + images_dir)
    network_module = getattr(keras.applications, network_mod[network])
    preprocess_input = getattr(network_module, "preprocess_input")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=split)
    folders = os.listdir(images_dir)
    numfolders = len(folders)
    if numclasses:
        numfolders = numclasses
    files = []
    target = []
    for folder in folders[0:numfolders]:
        imagepaths = os.listdir(images_dir + "/" + folder)
        for imagepath in imagepaths:
            files.append(images_dir + "/" + folder + "/" + imagepath)
            target.append(folder)
    df_images = pd.DataFrame({"imagepath": files, "wnid": target})
    df_images = df_images.sample(frac=1, random_state=seed_rn).reset_index(drop=True)  # shuffle input data
    logger.info("Preparing Train Set")
    images_train = datagen.flow_from_dataframe(df_images, x_col='imagepath', y_col='wnid', batch_size=1,
                                               target_size=(img_rows, img_cols), shuffle=False, seed=seed_rn,
                                               subset="training")
    logger.info("Preparing Test Set")
    images_test = datagen.flow_from_dataframe(df_images, x_col='imagepath', y_col='wnid', batch_size=1,
                                              target_size=(img_rows, img_cols), shuffle=False, seed=seed_rn,
                                              subset="validation")
    logger.info("images ready")
    return images_train, images_test


def get_activations(prep_images, method, cnn, layers, classnum, maxclass):
    '''
    computes the average or max activations of prep_images for specific layers of a certain network

    input:
    - prep_images: image data
    - method (string): either "max" or "average"
    - cnn: Keras model of the cnn
    - layers (list): contains the indeces of the layers to fetch the activations from

    returns:
    - prep_act: prepared activations
    '''
    logger.debug("get_activations")
    prep_act = 0
    if method not in ["max", "average"]:
        warn("no method for allocation of activations selected. Default is average")
        method = "average"

    logger.debug("get activations of images, selected combination is " + method)
    outputs = [cnn.layers[i].output for i in layers]
    outputs_model = keras.Model([cnn.input], outputs)
    num_pics = len(prep_images)
    logger.debug("number images of this class" + str(num_pics))
    for step in range(num_pics):
        logger.info("collect, scale and combine activations for image " + str(step + 1) + " of " + str(
            num_pics) + " (class " + str(classnum + 1) + " of " + str(maxclass) + ")")
        out_pic_new = scale_out(outputs_model.predict(prep_images[step]))
        if step == 0:
            prep_act = out_pic_new.copy()
        else:
            for count, layer in enumerate(prep_act):
                if method == "average":
                    layer = out_pic_new[count] * (1 / (step + 1)) + out_pic_old[count] * (1 / (step + 1))
                else:  # max
                    layer = np.maximum(out_pic_new[count], out_pic_old[count])
        out_pic_old = prep_act.copy()
    logger.info("activations collected for class " + str(classnum + 1) + " of " + str(maxclass))
    return prep_act


def scale_out(a):
    '''
    scales the activations of each layer to a range of [0,1]

    input:
    - a: raw activations

    returns:
    - b: scaled activations
    '''
    logger.debug("scale_out start")
    b = a.copy()
    logger.debug("scale_out")
    scaler = MinMaxScaler(copy=False)  # inplace operation
    for layer in b:
        layer = layer.reshape(-1, 1)
        scaler.fit_transform(layer)
    logger.debug("scale_out end")
    return b


def act_fil(acts):
    '''
    averages the activations of each feature map to get one single number that is assigned to the proceeding filter

    input:
    - acts: (scaled) activations

    return:
    - fil_act: "averaged filter activations"
    '''
    logger.debug("averaging activations per filter")
    fil_act = [None] * len(acts)
    for i, layer in enumerate(acts):
        numfil = layer.shape[3]
        fil_act[i] = np.zeros(numfil)
        for fil in range(numfil):
            fil_act[i][fil] = np.mean(layer[:, :, :, fil])
    return fil_act


""" #Execution """

logger.info("Selected network is " + network)

# preprocess images
train, test = preprocess_images_from_dir(images_dir, img_rows, img_cols, network, seed_rn, split, res_classes)

# extract class labels
labels_id_inv = train.class_indices  # get dict with mapping of labels
labels_id = {y: x for x, y in labels_id_inv.items()}  # invert dict
with open(save_vars, "wb") as p:
    pickle.dump(train.filenames, p)
    pickle.dump(test.filenames, p)
    pickle.dump(labels_id, p)
save_vars_list.append("train_filenames")
save_vars_list.append("test_filenames")
save_vars_list.append("labels_id")
with open(save_vars_info, "a") as q:
    q.write("- train_filenames: list wiht filenames of train set\n")
    q.write("- test_filenames: list with filenames of test set\n")
    q.write("- labels_id: dictionary of numerique train labels and corresponding folder name (class)\n")
logger.info("class labels added to " + save_vars)
num_classes = len(labels_id)

fil_act = []
for i in tqdm(range(num_classes)):
    logger.info("class: " + str(labels_id[i]) + " (labelnum:" + str(i) + ")")
    class_images = [train[j] for j in range(train.n) if train.labels[j] == i]
    logger.debug("number of images in class " + str(i) + ": " + str(len(class_images)))
    if not class_images:
        warn("no images for class " + str(labels_id[i]) + " (labelnum:" + str(i) + ") in train set")
        fil_act.append([])
        continue
    out = get_activations(class_images, method, cnn, layers, i, num_classes)
    fil_act.append(act_fil(out))
with open(save_vars, "ab") as p:
    pickle.dump(fil_act, p)
save_vars_list.append("fil_act")
with open(save_vars_names, "wb") as r:
    pickle.dump(save_vars_list, r)
with open(save_vars_info, "a") as q:
    q.write("- fil_act: activations before threshold\n")
logger.info("activations added to " + save_vars)
logger.info("---------------Run succesful-----------------")
