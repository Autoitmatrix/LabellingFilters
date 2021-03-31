"""# Step3 Evaluation"""

"""## Imports"""
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from collections import Counter
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from warnings import warn, simplefilter

"""## Inputs"""

k_q = [1, 5, 25]
filTags_file = "/content/Results/2021-01-01-0000-TF-VGG16-vars"# file from step 2
save_dir = "/content/Results"
images_dir = "/content/ILSVRC2012_img_train"  # directory with the imagefolder(s)
split = 0.2
seed_rn = 123
time_delta = 0

"""## Preparation"""

# create ID for files
network = pickle_file.split("-")[-2]
now = datetime.utcnow() + timedelta(hours=time_delta)
runid = now.strftime("%Y-%m-%d-%H%M")+"-E"
if not save_dir:
    save_dir = os.getcwd()
save_vars = save_dir + "/" + runid + "-" + network + "-vars"
save_vars_names = save_vars + "-names"
save_vars_list = []
save_vars_info = save_vars + "-info.txt"
with open(save_vars_info, "a") as txt:
    txt.write("Vars Info E\n----------\n\n")
    txt.write("This file gives an overview of the stored variables in pickle file "
            + save_vars + "\n")
    txt.write("To load them use e.g.:\n")
    txt.write("with open(\"" + save_vars_names + "\",\"rb\") as r:\n")
    txt.write("  var_names=pickle.load(r)\n")
    txt.write("with open(\"" + save_vars + "\",\"rb\") as p:\n")
    txt.write("  for var in var_names:\n")
    txt.write("    globals()[var]=pickle.load(p)\n\n")
    txt.write("stored variables are:\n")

# logging
logging.captureWarnings(True)
logger = logging.getLogger('logger1')
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()

info_handler = logging.FileHandler(save_dir + "/" + runid + '-info.log')
infoformat = logging.Formatter('%(asctime)s - %(message)s', 
                               datefmt='%d.%m.%Y %H:%M:%S')
info_handler.setFormatter(infoformat)
logger.addHandler(info_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(infoformat)
logger.addHandler(stream_handler)

os.environ["PYTHONWARNINGS"] = "always"
warnings_logger = logging.getLogger("py.warnings")
warnings_handler = logging.FileHandler(save_dir + "/" + runid + 
                                       '-warnings.log')
warnings_logger.addHandler(warnings_handler)
logger.propagate = False

# get filTags (from step2)
with open(filTags_file, "rb") as p:
  filTags_k=pickle.load(p)
  filTags_q=pickle.load(p)
logger.info("filTags loaded: filTags_k, filTags_q")

# images
if "ception" in network:  # Inception networks have other image resolution as input
    img_rows = 299
    img_cols = 299
else:
    img_rows = 224
    img_cols = 224

# network
network_mod = {"VGG16": "vgg16",
               "VGG19": "vgg19",
               "InceptionV3": "inception_v3",
               "ResNet50": "resnet",
               "ResNet101": "resnet",
               "ResNet152": "resnet",
               "ResNet50V2": "resnet_v2",
               "ResNet101V2": "resnet_v2",
               "ResNet152V2": "resnet_v2",
               "InceptionResNetV2": "inception_resnet_v2"}
cnn = getattr(keras.applications, network)()  # import network
network_module = getattr(keras.applications, network_mod[network])
preprocess_input = getattr(network_module, "preprocess_input")
layers = []  # interesting layers, omitting pooling layer and fc
for i, layer in enumerate(cnn.layers):
    if 'conv' not in layer.name:
        continue
    layers.append(i)
outputs = [cnn.layers[i].output for i in layers]
outputs_model = keras.Model([cnn.input], outputs)

# functions
def preprocess_images_from_dir(i_dir, preprocess_fun, i_rows=224, i_cols=224,
                               seed_rn="123", val_split=0.2):
    """
    prepares the images in images_dir as input for the networks.

    input:
    - i_dir: path to the directory with image data, the folders in it should be
             named after the image class of the images they contain.
    - preprocess_fun: preprocessing function
    - i_rows: height of images
    - i_cols: width of images
    - seed_rn: random seed
    - val_split: fraction used for test set, 1-split is used for train set

    returns:
    - images_train: train set
    - images_test: test set

    remark:
    uses tensorflow.keras as keras and ImageDataGenerator from
    tensorflow.keras.preprocessing.image
    """
    # prepare datagenerator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fun,
                                 validation_split=val_split)
    # create dataframe with infos for imagespaths and shuffle it
    folders = os.listdir(i_dir)
    numfolders = len(folders)
    files = []
    target = []
    for folder in folders[0:numfolders]:
        imagepaths = os.listdir(i_dir + "/" + folder)
        for imagepath in imagepaths:
            files.append(i_dir + "/" + folder + "/" + imagepath)
            target.append(folder)
    df_images = pd.DataFrame({"imagepath": files, "wnid": target})
    df_images = df_images.sample(frac=1,
                                 random_state=seed_rn).reset_index(drop=True)
    # collect train images
    logger.info("Preparing Train Set")
    images_train = datagen.flow_from_dataframe(df_images, x_col='imagepath',
                                               y_col='wnid', batch_size=1,
                                               target_size=(i_rows, i_cols),
                                               shuffle=False, seed=seed_rn,
                                               subset="training")
    # collect test images
    logger.info("Preparing Test Set")
    images_test = datagen.flow_from_dataframe(df_images, x_col='imagepath',
                                              y_col='wnid', batch_size=1,
                                              target_size=(i_rows, i_cols),
                                              shuffle=False, seed=seed_rn,
                                              subset="validation")
    logger.info("images ready")
    return images_train, images_test

def featuremap_values(acts):
    '''
    averages the activations of each feature map to get one single number for
    each feature map

    input:
    - acts: activations

    return:
    - featuremap_val: feature map values (averaged activations for each feature
      map)
    '''
    featuremap_val = [[] for x in range(len(acts))]
    for i, layer in enumerate(acts):
        featuremap_val[i] = layer.mean(axis=(0, 1, 2))
    return featuremap_val

def set_threshold(fil_act,k_q_,percent=False):
  '''
  determines the most active filters by choosing the k or q% highest average activations

  input:
    fil_act: (averaged) activation of filters
    k_q_: number indicating used method for threshold. 0 for layer average as threshold, else k or q%-most activations per layer
    percent: boolean wether q-percent most active filters or k-most active filters
  
  output:
    fil_act_t: thresholded 'most active' filters
  '''
  if k_q_ < 0:
    warn("k_q_ too small (negative not allowed). k_q_ is set to 0 (average) for all layers")
    k_q_=0

  fil_act_t=fil_act.copy()
  for i,layer in enumerate(fil_act_t):
    k_q_temp=k_q_
    if percent:
      con=100
    else:
      con=len(layer)
    if k_q_temp==0: # arithmetic mean as threshold
      threshold=np.average(layer)
      ind=np.nonzero(layer>threshold)
    elif k_q_temp>con: # arithmetic mean as threshold
      warn(k_q_temp+" too big for layer "+str(i)+" with length "+con+". k_q is set to 0 (average) for this layer")
      threshold=np.average(layer)
      ind=np.nonzero(layer>threshold)
    else:
      if percent: # q% threshold
        p=int(round((k_q_temp/100)*len(layer)))
        ind=np.argpartition(layer, -p)[-p:]
      else: # k threshold
        ind=np.argpartition(layer, -k_q_temp)[-k_q_temp:]
    fil_act_t[i]=np.zeros(len(fil_act_t[i]))
    fil_act_t[i][ind]=1
  return fil_act_t

def get_labels(filterlabels, fil_act_t):
    '''
    assigns the corresponding labels from the filters of the network (filterlabels) 
    to the most active filters of the image (fil_act_t)

    inputs:
    - filterlabels: filterlabels (from step2)
    - fil_act_t: thresholded activations

    returns:
    - prep_fil_labels: for each filter either "inactive" or contains the corresponding 
      filterlabels from step2 if the filter belongs to the most active ones
    '''
    prep_fil_labels = {}
    for step, layer in enumerate(filterlabels):
        prep_fil_labels[step] = {}
        for count, fil in enumerate(layer):
            prep_fil_labels[step][count] = []
            if fil_act_t[step][count] == 1:
                for l in fil:
                    prep_fil_labels[step][count].append(l)
            else:
                prep_fil_labels[step][count] = "inactive"
    return prep_fil_labels

def load_assign_labels(k_q_, networklabels, fil_act, scope):
    '''
    loads files with labels for the filters of the network and assigns them

    inputs:
    - k_q_: list with the different values for k or q
    - networklabels: filterlabels from step2)
    - fil_act: filter activations
    - scope: "k" or "q%"

    returns:
    - lab: labelled filters
    - fil_act_t: thresholded activations, for each filter 1 if above and 0 otherwise
    '''
    lab = {}
    fil_act_t = {}
    for i, el in enumerate(k_q_):
        try:
            networklabels[el]
        except KeyError:
            warn("No labels available for " + scope + "=" + str(el))
            continue
        if not any(networklabels[el]):
            warn("No labels available for " + scope + "=" + str(el))
            continue
        fil_act_t[el] = set_threshold(fil_act, el)
        lab[el] = get_labels(networklabels[el], fil_act_t[el])
    return lab, fil_act_t

def counting_labels(lab):
    '''
    counts the labels of the filters

    inputs:
    - lab: labels of each filter

    returns:
    - counts: labelcounts for each layer and k
    - uniquelabels: filters with only one label
    '''
    counts = {}
    uniquelabels = {}
    for i, el in enumerate(lab):
        uniquelabels[el] = []
        counts[el] = [None] * len(lab[el])
        for j, layer in enumerate(lab[el]):
            counts[el][j] = Counter()
            for m, fil in enumerate(lab[el][j]):
                if lab[el][j][m] is not "inactive":
                    for label in lab[el][j][m]:
                        counts[el][j][label] += 1
                        if len(lab[el][j][m]) == 1:
                            uniquelabels[el].append([j, m, label])
    return counts, uniquelabels

def totalcounts_predict(counts):
    '''
    computes the total counts for the labels and the predictions based on labels of last layer (appred_ll) and labels of all layers (appred)

    inputs:
    - counts: labelcounts for each layer and k

    returns:
    - appred: totallabelcounts for all layers together
    - appred_ll: totallabelcounts for last layer
    '''
    totalcounts = {}
    appred = {}
    appred_ll = {}
    for i, el in enumerate(counts):
        totalcounts[el] = Counter()
        for j, layer in enumerate(counts[el]):
            totalcounts[el] = totalcounts[el] + layer
        appred[el] = totalcounts[el].most_common()  # change counter to list object
        appred_ll[el] = counts[el][len(counts[el]) - 1].most_common()
    return appred, appred_ll

"""## Get Predictions for test images"""

# load images
train, test = preprocess_images_from_dir(images_dir, preprocess_input, img_rows,
                                         img_cols,  seed_rn, split)
# extract class labels
test_labels_id_inv = test.class_indices  # get dict with mapping of labels
test_labels_id = {y: x for x, y in test_labels_id_inv.items()}  # invert dict

num_classes = len(test_labels_id)
num_images=test.n

# info about saved variables
for el in ["y", "y_k", "y_k_ll", "y_q", "y_q_ll", "class_images_dict"]:
    save_vars_list.append(el)
with open(save_vars_names, "wb") as p:
    pickle.dump(save_vars_list, p)
with open(save_vars_info, "a") as txt:
    txt.write("y: predictions of cnn\n")
    txt.write("y_k_k: labelcounts (predictions) of approach for different k\n")
    txt.write("y_k_q: labelcounts (predictions) of approach for different q%\n")
    txt.write("y_k_k_ll: labelcounts (predictions) of last layer for different k\n")
    txt.write("y_k_q_ll: labelcounts (predictions) of last layer for different q%\n")
    txt.write("class_images_dict: contains the corresponding file names for each class\n")

for image_class in tqdm(range(num_classes)):
    logger.info("class: " + str(test_labels_id[image_class]) + " (labelnum:" + str(image_class) + ")")

    # reset the dicts every step to save RAM
    y = {}  # true label for each image
    y_k = {}  # prediction(s) of k for each image
    y_k_ll = {}  # prediction(s) of k for each image according to last layer
    y_q = {}  # prediction(s) of q% for each image
    y_q_ll = {}  # prediction(s) of q% for each image according to last layer
    class_images_dict = {}  # images of each class

    y[image_class] = {}  # add class info
    y_k[image_class] = {}  # add class info
    y_k_ll[image_class] = {}  # add class info
    y_q[image_class] = {}  # add class info
    y_q_ll[image_class] = {}  # add class info
    class_images = [test[j] for j in range(num_images) if test.labels[j] == image_class]
    class_images_dict[image_class] = [test.filenames[j] for j in range(num_images) if test.labels[j] == image_class]
    numimages = len(class_images)
    if not class_images:
        warn("no images for class " + str(test_labels_id[image_class]) + " (labelnum:" + str(
            image_class) + ") in train set")
        continue
    for imgnum, image in enumerate(tqdm(class_images)):
        logger.info(
            "image " + str(imgnum + 1) + " of " + str(numimages) + " (class " + str(image_class + 1) + " of " + str(
                num_classes) + ")")

        # True Label (Prediction of the network)
        preds = cnn(image[0]).numpy()  # [0]imagearray,[1]label
        y[image_class][imgnum] = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=3)

        # comppute activations of filters
        out = outputs_model.predict([image[0]])
        fil_act = featuremap_values(out)

        # get labels of active filters
        lab_k, fil_act_t_k = load_assign_labels(k_q, filTags_k, fil_act, "k")
        lab_q, fil_act_t_q = load_assign_labels(k_q, filTags_q, fil_act, "q%")

        # statistics
        ##counting the labels of the filters for each layer
        counts_k, uniquelabel_k = counting_labels(lab_k)
        counts_q, uniquelabel_q = counting_labels(lab_q)

        ##totalcounts(predictions)
        y_k[image_class][imgnum], y_k_ll[image_class][imgnum] = totalcounts_predict(counts_k)
        y_q[image_class][imgnum], y_q_ll[image_class][imgnum] = totalcounts_predict(counts_q)

    with open(save_vars + str(image_class), "wb") as p:
        for el in [y, y_k, y_k_ll, y_q, y_q_ll, class_images_dict]:
            pickle.dump(el, p)
    logger.info(
        "saved intermediate predictions (y, y_k, y_k_ll, y_q, y_q_ll, class_images_dict) in:" + save_vars + str(
            image_class))