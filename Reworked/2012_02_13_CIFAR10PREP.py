import os
import pickle
from warnings import warn

import numpy as np
from PIL import Image

cifardir = r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Pickles"


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


cifardict = unpickle(cifardir + r"\data_batch_1")
# ggf.batches.meta für cifarcalssdict
cifarclassdict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"}

for i in range(10):
    try:
        os.mkdir(cifardir + "/" + cifarclassdict[i])
    except FileExistsError:
        warn("directory already exists: " + cifardir + "/" + cifarclassdict[i])

for i, el in enumerate(cifardict[b'labels']):
    im = np.asarray(cifardict[b'data'][i].T).astype("uint8")
    im.resize(3, 32, 32)
    im = im.transpose([1, 2, 0])
    im = Image.fromarray(im)
    im.save(cifardir + "/" + cifarclassdict[el] + "/" + str(cifardict[b'filenames'][i])[0:-4] + ".jpg")
