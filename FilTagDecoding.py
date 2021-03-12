import pickle
import csv
from tqdm import tqdm

filTags_file = r'C:\Users\Daniel\PycharmProjects\dashboards\FilTags\2021-01-30-1722-VGG16-LF-vars'
decoding_file = r'C:\Users\Daniel\PycharmProjects\dashboards\FilTags\decoding_wnids.txt'
save_file = r'C:\Users\Daniel\PycharmProjects\dashboards\FilTags\2021-01-30-1722-VGG16-filTags-hr'

print("load data")
with open(decoding_file, mode='r') as infile:
    reader = csv.reader(infile)
    labels_dec = {rows[0]: rows[1] for rows in reader}
with open(filTags_file, 'rb') as pkl:
    filTags_orig = pickle.load(pkl)
filTags = filTags_orig.copy()
print("preparations done, ready to decode")

for n1, k in enumerate(tqdm(filTags)):
    for n2, layer in enumerate(filTags[k]):
        for n3, fil in enumerate(layer):
            for n4, label in enumerate(fil):
                filTags[n1][n2][n3][n4] = labels_dec[label]
print("decoding done, saving decoded labels as: "+save_file)

with open(save_file, 'wb') as pkl:
    pickle.dump(filTags, pkl)

print('run successful')
