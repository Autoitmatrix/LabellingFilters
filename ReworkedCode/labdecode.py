import pickle
import csv

filTags_file="/content/Results/2021-01-01-0000-TF-VGG16-vars"
decoding_file="/content/decoding_wnids.txt"

with open(decoding_file) as txt:
  reader=csv.reader(txt)
  decode= {rows[0]:rows[1].strip() for rows in reader}
print('decoding ready')

with open(filTags_file,'rb') as p:
  filTags_k=pickle.load(p)
  filTags_q=pickle.load(p)

def decodingLabels(filTags,decode):  
  for k_q in filTags: #dict
    for layer in range(len(filTags[k_q])): # list
      for fil in range(len(filTags[k_q][layer])): # list
        if filTags[k_q][layer][fil]:
          for label,el in enumerate(filTags[k_q][layer][fil]):
            try:
              filTags[k_q][layer][fil][label]=decode[el]
            except KeyError:
              pass
  return(filTags)

filTags_k=decodingLabels(filTags_k,decode)
filTags_q=decodingLabels(filTags_q,decode)

with open(filTags_file+'-decode','wb') as p:
  pickle.dump(filTags_k,p)
  pickle.dump(filTags_q,p)