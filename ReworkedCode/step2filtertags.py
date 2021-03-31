"""# Step2 Tagging of Filters"""

"""## Imports"""

import logging
import numpy as np
import os
import pickle
from datetime import datetime,timedelta
from tqdm import tqdm
from warnings import warn, simplefilter

"""## Inputs"""

#input
k_q=[i for i in range(101)]
activations_file="/content/Results/2021-01-01-0000-QFA-VGG16-vars"
save_dir="/content/Results"
time_delta= 0

"""## Preparation"""

# create ID for files
network=activations_file.split("-")[-2]
now = datetime.utcnow()+timedelta(hours=time_delta)
runid = now.strftime("%Y-%m-%d-%H%M")+"-TF"
if not save_dir:
  save_dir=os.getcwd()
save_vars=save_dir+"/"+runid+"-"+network+"-vars"
save_vars_names=save_vars+"-names"
save_vars_list=[]
save_vars_info=save_vars+"-info.txt"
with open(save_vars_info,"a") as txt:
  txt.write("Vars Info LF\n----------\n\n")
  txt.write("This file gives an overview of the stored variables in pickle file "+save_vars+"\n")
  txt.write("To load them use e.g.:\n")
  txt.write("with open(\""+save_vars_names+"\",\"rb\") as r:\n")
  txt.write("  var_names=pickle.load(r)\n")
  txt.write("with open(\""+save_vars+"\",\"rb\") as p:\n")
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

# load files from step1
with open(activations_file+"-names", "rb") as p_1:
  varnames_step1 = pickle.load(p_1)
for i, j in enumerate(varnames_step1):
  varnames_step1[i] = j.split(":")[0]
with open(activations_file,"rb") as p_2:
  for var in varnames_step1:
    globals()[var]=pickle.load(p_2)
logger.info("following variables loaded:"+str(varnames_step1))

"""## Preparation"""

# functions
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
      warn(str(k_q_temp)+" too big for layer "+str(i)+" with length "+str(con)+". k_q is set to 0 (average) for this layer")
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

def label_it(fil_act_t,label,labels_old=[]):
  '''
  labels the most active filters for each class with corresponding class label, adds label to labels_old for all filters above threshold determined by fil_act

  inputs:
  - fil_act_t: thresholded active filters; list with np-arrays for each layer as elements; these np-arrays contain the thresholded activations (0:filter inactive, 1: filter active)
  - label: name of the class
  - labels_old: list with labels (3.dim) of each filter (2.dim) of each layer (1.dim)

  returns:
  - labels_new: updated list of labels
  '''
  if not fil_act_t:
    logger.warning("Warning: label_it input is empty for this class")
    warn("label_it input is empty for this class")
    return []
    
  if not labels_old: #initialization
    labels_old=[None]*len(fil_act_t)
    for count,layerf in enumerate(fil_act_t):
      labels_old[count]=[None]*len(layerf)
      for step,filterf in enumerate(layerf):
        labels_old[count][step]=[]
  for count,layerf in enumerate(fil_act_t):
    for step,filterf  in enumerate(layerf):
      if filterf==1.0: #append if active
        labels_old[count][step].append(label)
  labels_new=labels_old
  return labels_new

"""## Compute labels for filters"""

# k-labels
filTags_k={}
num_classes=len(labels_id)
for i,el in enumerate(tqdm(k_q)):
  labels_filters=[]
  for j in range(num_classes):
    fil_act_t=set_threshold(globals()['filter_act_'+str(j)],el,False)
    labels_filters=label_it(fil_act_t,labels_id[j],labels_filters)
  filTags_k[el]=labels_filters
with open(save_vars,"wb") as p:  
  pickle.dump(filTags_k,p)
save_vars_list.append("filTags_k")
with open(save_vars_info,"a") as txt:
  txt.write("- filTags_k: labels of filters for k in "+str(k_q)+"\n")
logger.info("k labels added to: "+save_vars)

# q%-labels
filTags_q={}
for i,el in enumerate(tqdm(k_q)):
  labels_filters=[]
  for j in range(num_classes): 
    fil_act_t=set_threshold(globals()['filter_act_'+str(j)],el,True)
    labels_filters=label_it(fil_act_t,labels_id[j],labels_filters)
  filTags_q[el]=labels_filters
with open(save_vars,"ab") as p_1:
  pickle.dump(filTags_q,p_1)
save_vars_list.append("filTags_q")
with open(save_vars_names,"wb") as p_2:
  pickle.dump(save_vars_list,p_2)
with open(save_vars_info,"a") as txt:
  txt.write("- filTags_q: labels of filters for q% in "+str(k_q))
logger.info("q% labels added to: "+save_vars)
logger.info("---------------Run succesful-----------------")