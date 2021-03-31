"""# Hitrates and Evaluation Results"""

# imports
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys
from collections import Counter
from datetime import datetime, timedelta
from tqdm import tqdm
from warnings import warn

"""## Input"""
eval_file="/content/Results/2021-01-01-0000-E-VGG16-vars"
save_dir="/content/Results"
m=[i for i in range(1,101)]
k_q=[1,5,25]
classes_img=[0,1,2]#[i for i in range(1000)]# snake(52:69),dogs(151:269)
time_delta=2

"""## Preparations"""

#create ID for files
network=eval_file.split('-')[-2]
now = datetime.utcnow() + timedelta(hours=time_delta)
runid = now.strftime("%Y-%m-%d-%H%M")+"-HR"
if not save_dir:
    save_dir = os.getcwd()
save_vars = save_dir + "/" + runid + "-" + network + "-vars"
save_vars_names = save_vars + "-names"
save_vars_list = []
save_vars_info = save_vars + "-info.txt"
save_vars_results = save_vars + "-results.txt"
with open(save_vars_info,"w") as txt:
  txt.write("Vars Info HR\n----------\n\n")
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

"""### Functions"""
def compute_hitrates(z,z_k,z_q,images_files,k_q_,m=[1,5,10,100]):
  '''
  This method calculates the hitrates for a class

  inputs:
  - z: network predictions
  - z_k: approach predictions for k
  - z_q: approach predictions for q%
  - images_files: dict with paths to the images
  - k_q_: list with values for k and q
  - m: model parameter, determines the number of top label counts used as 
      predictions

  returns:
  - acc_k: hitrates for different k
  - acc_q: hitrates for different q%

  remark: This function may cause a high number of warnings especially for
          high m
  '''
  acc_k={}
  acc_q={}
  numimages=len(images_files)
  for el in k_q:
    acc_k[el]={}
    acc_q[el]={}
    for re in m:
      acc_k[el][re]=0
      acc_q[el][re]=0    
  for imgnum,image in enumerate(images_files):
    for re in m:
      for el in k_q_:
        correct_k=0
        correct_q=0
        #k
        for r in range(re+1):
          try:
            correct_k=correct_k+(z_k[imgnum][el][r][0]==z[imgnum][0][0][0][0])
          except IndexError:
            warn("only "+str(r)+" predictions for "+str(image)
                 +" (k="+str(el)+")")
            break
        #q
        for r in range(re+1): 
          try:
            correct_q=correct_q+(z_q[imgnum][el][r][0]
                                     ==z[imgnum][0][0][0][0])
          except IndexError:
            warn("only "+str(r)+" predictions for: "+str(image)
                 +" (q%="+str(el)+")")
            break
        acc_k[el][re]=acc_k[el][re]+correct_k
        acc_q[el][re]=acc_q[el][re]+correct_q
  for el in k_q_:
    for re in m:
      acc_k[el][re]=acc_k[el][re]/numimages
      acc_q[el][re]=acc_q[el][re]/numimages
  return acc_k,acc_q

def print_hitrates(txt_file,acc_k_,acc_q_,k_q,show_m=[1,10,50],chapterid=2):
  '''
  This method prints the hitrates for a class into a .txt file 

  input:
  - txt_file: (string) path to .txt file 
  - acc_k_: hitrates for different k
  - acc_q_: hitrates for different q%
  - k_q:list with values for k and q
  - show_m: values for parameter m that are shown
  - chapterid: chapter identifier
  '''
  # write Hitrates into txtfile

  with open(txt_file, 'a') as txt:
    txt.write("\n"+str(chapterid)+"Hitrates\n"+12*"-"+"\n\n")
    txt.write(" k\\m  ")
    for re in show_m:
      txt.write('{:7}'.format("| "+str(re)))
    txt.write("\n"+(6*"-"+len(show_m)*("|"+6*"-"))+"\n")
    for el in k_q:
      txt.write('{: <6}'.format("k="+str(el))) 
      for re in show_m:
        txt.write("|"+'{: >6.2f}'.format(acc_k_[el][re]*100))
      txt.write("\n")
    for el in k_q:
      txt.write('{: <6}'.format("q%="+str(el)))
      for re in show_m:
        txt.write("|"+'{: >6.2f}'.format(acc_q_[el][re]*100))
      txt.write("\n")
      
"""Get Hitrates"""

hit_k={}
hit_q={}
hit_k_ll={}
hit_q_ll={}
paths={}
for c,i in enumerate(tqdm(classes_img
                          )):
  logger.info("class "+str(i))
  #load files from step3
  num_vars=0
  try:
    with open(eval_file+str(i),"rb") as p:
      for el in ["y","y_k","y_k_ll","y_q","y_q_ll",
                "class_images_dict"]:
        globals()[el]=pickle.load(p)
        num_vars=num_vars+1
  except EOFError:
    warn("only "+num_vars+"/6 vars from pickle file")
  logger.info("number of variables loaded succesfully:"+str(num_vars))
  paths[i]=class_images_dict
  
  #compute hitrates per class
  logger.info("computing hitrates")
  try: 
    hit[i],hit_q[i]=compute_hitrates(y[i],y_k[i],y_q[i],class_images_dict,k_q,m)
  except IndexError:
    warn("not able to compute hitrate for class "+str(i))

  try:  
    hit_ll[i],hit_q_ll[i]=compute_hitrates(y[i],y_k_ll[i],y_q_ll[i],class_images_dict,k_q,m)
  except IndexError:
    warn("not able to compute hitrate for class "+str(i)+" (last layer")

#print hitrates into txtfile
for i in classes_img:
  with open(save_vars_results, 'a') as txt:
    txt.write("\nClass "+str(i)+"\n")
  print_hitrates(save_vars_results,hit[i],hit_q[i],k_q,[1,10,20,30,40,50,100],i)
  print_hitrates(save_vars_results,hit_ll[i],hit_q_ll[i],k_q,[1,10,20,30,40,50,100],2*i)
total_img=0
for i in classes_img:
  total_img=total_img+len(paths[i])

# hitrates of all classes
#k
totalhit_k={}
for p in k_q:  
  totalhit_k[p]={}
  for re in m:
    totalhit_k[p][re]=0
    for i in classes_img:
      totalhit_k[p][re]=totalhit_k[p][re]+hit[i][p][re]*len(paths[i])
#q%    
totalhit_q={}
for p in k_q:  
  totalhit_q[p]={}
  for re in m:
    totalhit_q[p][re]=0
    for i in classes_img:
      totalhit_q[p][re]=totalhit_q[p][re]+hit_q[i][p][re]*len(paths[i])
for p in k_q:  
  for re in m:
    totalhit_k[p][re]=totalhit_k[p][re]/total_img
    totalhit_q[p][re]=totalhit_q[p][re]/total_img

#save hitrates as pickle file
with open(save_vars,"ab") as p:
  for el in [totalhit_l,totalhit_q]:
    pickle.dump(el,p)
with open(save_vars_info,"a") as txt:
  for el in ["totalhit_k","totalhit_q"]:
    save_vars_list.append(el)
    txt.write("- "+el+"\n")
  logger.info("accuracies for k and q% saved in: "+save_vars)
print_hitrates(save_vars_results,totalhit,totalhit_q,k_q,[1,10,20,30,40,50,100],"Testset)")
print_hitrates(save_vars_results,totalhit,totalhit_q,k_q,[i for i in range(1,101)],"Testset)")
with open(save_vars_results, 'a') as txt:
  txt.write("\nTotal images in this superset "+str(total_img))