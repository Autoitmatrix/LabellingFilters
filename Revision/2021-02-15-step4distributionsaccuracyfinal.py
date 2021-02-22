# -*- coding: utf-8 -*-
"""Step4DistributionsAccuracyFINAL.ipynb

# Accuracy and Figures (Step 4)

This file calculates accuracies for the prediction based on the quantity of labels for each filter.
- for all convolutional layers (method A)
- for last convolutional layer (method B)

for single images
for classes

# Imports
"""

import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
from warnings import warn, filterwarnings

"""# Input"""
class_num=2
network="VGG16"
q=[2,3,4,10]
closer_look=[class_num] #choose classes for a more detailed analysis
pickle_file=(r"C:/Users/Daniel/PycharmProjects/LabellingFilters/Pickles/2021-02-09-1405-VGG16-E-vars"+str(class_num)) # q=[2,3,4,10,25,50]
#(r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Pickles\2021-02-03-1950-VGG16-E-vars"+str(class_num))# q=[1,5]
decoding_file="/home/ws/ns1888/FilterLabel/codes/decoding_wnids.txt"
save_dir=r"C:/Users/Daniel/PycharmProjects/LabellingFilters/Pickles"

#Test and debug
loglevel="INFO"
filterwarnings("ignore")

#Preparations

#create ID for files
now = datetime.utcnow()+timedelta(hours=1)
runid = now.strftime("%Y-%m-%d-%H%M")
if not save_dir:
  save_dir=os.getcwd()
save_vars=save_dir+"/"+runid+"-"+network+"-PH-vars"
save_vars_names=save_dir+"/"+runid+"-"+network+"-PH-vars-names"
save_vars_list=[]
save_vars_info=save_vars+"-info.txt"
save_vars_results=save_vars+"-results.txt"
with open(save_vars_info,"a") as txt:
  txt.write("Vars Info PH\n----------\n\n")
  txt.write("This file gives an overview of the stored variables in pickle file "
          +save_vars+"\n")
  txt.write("To load them use e.g.:\n")
  txt.write("with open(\""+save_vars_names+"\",\"rb\") as r:\n")
  txt.write("  var_names=pickle.load(r)\n")
  txt.write("with open(\""+save_vars+"\",\"rb\") as p:\n")
  txt.write("  for var in var_names:\n")
  txt.write("    globals()[var]=pickle.load(p)\n\n")
  txt.write("stored variables are:\n")

#logging
logging.captureWarnings(True)
logger = logging.getLogger('logger1')
logger.setLevel(loglevel)
if (logger.hasHandlers()):
    logger.handlers.clear()
info_handler = logging.FileHandler(save_dir+"/"+runid+'-PH-info.log')
info_handler.setLevel(logging.INFO)
infoformat = logging.Formatter('%(asctime)s - %(message)s',
                               datefmt='%d.%m.%Y %H:%M:%S')
info_handler.setFormatter(infoformat)
logger.addHandler(info_handler)
stream_handler=logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(infoformat)
logger.addHandler(stream_handler)
warnings_logger = logging.getLogger("py.warnings")
warnings_handler = logging.FileHandler(save_dir+"/"+runid+'-PH-warnings.log')
warnings_logger.addHandler(warnings_handler)
warnings_stream_handler=logging.StreamHandler()
warnings_logger.addHandler(warnings_stream_handler)
if logger.getEffectiveLevel()==10:
  debug_handler = logging.FileHandler(savedir+"/"+runid+'-PH-debug.log')
  debug_handler.setLevel(logging.DEBUG)
  debugformat = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s -'
                                  +' %(message)s',datefmt='%d.%m.%Y %H:%M:%S')
  debug_handler.setFormatter(debugformat)
  logger.addHandler(debug_handler)
logger.propagate = False

#load files from step3
# num_vars=0
# with open(pickle_file+"-names","rb") as r:
#   var_names=pickle.load(r)
# with open(pickle_file,"rb") as p:
#   for var in var_names:
#     try:
#       globals()[var]=pickle.load(p)
#       num_vars=num_vars+1
#     except EOFError:
#       warn("not all variables could be loaded. File only contains "
#            +str(num_vars))
#       break

with open(pickle_file, "rb") as p:
  for var in ["y","y_hat","y_hat_per","y_hat_ll","y_hat_per_ll","class_images_dict"]:
    globals()[var]=pickle.load(p)

#labels
try:
  with open(decoding_file, mode='r') as infile:
    reader = csv.reader(infile)
    labels_dec = {rows[0]:rows[1] for rows in reader}
  logger.debug("decoding ready")
except FileNotFoundError:
  warn("no decoding file: "+decoding_file)
  labels_dec={}


"""## Distributions and Predictions for Single Images """

def decode_label(labelx, label_dec):
  '''
  this function tries to decode label labelx

  input:
  - labelx: label to decode
  - label_dec: dict how to decode

  return:
  - labely: if possible decoded label, otherwise original label
  '''
  try:
    labely=labels_dec[labelx]
  except KeyError:
    warn("not able to decode "+str(labelx))
    labely=labelx
  return labely

def print_dist_pred(txt_file,z, z_hat, z_hat_per,images_files,q,labels_dec
                    ,chapterid=1.1,show_top=3):
  '''
  This function prints the distributions and predictions of the approach in a 
  txt.file

  inputs:
  - txt_file: (string) path to .txt file 
  - z: network predictions
  - z_hat: approach predictions for q
  - z_hat_per: approach predictions for q%
  - images_files:dict with paths to the images
  - q: list with values for q
  - labels_dec: dict to decode the labels from wnid into human readable label
  - chapterid: chapter identifier
  - show_top: (int) number of top label counts to be displayed

  remark: This function may cause a high number of warnings especially for small
          q and high show_top.
  '''
  with open(txt_file, 'a') as txt:
    txt.write(str(chapterid)+"Distributions and Predictions\n"+36*"-"+"\n")
    txt.write("counts of labels for each image\n\n")

    for imgnum,image in enumerate(images_files):
      txt.write(str(image)+"\nnetwork prediction: "+str(z[imgnum])+"\n")
      for j,el in enumerate(q):

        #q
        txt.write("{:7}".format("q="+str(el)+":  "))
        for t in range(show_top):
          if el not in z_hat[imgnum]:
            warn("no labels for: "+str(image)+" and q="+str(el))
            break
          try:
            txt.write(str(decode_label(z_hat[imgnum][el][t][0],labels_dec))+"("
                      +str(z_hat[imgnum][el][t][1])+"), ")
          except IndexError:
            warn("only "+str(t)+" predictions available for: "+str(image)
            +" (q="+str(el)+").")
            break

        #add labelcount of network prediction
        pred_avail=False   
        for i in range(len(z_hat[imgnum][el])):
          if z_hat[imgnum][el][i][0]==z[imgnum][0][0][0]:
            txt.write("|"+str(i+1)+"."
                      +str(decode_label(z_hat[imgnum][el][i][0],labels_dec))
                      +"("+str(z_hat[imgnum][el][i][1])+")")
            pred_avail=True
            break
        if not pred_avail:
          txt.write("|"+str(decode_label(z[imgnum][0][0][0],labels_dec))+"(0)")

        #q%
        txt.write("\n"+"{:7}".format("q%="+str(el)+":  "))
        for t in range(show_top):
          if el not in z_hat_per[imgnum]:
            warn("no labels for "+str(image)+"and q%="+str(el))
            break
          try:  
            txt.write(str(decode_label(z_hat_per[imgnum][el][t][0],labels_dec))
                      +"("+str(z_hat_per[imgnum][el][t][1])+"), ")
          except KeyError:
            warn("not able to decode label "+z_hat_per[imgnum][el][t][0])
            txt.write(str(z_hat_per[imgnum][el][t][0])+"("
                      +str(z_hat_per[imgnum][el][t][1])+"), ")
          except IndexError:
            warn("only "+str(t)+" predictions available for "+str(image)
                 +" (q%="+str(el)+")")
            break

        #add labelcount of network prediction
        pred_avail=False
        try:    
          for i in range(len(z_hat_per[imgnum][el])):
            if z_hat_per[imgnum][el][i][0]==z[imgnum][0][0][0]:
              txt.write("|"+str(i+1)+"."
                        +str(decode_label(z_hat_per[imgnum][el][i][0],
                                          labels_dec))
                        +"("+str(z_hat_per[imgnum][el][i][1])+")")
              pred_avail=True
              break
        except KeyError:
            pass
        if not pred_avail:
          txt.write("|"+str(decode_label(z[imgnum][0][0][0],labels_dec))+"(0)")
        txt.write("\n")
      txt.write("\n")
    logger.debug("finished printing of distributions")

for image_class in closer_look:
  #Method A all layers
  print_dist_pred(save_vars_results,y[image_class], y_hat[image_class], y_hat_per[image_class],
                  class_images_dict[image_class],q,labels_dec,1.1,
                  show_top=3)

  #Method B last conv layer
  print_dist_pred(save_vars_results,y[image_class], y_hat_ll[image_class], y_hat_per_ll[image_class],
                  class_images_dict[image_class],q,labels_dec,1.2,
                  show_top=3)

for image_class in closer_look:
  with open(save_vars_results, 'a') as txt:
    txt.write("\nExample Predictions\n"+36*"-"
              +"\nWe consider m=1, i.e. only the most occuring label is taken as "
              +"predictor for the\nclass. Most of the predictions are therefore "
              +"False\n\n")
    txt.write("q    |image name          |    y    |predict q|correct?|predict q%"
              +"|correct?\n"+5*"-"+"|"+20*"-"+"|"+2*(9*"-"+"|")+8*"-"+"|"+10*"-"
              +"|"+8*"-"+"\n")
    for el in q:
      for imgnum, image in enumerate(class_images_dict[image_class]):
        txt.write("q="+'{:3}'.format(el)+"|"+"{:20}".format(image[-19:-6])+"|"
                  +str(y[image_class][imgnum][0][0][0])+"|")
        try:
          txt.write("{:9}".format(y_hat[image_class][imgnum][el][0][0])+"|")
          if (y[image_class][imgnum][0][0][0]==y_hat[image_class][imgnum][el][0][0]):
            txt.write("  True  |")
          else:
            txt.write("  False |")
        except IndexError:
          warn("no prediction available for "+str(image)+" (q="+str(el)+")")
          txt.write("{:9}".format(str(y_hat[image_class][imgnum][el]))+"|"+"  False |")
        try:
          txt.write('{:10}'.format(y_hat_per[image_class][imgnum][el][0][0])+"|")
          if (y[image_class][imgnum][0][0][0]==y_hat_per[image_class][imgnum][el][0][0]):
            txt.write("  True  \n")
          else:
            txt.write("  False \n")
        except IndexError:
          warn("no prediction available for "+str(image)+" (q%="+str(el)+")")
          txt.write("{:10}".format(str(y_hat_per[image_class][imgnum][el]))+"|"+"  False |\n")
  logger.info("Example saved in "+str(save_vars_results))

"""## Accuracies for Classes of Images, Supersets and the whole Test Set"""

def compute_hitrates(z,z_hat,z_hat_per,images_files,q):
  '''
  This method calculates the accuracies for a class

  inputs:
  - z: network predictions
  - z_hat: approach predictions for q
  - z_hat_per: approach predictions for q%
  - images_files:dict with paths to the images
  - q: list with values for q

  returns:
  - acc: accuracies for different q
  - acc_per: accuracies for different q%

  remark: This function may cause a high number of warnings especially for
          high m
  '''
  logger.debug("computing hitrates")
  m=[i for i in range(101)] #relaxation: take m most labels as prediction
  acc={}
  acc_per={}
  numimages=len(images_files)
  for el in q:
    acc[el]={}
    acc_per[el]={}
    for re in m:
      acc[el][re]=0
      acc_per[el][re]=0    
  for imgnum,image in enumerate(images_files):
    for re in m:
      for el in q:
        correct=0
        correct_per=0
        for r in range(re+1):
          try:
            correct=correct+(z_hat[imgnum][el][r][0]==z[imgnum][0][0][0])
          except IndexError:
            warn("only "+str(r)+" predictions for "+str(image)
                 +" (q="+str(el)+")")
            break
        for r in range(re+1): 
          try:
            correct_per=correct_per+(z_hat_per[imgnum][el][r][0]
                                     ==z[imgnum][0][0][0])
          except IndexError:
            warn("only "+str(r)+" predictions for: "+str(image)
                 +" (q%="+str(el)+")")
            break
        acc[el][re]=acc[el][re]+correct
        acc_per[el][re]=acc_per[el][re]+correct_per
  for el in q:
    for re in m:
      acc[el][re]=acc[el][re]/numimages
      acc_per[el][re]=acc_per[el][re]/numimages
  return acc,acc_per

def compute_MRR(z,z_hat,z_hat_per,images_files,q):
  '''
  This method calculates the accuracies for a class

  inputs:
  - z: network predictions
  - z_hat: approach predictions for q
  - z_hat_per: approach predictions for q%
  - images_files:dict with paths to the images
  - q: list with values for q

  returns:
  - mrr: mean reciprocal rank for different q
  - mrr_per: mean reciprocal rank for different q%
  '''
  logger.debug("computing mrr")
  mrr={}
  mrr_per={}
  rr={}
  rr_per={}
  numimages=len(images_files)
  for el in q:
    rr[el] = {}
    rr_per[el] = {}
  pred_avail = False
  for el in q:
    for imgnum in range(numimages):
      #q
      for i in range(len(z_hat[imgnum][el])):
        if z_hat[imgnum][el][i][0] == z[imgnum][0][0][0]:
          rr[el][imgnum]=i+1
          pred_avail = True
          break
      if not pred_avail:
        rr[el][imgnum] = float("NaN")
      #q%
      pred_avail=False
      for i in range(len(z_hat_per[imgnum][el])):
        if z_hat_per[imgnum][el][i][0] == z[imgnum][0][0][0]:
          rr_per[el][imgnum]=i+1
          pred_avail = True
          break
      if not pred_avail:
        rr_per[el][imgnum] = float("NaN")

    mrr[el]=np.nanmedian(list(rr[el].values()))
    mrr_per[el]=np.nanmedian(list(rr_per[el].values()))

  return rr, rr_per, mrr, mrr_per


def print_hitrates(txt_file,acc,acc_per,q,show_m=[1,10,50],chapterid=2):
  '''
  This method prints the accuracies for a class into a .txt file 

  input:
  - txt_file: (string) path to .txt file 
  - acc: hitrates for different q
  - acc_per: hitrates for different q%
  - q:list with values for q
  - show_m: values for parameter m that are shown
  - chapterid: chapter identifier
  '''
  # write Hitrates into txtfile

  with open(txt_file, 'a') as txt:
    txt.write("\n"+str(chapterid)+"Hits@k\n"+12*"-"+"\n\n")
    txt.write(" q\\k  ")
    for re in show_m:
      txt.write('{:7}'.format("| "+str(re)))
    txt.write("\n"+(6*"-"+len(show_m)*("|"+6*"-"))+"\n")
    for el in q:
      txt.write('{: <6}'.format("q="+str(el))) 
      for re in show_m:
        txt.write("|"+'{: >6.2f}'.format(acc[el][re]*100))
      txt.write("\n")
    for el in q:
      txt.write('{: <6}'.format("q%="+str(el)))
      for re in show_m:
        txt.write("|"+'{: >6.2f}'.format(acc_per[el][re]*100))
      txt.write("\n")

for image_class in closer_look:
  #calculate hitrates
  hit,hit_per=compute_hitrates(y[image_class],y_hat[image_class],y_hat_per[image_class],class_images_dict[image_class],q)
  hit_ll,hit_per_ll=compute_hitrates(y[image_class],y_hat_ll[image_class],y_hat_per_ll[image_class],class_images_dict[image_class],q)
  rrs, rrs_per, mrrs, mrrs_per = compute_MRR(y[image_class], y_hat[image_class], y_hat_per[image_class], class_images_dict[image_class], q)


  #print hitrates into txtfile
  print_hitrates(save_vars_results,hit,hit_per,q,show_m=[1,2,3,4,5,10,20,30]) #[i for i in range(1,101)]
  print_hitrates(save_vars_results,hit_ll,hit_per_ll,q,show_m=[1,2,3,4,5,10,20,30]) #[i for i in range(1,101)]
  #save accuracies as pickle file
  with open(save_vars,"ab") as p:
    for el in [hit,hit_per]:
      pickle.dump(el,p)
  with open(save_vars_info,"a") as txt:
    for el in ["hit","hit_per"]:
      save_vars_list.append(el)
      txt.write("- "+el+"\n")
  logger.info("accuracies for q and q% saved in: "+save_vars)

  #plot q
  imclass=str(decode_label(image_class, labels_dec))# list(labels_dec.items())[image_class][1]
  acc_df={}
  lineformat=['o-b','^-g','*-r','+-y','-c','s-m','D-q','<-w']
  for l,el in enumerate(q):
    acc_df[el]=pd.DataFrame.from_dict(hit[el],orient='index')[0:31]
    plt.plot(acc_df[el], lineformat[l] , label='q='+str(el) ) #
  plt.title("Hits@k of class"+imclass+" for different q")
  plt.ylabel("Hits@k")
  plt.xlabel("k")
  plt.legend()
  fig = plt.gcf()
  fig.set_size_inches(8, 6)
  plt.savefig(save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+"-q.pdf", bbox_inches='tight')
  plt.show()
  logger.info("Plot hitsa@k for q saved in: "+save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+"-q.pdf")

  #plot q%
  acc_df={}
  for l,el in enumerate(q):
    acc_df[el]=pd.DataFrame.from_dict(hit_per[el],orient='index')[0:31]
    plt.plot(acc_df[el],lineformat[l],label='q%='+str(el))
  plt.title("Hits@k of class"+imclass+" for different q%")
  plt.ylabel("Hits@k")
  plt.xlabel("k")
  plt.legend()
  fig = plt.gcf()
  fig.set_size_inches(8, 6)
  plt.savefig(save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+'-qper.pdf', bbox_inches='tight')
  plt.show()
  logger.info("Plot hitsa@k for q% saved in: " +save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+'-qper.pdf')

  # #plot method B q
  # acc_df={}
  # for el in q:
  #   acc_df[el]=pd.DataFrame.from_dict(hit_ll[el],orient='index')
  #   plt.plot(acc_df[el],label='q='+str(el))
  # plt.title("Accuracies for class"+imclass+" (method B)")
  # plt.ylabel("Accuracy")
  # plt.xlabel("m")
  # plt.legend()
  # fig = plt.gcf()
  # fig.set_size_inches(8, 6)
  # plt.savefig('hitrates_'+imclass+'_B1.pdf', bbox_inches='tight')
  # plt.show()
  # #plot method B q%
  # acc_df={}
  # for el in q:
  #   acc_df[el]=pd.DataFrame.from_dict(hit_per_ll[el],orient='index')
  #   plt.plot(acc_df[el],label='q%='+str(el))
  # plt.title("Accuracies for class"+imclass+" (method B)")
  # plt.ylabel("Accuracy")
  # plt.xlabel("m")
  # plt.legend()
  # fig = plt.gcf()
  # fig.set_size_inches(8, 6)
  # plt.savefig('hitrates_'+imclass+'_B2.pdf', bbox_inches='tight')
  # plt.show()