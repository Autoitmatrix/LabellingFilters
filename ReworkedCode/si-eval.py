"""# Single Image Evaluation"""

#Imports
import pickle
import csv
from warnings import warn
import os
import logging

"""## Input"""

eval_file="/content/Results/2021-01-01-0000-E-VGG16-vars"
save_dir="/content/Results"
time_delta=2
image_class=0

"""## Preparation"""
# create runID
now = datetime.utcnow() + timedelta(hours=time_delta)
runid = now.strftime("%Y-%m-%d-%H%M")+"-SI"
if not save_dir:
    save_dir = os.getcwd()
save_vars_results = save_dir + "/" + runid + "-" + network + "-results.txt"

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

#labels
with open(decoding_file, mode='r') as txt:
  reader = csv.reader(txt)
  decode = {rows[0]:rows[1].strip() for rows in reader}

for img_cl in img_classes:
  #load files from step3
  num_vars=0
  with open(eval_file+str(img_cl),"rb") as r:
    for el in ["y","y_k","y_q","y_k_ll","y_q_ll","class_images_dict"]:
      try:
        globals()[el]=pickle.load(r)
        num_vars=num_vars+1
      except EOFError:
        warn("not all variables could be loaded, only "+str(num_vars)+"/6")
        break  
  logger.info("number of variables loaded succesfully:"+str(num_vars)+"/6")

  #prints of most counts labels (predictions)
  with open(save_vars_results+str(img_cl), 'w') as txt:
    txt.write("Single Image Evaluation\n++++++++++++++++++++++++\n")

  logger.info("printing predictions into: "+save_vars_results)
  top=10
  with open(save_vars_results+str(img_cl), 'a') as txt:
    txt.write("Predictions\n-----------\n\n")
    for image_class in class_images_dict:
      for image,imagepath in enumerate(class_images_dict[image_class]):
        txt.write(str(image_class)+""+str(image)+"\nnetwork prediction: "+str(decode[y[image_class][image][0][0][0]])+"\n")
        for j,el in enumerate(k_q):
          txt.write("approach prediction for k="+str(el)+":  ")
          for t in range(top):
            if el not in y_k[image_class][image]:
              warn("no labels for "+test_labels_id[image_class]+"/"+str(image)+"and k="+str(el))
              break
            try:
              txt.write(str(decode[y_k[image_class][image][el][t][0]])+" "+str(y_k[image_class][image][el][t][1])+", ")
            except KeyError:
              warn("not able to decode label "+y_k[image_class][image][el][t][0])
              txt.write(str(y_k[image_class][image][el][t][0])+" "+str(y_k[image_class][image][el][t][1])+", ")
            except IndexError:
              warn("only "+str(t)+" predictions available for "+test_labels_id[image_class]+"/"+str(image)+" (k="+str(el)+"). Change of parameter top may solve this")
              break
          txt.write("\napproach prediction for q%="+str(el)+": ")
          for t in range(top):
            if el not in y_q[image_class][image]:
              warn("no labels for "+test_labels_id[image_class]+"/"+str(image)+"and q%="+str(el))
              break
            try:  
              txt.write(str(decode[y_q[image_class][image][el][t][0]])+" "+str(y_q[image_class][image][el][t][1])+", ")
            except KeyError:
              warn("not able to decode label "+y_q[image_class][image][el][t][0])
              txt.write(str(y_q[image_class][image][el][t][0])+" "+str(y_q[image_class][image][el][t][1])+", ")
            except IndexError:
              warn("only "+str(t)+" predictions available for "+test_labels_id[image_class]+"/"+str(image)+" (q%="+str(el)+") Change of parameter top may solve this")
              break
          txt.write("\n")
        txt.write("\n") 
    txt.write("k    |image name          |    y    |predict k|correct?|predict q%|correct?\n-----|--------------------|---------|---------|--------|----------|--------\n")
    for el in k_q:
      for image_class in class_images_dict:
        for image, imagepath in enumerate(class_images_dict[image_class]):
          txt.write("k="+'{:3}'.format(el)+"|"+"{:20}".format(image)+"|"+str(y[image_class][image][0][0][0])+"|")
          try:
            txt.write("{:9}".format(y_k[image_class][image][el][0][0])+"|")
            if (y[image_class][image][0][0][0]==y_k[image_class][image][el][0][0]):
              txt.write("  True  |")
            else:
              txt.write("  False |")
          except IndexError:
            warn("no prediction available for "+test_labels_id[image_class]+"/"+str(image)+" (k="+str(el)+")")
            txt.write("{:9}".format(str(y_k[image_class][image][el]))+"|"+"  False |")
          try:
            txt.write('{:10}'.format(y_q[image_class][image][el][0][0])+"|")
            if (y[image_class][image][0][0][0]==y_q[image_class][image][el][0][0]):
              txt.write("  True  \n")
            else:
              txt.write("  False \n")
          except IndexError:
            warn("no prediction available for "+test_labels_id[image_class]+"/"+str(image)+" (q%="+str(el)+")")
            txt.write("{:10}".format(str(y_q[image_class][image][el]))+"|"+"  False |\n")

  #prints of most counts labels (predictions) of last layer
  logger.info("printing predictions of last layer into: "+save_vars_results)
  top=10
  with open(save_vars_results+str(img_cl), 'a') as txt:
    txt.write("\nPredictions of last layer\n-------------------------\n\n")
    for image_class in class_images_dict:
      for image, imagepath in enumerate(class_images_dict[image_class]):
        txt.write(str(image_class)+""+str(image)+"\nnetwork prediction: "+str(decode[y[image_class][image][0][0][0]])+"\n")
        for j,el in enumerate(k_q):
          txt.write("approach prediction for k="+str(el)+":  ")
          for t in range(top):
            if el not in y_k_ll[image_class][image]:
              warn("no labels for "+test_labels_id[image_class]+"/"+str(image)+"and k="+str(el))
              break
            try:
              txt.write(str(decode[y_k_ll[image_class][image][el][t][0]])+" "+str(y_k_ll[image_class][image][el][t][1])+", ")
            except KeyError:
              warn("not able to decode label "+y_k_ll[image_class][image][el][t][0])
              txt.write(str(y_k_ll[image_class][image][el][t][0])+" "+str(y_k_ll[image_class][image][el][t][1])+", ")
            except IndexError:
              warn("only "+str(t)+" predictions available for "+test_labels_id[image_class]+"/"+str(image)+" (k="+str(el)+"). Change of parameter top may solve this")
              break
          txt.write("\napproach prediction for q%="+str(el)+": ")
          for t in range(top):
            if el not in y_q_ll[image_class][image]:
              warn("no labels for "+test_labels_id[image_class]+"/"+str(image)+"and q%="+str(el))
              break
            try:  
              txt.write(str(decode[y_q_ll[image_class][image][el][t][0]])+" "+str(y_q_ll[image_class][image][el][t][1])+", ")
            except KeyError:
              warn("not able to decode label "+y_q_ll[image_class][image][el][t][0])
              txt.write(str(y_q_ll[image_class][image][el][t][0])+" "+str(y_q_ll[image_class][image][el][t][1])+", ")
            except IndexError:
              warn("only "+str(t)+" predictions available for "+test_labels_id[image_class]+"/"+str(image)+" (q%="+str(el)+") Change of parameter top may solve this")
              break
          txt.write("\n")
        txt.write("\n") 
    txt.write("k    |image name          |    y    |predict k|correct?|predict q%|correct?\n-----|--------------------|---------|---------|--------|----------|--------\n")
    for el in k_q:
      for image_class in class_images_dict:
        for image, imagepath in enumerate(class_images_dict[image_class]):
          txt.write("k="+'{:3}'.format(el)+"|"+"{:20}".format(image)+"|"+str(y[image_class][image][0][0][0])+"|")
          try:
            txt.write("{:9}".format(y_k_ll[image_class][image][el][0][0])+"|")
            if (y[image_class][image][0][0][0]==y_k_ll[image_class][image][el][0][0]):
              txt.write("  True  |")
            else:
              txt.write("  False |")
          except IndexError:
            warn("no prediction available for "+str(test_labels_id[image_class])+"/"+str(image)+" (k="+str(el)+")")
            txt.write("{:9}".format(str(y_k_ll[image_class][image][el]))+"|"+"  False |")
          try:
            txt.write('{:10}'.format(y_q_ll[image_class][image][el][0][0])+"|")
            if (y[image_class][image][0][0][0]==y_q_ll[image_class][image][el][0][0]):
              txt.write("  True  \n")
            else:
              txt.write("  False \n")
          except IndexError:
            warn("no prediction available for "+str(test_labels_id[image_class])+"/"+str(image)+" (q%="+str(el)+")")
            txt.write("{:10}".format(str(y_q_ll[image_class][image][el]))+"|"+"  False |\n")