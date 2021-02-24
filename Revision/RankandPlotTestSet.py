import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import rankdata

now = datetime.utcnow()+timedelta(hours=1)
runid = now.strftime("%Y-%m-%d-%H%M")
save_dir = r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Pickles"

q=[1,2,3,4,5,10,25]
network="VGG16"
pickle_file=r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Pickles\2021-02-23-1225-VGG16-PH-vars"

with open (pickle_file+"-names", "rb") as p:
    varnames=pickle.load(p)

with open(pickle_file, "rb") as p:
  for var in varnames:
    globals()[var]=pickle.load(p)


#ranking of classes according to hit@k
rank_k=0
class_ranks={}
for i in range(len(hit)):
    class_ranks[i]={}
for i,el in enumerate(q):
    temp_list= []
    for imgclass in hit:
        temp_list.append(hit[imgclass][el][rank_k])
    temp_ranks=rankdata(temp_list,'min')#'ordinal'?
    for j, rank in enumerate(temp_ranks):
        class_ranks[j]["q="+str(el)]=rank
    temp_list = []
    for imgclass in hit_per:
        temp_list.append(hit_per[imgclass][el][rank_k])
    temp_ranks = rankdata(temp_list,'min')
    for j, rank in enumerate(temp_ranks):
        class_ranks[j]["q%="+str(el)] = rank

#sort classes according to mean rank over q and q%:
sortvalue={}
for i in class_ranks:
    sortvalue[i]= np.mean(list(class_ranks[i].values()))
class_rank=rankdata(list(sortvalue.values()),'ordinal')

#best and worst class
best=np.argmax(class_rank)
worst=np.argmin(class_rank)

id_classname={}
for el in paths:
    id_classname[el]=paths[el][el][0].split('/')[-2]

print(id_classname[best]) # website
print(id_classname[worst]) # labrador


#plot q
imclass="TestSet"
acc_df={}
lineform=['o-b','^-g','*-r','+-y','-c','s-m','D-k','<-w']
for l,el in enumerate(q):
    acc_df[el]=pd.DataFrame.from_dict(totalhit[el],orient='index')[0:51]
    plt.plot(acc_df[el],lineform[l],label="q="+str(el))
plt.title("Hits@k of "+imclass+" for different q")
plt.ylabel("Hits@k")
plt.xlabel("k")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+"-q.pdf", bbox_inches='tight')
plt.show()

#plot q%
acc_df={}
for l,el in enumerate(q):
    acc_df[el]=pd.DataFrame.from_dict(totalhit_per[el],orient='index')[0:51]
    plt.plot(acc_df[el], lineform[l] ,label="q%="+str(el))
plt.title("Hits@k of "+imclass+" for different q%")
plt.ylabel("Hits@k")
plt.xlabel("k")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+'-qper.pdf', bbox_inches='tight')
plt.show()
