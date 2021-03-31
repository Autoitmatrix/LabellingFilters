"""# Hitrates Plots"""

#imports
import matplotlib.pyplot as plt
import pickle

"""# Input"""
hitrates_file="/content/Results/2021-01-01-0000-HR-VGG16-vars
imclass="TestSet"

"""# Preparation"""
with open(hitrates_file,'rb') as p:
  totalhit_k=pickle.load(p)
  totalhit_q=pickle.load(p)
network=hitrates_file.split('-')[-2]

"""# Plots"""
#plot q
acc_df={}
lineform=['o-b','^-g','*-r','+-y','-c','s-m','D-k','<-w']
for l,el in enumerate(k_q):
    acc_df[el]=pd.DataFrame.from_dict(totalhit_k[el],orient='index')
    plt.plot(acc_df[el],lineform[l],label="k="+str(el))
plt.title("Hits@m of "+imclass+" for different k")
plt.ylabel("Hits@m")
plt.xlabel("m")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(save_dir+"/"+network+"-hitrates-"+imclass+"-q.pdf", bbox_inches='tight')
plt.show()

#plot q%
acc_df={}
for l,el in enumerate(k_q):
    acc_df[el]=pd.DataFrame.from_dict(totalhit_q[el],orient='index')
    plt.plot(acc_df[el], lineform[l] ,label="q%="+str(el))
plt.title("Hits@m of "+imclass+" for different q%")
plt.ylabel("Hits@m")
plt.xlabel("m")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(save_dir+"/"+runid+"-"+network+"-hitrates-"+imclass+'-q.pdf', bbox_inches='tight')
plt.show()