import torch
import numpy as np
from keras import utils as k_np
from sklearn import preprocessing
import random
from matplotlib import pyplot as plt

bins=[0,1,2,3,4]
names = ['Slight-Right-Turn','Sharp-Right-Turn','Move-Forward','Slight-Left-Turn']


#24 sensors
with open('usableData/train_data_24sensors.pt','rb') as f:
    data = torch.load(f)

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
_, _, histogram = plt.hist(data[:,24], bins=bins, density=False, align='left',rwidth=0.3)
ax.set_xticks(bins)
ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
plt.savefig("usableData/train_data_class_distrib.pdf") 
    
    
#Scale data
#scaler = preprocessing.StandardScaler().fit(data[:,0:24])

with open('usableData/test_data_24sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
tst_idx = random.sample(range(data_r), int(np.round(0.3*data_r)))
data_save = np.zeros((len(tst_idx),data_c))

#data_test_scaled = scaler.transform(data[tst_idx,0:24])
data_save[:,0:24] = data[tst_idx,0:24] #data_test_scaled
data_save[:,24] = data[tst_idx,24] #k_np.to_categorical(data[tst_idx,24],4)
with open('usableData/test_data_24sensors_1dimTarget_30pc.pt','wb') as f:
    torch.save(data_save,f)

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
_, _, histogram = plt.hist(data[tst_idx,24], bins=bins, density=False, align='left',rwidth=0.3)
ax.set_xticks(bins)
ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
plt.savefig("usableData/test_data_class_distrib.pdf") 

    
data_save = np.zeros((data_r-len(tst_idx),data_c))
#Scale data
val_idx = np.setdiff1d(np.arange(0,data_r,1),tst_idx)
#data_val_scaled = scaler.transform(data[val_idx,0:24])
data_save[:,0:24] = data[val_idx,0:24] #data_val_scaled
data_save[:,24] = data[val_idx,24] #k_np.to_categorical(data[val_idx,24],4)
with open('usableData/valid_data_24sensors_1dimTarget_70pc.pt','wb') as f:
    torch.save(data_save,f)

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
_, _, histogram = plt.hist(data[val_idx,24], bins=bins, density=False, align='left',rwidth=0.3)
ax.set_xticks(bins)
ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
plt.savefig("usableData/valid_data_class_distrib.pdf") 
    

with open('usableData/valid_data_24sensors_indices.pt','wb') as f:
    torch.save(val_idx,f)
with open('usableData/test_data_24sensors_indices.pt','wb') as f:
    torch.save(tst_idx,f)




#4 sensors
with open('usableData/train_data_4sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]


#Scale data
#scaler = preprocessing.StandardScaler().fit(data[:,0:4])

with open('usableData/test_data_4sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
tst_idx = random.sample(range(data_r), int(np.round(0.3*data_r)))
data_save = np.zeros((len(tst_idx),data_c))
#data_test_scaled = scaler.transform(data[tst_idx,0:4])
data_save[:,0:4] = data[tst_idx,0:4] #data_test_scaled
data_save[:,4] = data[tst_idx,4] #k_np.to_categorical(data[tst_idx,4],4)
with open('usableData/test_data_4sensors_1dimTarget_30pc.pt','wb') as f:
    torch.save(data_save,f)

    
data_save = np.zeros((data_r-len(tst_idx),data_c))
#Scale data
val_idx = np.setdiff1d(np.asarray(range(data_r)),tst_idx)
#data_val_scaled = scaler.transform(data[val_idx,0:4])
data_save[:,0:4] = data[val_idx,0:4] #data_val_scaled
data_save[:,4] = data[val_idx,4] #k_np.to_categorical(data[val_idx,4],4)
with open('usableData/valid_data_4sensors_1dimTarget_70pc.pt','wb') as f:
    torch.save(data_save,f)


with open('usableData/valid_data_4sensors_indices.pt','wb') as f:
    torch.save(val_idx,f)
with open('usableData/test_data_4sensors_indices.pt','wb') as f:
    torch.save(tst_idx,f)

    

#2 sensors
with open('usableData/train_data_2sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
#scaler = preprocessing.StandardScaler().fit(data[:,0:2])

with open('usableData/test_data_2sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
tst_idx = random.sample(range(data_r), int(np.round(0.3*data_r)))
data_save = np.zeros((len(tst_idx),data_c))
#data_test_scaled = scaler.transform(data[tst_idx,0:2])
data_save[:,0:2] = data[tst_idx,0:2] #data_test_scaled
data_save[:,2] = data[tst_idx,2] #k_np.to_categorical(data[tst_idx,2],4)
with open('usableData/test_data_2sensors_1dimTarget_30pc.pt','wb') as f:
    torch.save(data_save,f)

data_save = np.zeros((data_r-len(tst_idx),data_c))
#Scale data
val_idx = np.setdiff1d(np.asarray(range(data_r)),tst_idx)
#data_val_scaled = scaler.transform(data[val_idx,0:2])
data_save[:,0:2] = data[val_idx,0:2] #data_val_scaled
data_save[:,2] = data[val_idx,2] #k_np.to_categorical(data[val_idx,2],4)
with open('usableData/valid_data_2sensors_1dimTarget_70pc.pt','wb') as f:
    torch.save(data_save,f)


with open('usableData/valid_data_2sensors_indices.pt','wb') as f:
    torch.save(val_idx,f)
with open('usableData/test_data_2sensors_indices.pt','wb') as f:
    torch.save(tst_idx,f)


