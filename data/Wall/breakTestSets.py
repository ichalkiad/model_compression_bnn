import torch
import numpy as np
from keras import utils as k_np
from sklearn import preprocessing
import random

#24 sensors
with open('train_data_24sensors.pt','rb') as f:
    data = torch.load(f)
#Scale data
scaler = preprocessing.StandardScaler().fit(data[:,0:24])

with open('test_data_24sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
tst_idx = random.sample(range(data_r), int(np.round(0.3*data_r)))
data_save = np.zeros((len(tst_idx),data_c+3))

data_test_scaled = scaler.transform(data[tst_idx,0:24])
data_save[:,0:24] = data_test_scaled
data_save[:,24:28] = k_np.to_categorical(data[tst_idx,24],4)
with open('test_data_24sensors_1hot_scaled_30pc.pt','wb') as f:
    torch.save(data_save,f)

data_save = np.zeros((data_r-len(tst_idx),data_c+3))
#Scale data
val_idx = np.setdiff1d(np.arange(0,data_r,1),tst_idx)
data_val_scaled = scaler.transform(data[val_idx,0:24])
data_save[:,0:24] = data_val_scaled
data_save[:,24:28] = k_np.to_categorical(data[val_idx,24],4)
with open('valid_data_24sensors_1hot_scaled_70pc.pt','wb') as f:
    torch.save(data_save,f)



#4 sensors
with open('train_data_4sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
scaler = preprocessing.StandardScaler().fit(data[:,0:4])

with open('test_data_4sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]


#Scale data
tst_idx = random.sample(range(data_r), int(np.round(0.3*data_r)))
data_save = np.zeros((len(tst_idx),data_c+3))
data_test_scaled = scaler.transform(data[tst_idx,0:4])
data_save[:,0:4] = data_test_scaled
data_save[:,4:8] = k_np.to_categorical(data[tst_idx,4],4)
with open('test_data_4sensors_1hot_scaled_30pc.pt','wb') as f:
    torch.save(data_save,f)

data_save = np.zeros((data_r-len(tst_idx),data_c+3))
#Scale data
val_idx = np.setdiff1d(np.asarray(range(data_r)),tst_idx)
data_val_scaled = scaler.transform(data[val_idx,0:4])
data_save[:,0:4] = data_val_scaled
data_save[:,4:8] = k_np.to_categorical(data[val_idx,4],4)
with open('valid_data_4sensors_1hot_scaled_70pc.pt','wb') as f:
    torch.save(data_save,f)

    
#2 sensors
with open('train_data_2sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
scaler = preprocessing.StandardScaler().fit(data[:,0:2])
with open('test_data_2sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

#Scale data
tst_idx = random.sample(range(data_r), int(np.round(0.3*data_r)))
data_save = np.zeros((len(tst_idx),data_c+3))
data_test_scaled = scaler.transform(data[tst_idx,0:2])
data_save[:,0:2] = data_test_scaled
data_save[:,2:6] = k_np.to_categorical(data[tst_idx,2],4)
with open('test_data_2sensors_1hot_scaled_30pc.pt','wb') as f:
    torch.save(data_save,f)

data_save = np.zeros((data_r-len(tst_idx),data_c+3))
#Scale data
val_idx = np.setdiff1d(np.asarray(range(data_r)),tst_idx)
data_val_scaled = scaler.transform(data[val_idx,0:2])
data_save[:,0:2] = data_val_scaled
data_save[:,2:6] = k_np.to_categorical(data[val_idx,2],4)
with open('valid_data_2sensors_1hot_scaled_70pc.pt','wb') as f:
    torch.save(data_save,f)
