import torch
import numpy as np
from keras import utils as k_np
from sklearn import preprocessing


#24 sensors
with open('train_data_24sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

data_save = np.zeros((data_r,data_c+3))
#Scale data
scaler = preprocessing.StandardScaler().fit(data[:,0:24])
data_train_scaled = scaler.transform(data[:,0:24])
data_save[:,0:24] = data_train_scaled
data_save[:,24:28] = k_np.to_categorical(data[:,24],4)
with open('train_data_24sensors_1hot_scaled.pt','wb') as f:
    torch.save(data_save,f)


with open('test_data_24sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

data_save = np.zeros((data_r,data_c+3))
#Scale data
data_test_scaled = scaler.transform(data[:,0:24])
data_save[:,0:24] = data_test_scaled
data_save[:,24:28] = k_np.to_categorical(data[:,24],4)
with open('test_data_24sensors_1hot_scaled.pt','wb') as f:
    torch.save(data_save,f)


#4 sensors
with open('train_data_4sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

data_save = np.zeros((data_r,data_c+3))
#Scale data
scaler = preprocessing.StandardScaler().fit(data[:,0:4])
data_train_scaled = scaler.transform(data[:,0:4])
data_save[:,0:4] = data_train_scaled
data_save[:,4:8] = k_np.to_categorical(data[:,4],4)
with open('train_data_4sensors_1hot_scaled.pt','wb') as f:
    torch.save(data_save,f)

with open('test_data_4sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

data_save = np.zeros((data_r,data_c+3))
#Scale data
data_test_scaled = scaler.transform(data[:,0:4])
data_save[:,0:4] = data_test_scaled
data_save[:,4:8] = k_np.to_categorical(data[:,4],4)
with open('test_data_4sensors_1hot_scaled.pt','wb') as f:
    torch.save(data_save,f)

    
#2 sensors
with open('train_data_2sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

data_save = np.zeros((data_r,data_c+3))
#Scale data
scaler = preprocessing.StandardScaler().fit(data[:,0:2])
data_train_scaled = scaler.transform(data[:,0:2])
data_save[:,0:2] = data_train_scaled
data_save[:,2:6] = k_np.to_categorical(data[:,2],4)
with open('train_data_2sensors_1hot_scaled.pt','wb') as f:
    torch.save(data_save,f)


with open('test_data_2sensors.pt','rb') as f:
    data = torch.load(f)
data_r = data.shape[0]
data_c = data.shape[1]

data_save = np.zeros((data_r,data_c+3))
#Scale data
data_test_scaled = scaler.transform(data[:,0:2])
data_save[:,0:2] = data_test_scaled
data_save[:,2:6] = k_np.to_categorical(data[:,2],4)
with open('test_data_2sensors_1hot_scaled.pt','wb') as f:
    torch.save(data_save,f)
