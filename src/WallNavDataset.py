from torch import load
import torch.utils.data as data
import os
import cPickle

class WallNavDataset(data.Dataset):
    """Wall-Following Navigation dataset."""

    def __init__(self, train=True, valid=False, sensor_dimensions=2, root_dir='../data', map_location='cpu',transform=None):
        """
        Args:
            torch_file (string): Path to the torch file with input-labels pairs.
            root_dir (string): Directory with the torch file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.valid = valid
        self.features_no = sensor_dimensions

        if self.train:
            self.torch_file = "usableData/train_data_"+str(self.features_no)+"sensors.pt"
            self.train_data = load(os.path.join(root_dir, self.torch_file),map_location)
        elif self.valid:
            self.torch_file = "usableData/valid_data_"+str(self.features_no)+"sensors_1dimTarget_70pc.pt"
            self.valid_data = load(os.path.join(root_dir, self.torch_file),map_location)
        else:
            self.torch_file = "usableData/test_data_"+str(self.features_no)+"sensors_1dimTarget_30pc.pt"
            self.test_data = load(os.path.join(root_dir, self.torch_file),map_location)
 
        self.root_dir = root_dir
        
        
    def validS(self):
        return self.valid_data
    def trainS(self):
        return self.train_data
    def testS(self):
        return self.test_data

        
    def __len__(self):
     
        if self.train:
           return len(self.train_data)
        elif self.valid:
           return len(self.valid_data)
        else:
           return len(self.test_data)


    def __getitem__(self, index):

        if self.train:
            features, target = self.train_data[index,0:self.features_no], self.train_data[index,self.features_no]
        elif self.valid:
            features, target = self.valid_data[index,0:self.features_no], self.valid_data[index,self.features_no]
        else:
            features, target = self.test_data[index,0:self.features_no], self.test_data[index,self.features_no]

        return features,target



class WallNavLogitsDataset(data.Dataset):
    """Wall-Following Navigation dataset."""

    def __init__(self, train=True, root_dir='../data', map_location='cpu',transform=None):
        """
        Args:
            torch_file (string): Path to the torch file with input-labels pairs.
            root_dir (string): Directory with the torch file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.features_no = 24

        if self.train:
            self.torch_file = 'train_data_24sensors_logits.pt'
            self.train_data = load(os.path.join(root_dir, self.torch_file),map_location)
        else:
            self.torch_file = 'test_data_24sensors_logits.pt'
            self.test_data = load(os.path.join(root_dir, self.torch_file),map_location)

        self.root_dir = root_dir
        
        

    def __len__(self):
     
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)


    def __getitem__(self, index):

        if self.train:
            features, target = self.train_data[index,0:self.features_no], self.train_data[index,self.features_no:self.features_no+4]
        else:
            features, target = self.test_data[index,0:self.features_no], self.test_data[index,self.features_no:self.features_no+4]

        return features,target




