from torch import load
import torch.utils.data as data
import os

class WallNavDataset(data.Dataset):
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

        if self.train:
            self.torch_file = 'train_data_4sensors.pt'
            self.train_data = load(os.path.join(root_dir, self.torch_file),map_location)
        else:
            self.torch_file = 'test_data_4sensors.pt'
            self.test_data = load(os.path.join(root_dir, self.torch_file),map_location)

        self.root_dir = root_dir
        
        

    def __len__(self):
     
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)


    def __getitem__(self, index):

        if self.train:
            features, target = self.train_data[index,0:4], self.train_data[index,4]
        else:
            features, target = self.test_data[index,0:4], self.test_data[index,4]

        return features,target
