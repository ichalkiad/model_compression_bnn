import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize 
from torch.autograd import Variable
import cPickle
import torch.utils as utils
import WallNavDataset
import torch.optim as optim


def train(model, CUDA_, train_loader, optimizer, criterion):
        """Train for 1 epoch."""
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.type('torch.FloatTensor')
            target = target.type('torch.FloatTensor')
            if CUDA_:
               data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            np_loss = loss.cpu().data.numpy()
            if np.isnan(np_loss):
                print('stopping training - nan loss')
                return -1
            elif loss.cpu().data.numpy()[0] > 100000:
                print('Qutting, loss too high', np_loss)
                return -1
            #else:                                                                                                                                                                                                             #   print(np_loss)                                                                                                                                                                                      
            loss.backward()
            optimizer.step()
#            print("epoch loss {}".format(np_loss))
            print(np_loss)

        return 0



# NN 
class RegressionModel(nn.Module):
    def __init__(self, p, hidden_nodes,output_nodes):
        super(RegressionModel, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.fc1 = nn.Linear(p, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.output_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x


    

p = 24  # number of features
hidden_nodes = 5433
output_nodes = 4
regression_model = RegressionModel(p,hidden_nodes,output_nodes)


def main(args):
    global CUDA_    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(regression_model.parameters(), lr=0.000001) 
    train_loader = utils.data.DataLoader(WallNavDataset.WallNavLogitsDataset(root_dir='../data/Wall', train=True),batch_size=args.batch_size, shuffle=True)
    if args.cuda:
        # make tensors and modules CUDA
        CUDA_ = True
        regression_model.cuda()
    for j in range(args.num_epochs):
        train(regression_model, CUDA_, train_loader, optimizer, criterion)
               


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    CUDA_ = False

    main(args)
