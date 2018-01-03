import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import WallNavDataset
import torch.utils as utils 
import numpy as np
import torch
CUDA_ = True



def train(model, CUDA_, train_loader, optimizer, criterion):
        """Train for 1 epoch."""
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA_:
               data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # print(type(loss))                                                                                                             
            np_loss = loss.cpu().data.numpy()
            if np.isnan(np_loss):
                print('stopping training - nan loss')
                return -1
            elif loss.cpu().data.numpy()[0] > 100000:
                print('Qutting, loss too high', np_loss)
                return -1
            #else:
            #    print(np_loss)

            loss.backward()
            optimizer.step()
            

def test(model, test_loader, criterion, CUDA_):
        """Evaluate a model."""
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA_:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data = Variable(data)
            target = Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data[0]
            # get the index of the max log-probability                                                                              
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy


batch_size = 64
train_loader = utils.data.DataLoader(WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=True),batch_size=batch_size, shuffle=True)
test_loader = utils.data.DataLoader(WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=False),batch_size=batch_size, shuffle=True)


#Set up basic network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_features = 2
        self.fc1 = nn.Linear(self.input_features, 6)
        self.fc2 = nn.Linear(6, 4)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=-1)
        
        return x

    def get_logits(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



model = Net()
if CUDA_:
   model.cuda()

#Training criterion and strategy
criterion = nn.NLLLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.05)

for epoch in range(500):  # loop over the dataset multiple times
    train(model, CUDA_, train_loader, optimizer, criterion)
    
print('\nAccuracy:{}'.format(test(model, test_loader, criterion, CUDA_)))

print('Finished Training')
