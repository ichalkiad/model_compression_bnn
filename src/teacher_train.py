import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import WallNavDataset
import torch.utils as utils 
import numpy as np
import torch
import cPickle
from sklearn import preprocessing
import random

CUDA_ = True


def get_teacher_dataset(sensor_dimensions,CUDA,model,train_loader,valid_loader,out_train_filename,out_valid_filename,scale=True):

        save_data = []
        for data, target in train_loader:
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data = Variable(data,requires_grad=False)
            target = Variable(target)
            output = model.forward(data)
            #output = model.get_logits(data)
            d = (data.data).cpu().numpy()
            o1 = np.argmax((output.data).cpu().numpy(),axis=1)
            #t = (target.data).cpu().numpy()
            o = np.zeros((len(data), 4))
            o[np.arange(len(data)), o1] = 1
            s = np.concatenate((d,o),axis=1)    #((d,o,np.reshape(t,(len(t),1))),axis=1)            
            save_data.append(s)


        k = np.concatenate(save_data,axis=0)

        if scale:
           scaler = preprocessing.StandardScaler().fit(k[:,0:sensor_dimensions])                                                                                   
           d = scaler.transform(k[:,0:sensor_dimensions])
           k[:,0:sensor_dimensions] = d
        
        
        with open(out_train_filename,'wb') as f:
                torch.save(k,f)

        
        save_data = []
        for data, target in valid_loader:
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data = Variable(data,requires_grad=False)
            target = Variable(target)
            output = model.forward(data)
            #output = model.get_logits(data)
            d_ = (data.data).cpu().numpy()
            if scale:
               d = scaler.transform(d_)
            o1 = np.argmax((output.data).cpu().numpy(),axis=1)
            #t = (target.data).cpu().numpy()
            o = np.zeros((len(data), 4))
            o[np.arange(len(data)), o1] = 1
            s = np.concatenate((d,o),axis=1)  #((d,o,np.reshape(t,(len(t),1))),axis=1)            
            save_data.append(s)
        

        k2 = np.concatenate(save_data,axis=0)
        with open(out_valid_filename,'wb') as f:
                torch.save(k,f)
       

        return k,k2


def train(model, CUDA, train_loader, optimizer, criterion):
        """Train for 1 epoch."""
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA:
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
            #   print(np_loss)

            loss.backward()
            optimizer.step()
            return np_loss


def test(model, test_loader, criterion, CUDA):
        """Evaluate a model.
           Target has to be a number between 0-C for C classes.
        """
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data = Variable(data,requires_grad=False)
            target = Variable(target) 
            output = model(data)
            test_loss += criterion(output, target).data[0]
            # get the index of the max log-probability                                                                              
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

"""
batch_size = 64
sensor_dimensions = 2
train_loader = utils.data.DataLoader(WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=True,sensor_dimensions=sensor_dimensions),batch_size=batch_size, shuffle=True)
test_loader = utils.data.DataLoader(WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=False,sensor_dimensions=sensor_dimensions),batch_size=batch_size, shuffle=True)


#Set up basic network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_features = 2#24
        self.fc1 = nn.Linear(self.input_features,6) #, 72)
        self.fc2 = nn.Linear(6,4)#(72, 107)
        #self.fc3 = nn.Linear(107,108)
        #self.fc4 = nn.Linear(108,124)
        #self.fc5 = nn.Linear(124,4)


    
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        #x = self.fc3(x)
        #x = self.fc4(x)
        #x = self.fc5(x)
        x = F.log_softmax(x,dim=-1)
        
        return x

    def get_logits(self,x):
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        return x



model = Net()
if CUDA_:
   model.cuda()

#Training criterion and strategy
criterion = nn.NLLLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.182)

for epoch in range(1000):  # loop over the dataset multiple times
    train(model, CUDA_, train_loader, optimizer, criterion)

print('Finished Training')

    
print('\nAccuracy:{}'.format(test(model, test_loader, criterion, CUDA_)))



save_data = []
for data, target in train_loader:
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA_:
                data, target = data.cuda(), target.cuda()
            data = Variable(data,requires_grad=False)
            target = Variable(target)
            output = model.forward(data)
            #output = model.get_logits(data)
            d = (data.data).cpu().numpy()
            o1 = np.argmax((output.data).cpu().numpy(),axis=1)
            #t = (target.data).cpu().numpy()
            o = np.zeros((len(data), 4))
            o[np.arange(len(data)), o1] = 1
            s = np.concatenate((d,o),axis=1)    #((d,o,np.reshape(t,(len(t),1))),axis=1)            
            save_data.append(s)

k = np.concatenate(save_data,axis=0)
with open('../data/Wall/train_data_2sensors_teacherLabels.pt','wb') as f:
     torch.save(k,f)

save_data = []
for data, target in test_loader:
            data = data.type('torch.FloatTensor')
            target = target.type('torch.LongTensor')
            if CUDA_:
                data, target = data.cuda(), target.cuda()
            data = Variable(data,requires_grad=False)
            target = Variable(target)
            output = model.forward(data)
            #output = model.get_logits(data)
            d = (data.data).cpu().numpy()
            o1 = np.argmax((output.data).cpu().numpy(),axis=1)
            #t = (target.data).cpu().numpy()
            o = np.zeros((len(data), 4))
            o[np.arange(len(data)), o1] = 1
            s = np.concatenate((d,o),axis=1)  #((d,o,np.reshape(t,(len(t),1))),axis=1)            
            save_data.append(s)

k = np.concatenate(save_data,axis=0)
with open('../data/Wall/test_data_2sensors_teacherLabels.pt','wb') as f:
     torch.save(k,f)



"""
     
