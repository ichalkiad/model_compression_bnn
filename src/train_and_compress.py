from sacred.observers import FileStorageObserver,MongoObserver
from sacred import Experiment
import sys
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import WallNavDataset
import torch.utils as utils 
import numpy as np
import torch
import cPickle
from teacher_train import train, test, get_teacher_dataset
from collections import OrderedDict, defaultdict
from bnn_student_train import main as bnn_main
from bnn_student_train import get_batch_indices
from torch.nn.functional import normalize 
from tensorboardX import SummaryWriter
import os
import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam, SGD
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import pyplot as plt
from debug_bnn import wdecay, create_dataset, create_grid
from debug_visualisation import visualise_and_debug, plot_confusion_matrix




"""
Teacher network
"""
class Net(nn.Module):
    def __init__(self,input_feat):
        super(Net, self).__init__()
        self.input_features = input_feat
        self.fc1 = nn.Linear(self.input_features, 72)
        self.fc2 = nn.Linear(72, 107)
        self.fc3 = nn.Linear(107,108)
        self.fc4 = nn.Linear(108,124)
        self.fc5 = nn.Linear(124,4)

    
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = F.log_softmax(x,dim=-1)
        
        return x


"""
Student network
"""
#  
class BNNModel(nn.Module):
    def __init__(self, p, hidden_nodes,output_nodes):
        super(BNNModel, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.fc1 = nn.Linear(p, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.output_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        x = F.softmax(x,dim=-1) 

        return x


"""
Initialisation of globals for model,guide loading.
"""
softplus = nn.Softplus()
hidden_nodes = output_nodes = feature_num = p = N = 0
bnn_model = False
CUDA_ = True

ex = Experiment('BayesianCompression')

@ex.config
def config():
    
    db = ""
    if db=="mongo":
        print("Using mongodb for logging")
        ex.observers.append(MongoObserver.create())
    elif db=="file":
        print("Using local file storage for logging")        
        ex.observers.append(FileStorageObserver.create('SacredRunLog'))
    

    ID = 0 
    CUDA = True 
    """
    Teacher parameters
    """
    batch_size = 64
    sensor_dimensions = 24
    teacher_learning_rate = 0.05
    teacher_criterion_ = "NLLLoss"
    teacher_optimizer_ = "Adagrad"
    epochs = 1500
    log_directory = "/tmp/bayesian_compression_"+str(sensor_dimensions)+"sensors"+str(ID)+"/"
    out_train_filename = log_directory+"train_data_"+str(sensor_dimensions)+"sensors_teacherLabels.pt"
    out_valid_filename = log_directory+"valid_data_"+str(sensor_dimensions)+"sensors_teacherLabels.pt"
    fname = log_directory+"teacher_confusion_mat.pdf"
    fname_teacher = log_directory+"teacher_model.pt"

    """    
    Student parameters
    """
    bnn_batch_size = 4400
    batch_training = False
    hidden_nodes_ = 7
    output_nodes_ = 4
    bnn_learning_rate = 0.1
    num_particles = 1
    rec_step = 10
    bnn_epochs = 1000
    point_pdf_file = log_directory+str(ID)+"_point_PDF.pt" 
    parameter_samples_no = 10000
    """
    Scale dataset for bayesian training
    """
    scale = True
    fname_student = log_directory+"student_confusion_mat.pdf"         
   
"""
Bayesian training requisites.
"""
def custom_step(svi,data,cuda,epoch,rec_step,gradient_norms):
#Customised step() function for monitoring gradients
    
    epoch_loss = svi.loss_and_grads(svi.model, svi.guide, data)
    if epoch % rec_step == 0:
        for name, value in pyro.get_param_store().named_parameters():
            if "weight" in name:
               if cuda:
                  value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().cpu().data.numpy()[0]))
               else:
                  value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().data.numpy()[0]))

    # get active params
    params = pyro.get_param_store().get_active_params()
    # actually perform gradient steps
    # torch.optim objects gets instantiated for any params that haven't been seen yet
    svi.optim(params)
    # zero gradients
    pyro.util.zero_grads(params)
    # mark parameters in the param store as inactive
    pyro.get_param_store().mark_params_inactive(params)

    return epoch_loss



def model(data):
    # Create unit normal priors over the parameters

    if CUDA_:
        mu1 = Variable(torch.zeros(hidden_nodes, p)).cuda()
        sigma1 = Variable(torch.ones(hidden_nodes, p)).cuda()
        bias_mu1 = Variable(torch.zeros(1, hidden_nodes)).cuda()
        bias_sigma1 = Variable(torch.ones(1, hidden_nodes)).cuda()
        mu2 = Variable(torch.zeros(output_nodes, hidden_nodes)).cuda()
        sigma2 = Variable(torch.ones(output_nodes, hidden_nodes)).cuda()
        bias_mu2 = Variable(torch.zeros(1, output_nodes)).cuda()
        bias_sigma2 = Variable(torch.ones(1, output_nodes)).cuda()
    else:
        mu1 = Variable(torch.zeros(hidden_nodes, p))
        sigma1 = Variable(torch.ones(hidden_nodes, p))
        bias_mu1 = Variable(torch.zeros(1, hidden_nodes))
        bias_sigma1 = Variable(torch.ones(1, hidden_nodes))
        mu2 = Variable(torch.zeros(output_nodes, hidden_nodes))
        sigma2 = Variable(torch.ones(output_nodes, hidden_nodes))
        bias_mu2 = Variable(torch.zeros(1, output_nodes))
        bias_sigma2 = Variable(torch.ones(1, output_nodes))


    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)

    priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc2.weight': w_prior2, 'fc2.bias': b_prior2}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", bnn_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    
    with pyro.iarange("map", N, subsample=data):
        x_data = data[:, 0:feature_num]
        y_data = data[:, feature_num:feature_num+4] # feature_num+4: change if not 1-hot
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze()
        pyro.sample("obs",
                    Normal(prediction_mean, Variable(torch.ones(prediction_mean.size())).type_as(data)),
                    obs=y_data.squeeze())


def guide(data):
    if CUDA_:
        w_mu1 = Variable(torch.randn(hidden_nodes, p).cuda(), requires_grad=True)
        w_log_sig1 = Variable((-3.0 * torch.ones(hidden_nodes, p) + 0.05 * torch.randn(hidden_nodes, p)).cuda(), requires_grad=True)
        b_mu1 = Variable(torch.randn(1, hidden_nodes).cuda(), requires_grad=True)
        b_log_sig1 = Variable((-3.0 * torch.ones(1, hidden_nodes) + 0.05 * torch.randn(1, hidden_nodes)).cuda(), requires_grad=True)
        w_mu2 = Variable(torch.randn(output_nodes, hidden_nodes).cuda(), requires_grad=True)
        w_log_sig2 = Variable((-3.0 * torch.ones(output_nodes, hidden_nodes) + 0.05 * torch.randn(output_nodes, hidden_nodes)).cuda(), requires_grad=True)
        b_mu2 = Variable(torch.randn(1, output_nodes).cuda(), requires_grad=True)
        b_log_sig2 = Variable((-3.0 * torch.ones(1, output_nodes) + 0.05 * torch.randn(1, output_nodes)).cuda(), requires_grad=True)
    else:
        w_mu1 = Variable(torch.randn(hidden_nodes, p), requires_grad=True)
        w_log_sig1 = Variable((-3.0 * torch.ones(hidden_nodes, p) + 0.05 * torch.randn(hidden_nodes, p)), requires_grad=True)
        b_mu1 = Variable(torch.randn(1, hidden_nodes), requires_grad=True)
        b_log_sig1 = Variable((-3.0 * torch.ones(1, hidden_nodes) + 0.05 * torch.randn(1, hidden_nodes)), requires_grad=True)
        w_mu2 = Variable(torch.randn(output_nodes, hidden_nodes), requires_grad=True)
        w_log_sig2 = Variable((-3.0 * torch.ones(output_nodes, hidden_nodes) + 0.05 * torch.randn(output_nodes, hidden_nodes)), requires_grad=True)
        b_mu2 = Variable(torch.randn(1, output_nodes), requires_grad=True)
        b_log_sig2 = Variable((-3.0 * torch.ones(1, output_nodes) + 0.05 * torch.randn(1, output_nodes)), requires_grad=True)

    # register learnable params in the param store
    mw_param1 = pyro.param("guide_mean_weight1", w_mu1)
    sw_param1 = softplus(pyro.param("guide_log_sigma_weight1", w_log_sig1))
    mb_param1 = pyro.param("guide_mean_bias1", b_mu1)
    sb_param1 = softplus(pyro.param("guide_log_sigma_bias1", b_log_sig1))
    # gaussian guide distributions for w and b
    w_dist1 = Normal(mw_param1, sw_param1)
    b_dist1 = Normal(mb_param1, sb_param1)
    # register learnable params in the param store
    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))
    # gaussian guide distributions for w and b
    w_dist2 = Normal(mw_param2, sw_param2)
    b_dist2 = Normal(mb_param2, sb_param2)
    
    dists = {'fc1.weight': w_dist1, 'fc1.bias': b_dist1, 'fc2.weight': w_dist2, 'fc2.bias': b_dist2}
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", bnn_model, dists)
    # sample a regressor
    return lifted_module()



    


@ex.automain
def train_and_compress(batch_size,sensor_dimensions,teacher_criterion_,teacher_learning_rate,teacher_optimizer_,CUDA,epochs,out_train_filename,out_valid_filename,bnn_batch_size,bnn_learning_rate,num_particles,rec_step,bnn_epochs,point_pdf_file,log_directory,hidden_nodes_,output_nodes_,parameter_samples_no,batch_training,scale,fname,fname_teacher,fname_student):

    global feature_num
    feature_num = sensor_dimensions
    global p
    p = sensor_dimensions
    global CUDA_
    CUDA_ = CUDA
    global hidden_nodes
    hidden_nodes = hidden_nodes_
    global output_nodes
    output_nodes = output_nodes_
    """
    Set up models.
    """
    teacher_model = Net(sensor_dimensions)
    if CUDA:
        teacher_model.cuda()
    
    if teacher_criterion_=="NLLLoss":
        teacher_criterion = nn.NLLLoss()
    else:
        print("Not implemented criterion - exit")
        sys.exit(1)
    if teacher_optimizer_=="Adagrad":
       teacher_optimizer = optim.Adagrad(teacher_model.parameters(), lr=teacher_learning_rate)
    else:
        print("Not implemented optimizer - exit")
        sys.exit(1)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    bnn_model_ = BNNModel(sensor_dimensions,hidden_nodes_,output_nodes_)
    if CUDA:
        bnn_model_.cuda()
    global bnn_model
    bnn_model = bnn_model_
      
    """
    Set up data loaders for teacher training.
    """
    trainSet = WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=True,sensor_dimensions=sensor_dimensions)
    validSet = WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=False,valid=True,sensor_dimensions=sensor_dimensions)
    train_loader = utils.data.DataLoader(trainSet,batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(validSet,batch_size=batch_size, shuffle=True)


    """
    Train teacher network and evaluate accuracy on test set.
    """
    for epoch in xrange(epochs):
        train(teacher_model, CUDA, train_loader, teacher_optimizer, teacher_criterion)
    print('Finished teacher training\n')
    print('Accuracy - validation set:{}\n'.format(test(teacher_model, valid_loader, teacher_criterion, CUDA)))

    """
    Save teacher model for evaluation
    """
    torch.save(teacher_model.state_dict(), fname_teacher)
    

    """
    Get teacher data for bayesian compression.
    """
    train_teacher, valid_teacher = get_teacher_dataset(sensor_dimensions,CUDA,teacher_model,train_loader,valid_loader,out_train_filename,out_valid_filename,scale=True)
    global N
    N = train_teacher.shape[0]
    if batch_training==False:
        bnn_batch_size = N

    """
    Save teacher confusion matrix
    """
    names = ['Slight-Right-Turn','Sharp-Right-Turn','Move-Forward','Slight-Left-Turn']
    cnf_matrix = confusion_matrix(np.argmax(validSet.validS()[:,-5:-1],axis=1), np.argmax(valid_teacher[:,-5:-1],axis=1))  
    fig1 = plt.figure(1, figsize=(15, 7))
    plot_confusion_matrix(cnf_matrix, classes=names,normalize=False,title='Teacher confusion matrix, without normalization')
    plt.savefig(fname) 

        

    print("Teacher network architecture:")
    for idx, module in enumerate(teacher_model.cpu().named_modules()):
        print(module)
        break


    """
    Bayesian compression.
    """
    #Train and valid data scaled and labeled by teacher model.
    train_data = Variable(torch.Tensor(train_teacher))
    valid_data = Variable(torch.Tensor(valid_teacher))


    """
    Train and log Bayesian Neural Network.
    """
    log = dict()
    gradient_norms = defaultdict(list)

    
    bnn_main(model,guide,custom_step,rec_step,num_particles,log,gradient_norms,CUDA,bnn_epochs,bnn_batch_size, train_data,valid_data,softplus,bnn_model,sensor_dimensions,train_teacher.shape[0],debug=False,logdir=log_directory,point_pdf_file=point_pdf_file,learning_rate=bnn_learning_rate,n_samples_=parameter_samples_no,log_directory=log_directory,fname_student=fname_student)

    
    print("Student network architecture:")
    for idx, module in enumerate(bnn_model_.cpu().named_modules()):
        print(module)
        break
