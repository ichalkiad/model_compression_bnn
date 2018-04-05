from sacred.observers import FileStorageObserver
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

from debug_bnn import wdecay, create_dataset, create_grid
from debug_visualisation import visualise_and_debug


"""
Teacher network
"""
class Net(nn.Module):
    def __init__(self,input_feat):
        super(Net, self).__init__()
        self.input_features = input_feat
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


"""
Student network
"""
#  
class BNNModel(nn.Module):
    def __init__(self, p, hidden_nodes,output_nodes):
        super(BNNModel, self).__init__()
        self.hidden_nodes = hidden_nodes
#        self.hidden_nodes2 = hidden_nodes2
        """
        self.hidden_nodes3 = hidden_nodes3
        """
        self.output_nodes = output_nodes
        self.fc1 = nn.Linear(p, self.hidden_nodes)
#        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes2)
        """
        self.fc3 = nn.Linear(self.hidden_nodes2, self.hidden_nodes3)
        """
        self.fc4 = nn.Linear(self.hidden_nodes, self.output_nodes)

    def forward(self, x):
        x = self.fc1(x)
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
#        x = F.log_softmax(x,dim=-1) 

        return x


"""
Initialisation of globals for model,guide loading.
"""
softplus = nn.Softplus()
hidden_nodes = output_nodes = feature_num = p = N = 0
bnn_model = False
CUDA_ = False

ex = Experiment('BayesianCompression')

@ex.config
def config():
    
    print("Using local file storage for logging")        
    ex.observers.append(FileStorageObserver.create('SacredRunLog'))
    

    ID = 0 

    """
    Teacher parameters
    """
    batch_size = 64
    sensor_dimensions = 2
    teacher_learning_rate = 0.05
    teacher_model = Net(sensor_dimensions)
    CUDA = False
    if CUDA:
        model.cuda()
    teacher_criterion = "NLLLoss"
    if teacher_criterion=="NLLLoss":
        teacher_criterion = nn.NLLLoss()
    else:
        print("Not implemented criterion - exit")
        sys.exit(1)
    teacher_optimizer = "Adagrad"
    if teacher_optimizer=="Adagrad":
        teacher_optimizer = optim.Adagrad(teacher_model.parameters(), lr=teacher_learning_rate)
    else:
        print("Not implemented optimizer - exit")
        sys.exit(1)
    epochs = 1500
    net_arch = OrderedDict()
    for idx, module in enumerate(teacher_model.cpu().named_modules()):
        net_arch[idx] = module
        break
   
    log_directory = "/tmp/bayesian_compression_2sensors"+str(ID)+"/"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    out_train_filename = log_directory+"train_data_"+str(sensor_dimensions)+"sensors_teacherLabels.pt"
    out_test_filename = log_directory+"test_data_"+str(sensor_dimensions)+"sensors_teacherLabels.pt"


    """    
    Student parameters
    """
    bnn_batch_size = 2048
    batch_training = False
    hidden_nodes_ = 6
    output_nodes_ = 4
    bnn_model_ = BNNModel(sensor_dimensions,hidden_nodes_,output_nodes_)
    bnn_net_arch = OrderedDict()
    for idx, module in enumerate(bnn_model_.cpu().named_modules()):
        bnn_net_arch[idx] = module
        break
    bnn_learning_rate = 0.1
    num_particles = 1
    rec_step = 10
    bnn_epochs = 1000
    filename_valid =  "../data/Wall/valid_data_"+str(sensor_dimensions)+"sensors_1hot_scaled_70pc.pt"
    filename_test =  "../data/Wall/test_data_"+str(sensor_dimensions)+"sensors_1hot_scaled_30pc.pt"
    point_pdf_file = log_directory+str(ID)+"_point_PDF.pt" 
    parameter_samples_no = 2000





   
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
        """
        mu2 = Variable(torch.zeros(hidden_nodes2, hidden_nodes)).cuda()
        sigma2 = Variable(torch.ones(hidden_nodes2, hidden_nodes)).cuda()
        bias_mu2 = Variable(torch.zeros(1, hidden_nodes2)).cuda()
        bias_sigma2 = Variable(torch.ones(1, hidden_nodes2)).cuda()
        
        mu3 = Variable(torch.zeros(hidden_nodes3, hidden_nodes2)).cuda()
        sigma3 = Variable(torch.ones(hidden_nodes3, hidden_nodes2)).cuda()
        bias_mu3 = Variable(torch.zeros(1, hidden_nodes3)).cuda()
        bias_sigma3 = Variable(torch.ones(1, hidden_nodes3)).cuda()
        """
        mu4 = Variable(torch.zeros(output_nodes, hidden_nodes)).cuda()
        sigma4 = Variable(torch.ones(output_nodes, hidden_nodes)).cuda()
        bias_mu4 = Variable(torch.zeros(1, output_nodes)).cuda()
        bias_sigma4 = Variable(torch.ones(1, output_nodes)).cuda()
    else:
        mu1 = Variable(torch.zeros(hidden_nodes, p))
        sigma1 = Variable(torch.ones(hidden_nodes, p))
        bias_mu1 = Variable(torch.zeros(1, hidden_nodes))
        bias_sigma1 = Variable(torch.ones(1, hidden_nodes))
        """
        mu2 = Variable(torch.zeros(hidden_nodes2, hidden_nodes))
        sigma2 = Variable(torch.ones(hidden_nodes2, hidden_nodes))
        bias_mu2 = Variable(torch.zeros(1, hidden_nodes2))
        bias_sigma2 = Variable(torch.ones(1, hidden_nodes2))
        
        mu3 = Variable(torch.zeros(hidden_nodes3, hidden_nodes2))
        sigma3 = Variable(torch.ones(hidden_nodes3, hidden_nodes2))
        bias_mu3 = Variable(torch.zeros(1, hidden_nodes3))
        bias_sigma3 = Variable(torch.ones(1, hidden_nodes3))
        """
        mu4 = Variable(torch.zeros(output_nodes, hidden_nodes))
        sigma4 = Variable(torch.ones(output_nodes, hidden_nodes))
        bias_mu4 = Variable(torch.zeros(1, output_nodes))
        bias_sigma4 = Variable(torch.ones(1, output_nodes))


    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)
    """
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)
    
    w_prior3, b_prior3 = Normal(mu3, sigma3), Normal(bias_mu3, bias_sigma3)
    """
    w_prior4, b_prior4 = Normal(mu4, sigma4), Normal(bias_mu4, bias_sigma4)

    #priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc2.weight': w_prior2, 'fc2.bias': b_prior2, 'fc3.weight': w_prior3, 'fc3.bias': b_prior3, 'fc4.weight': w_prior4, 'fc4.bias': b_prior4}
    priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc4.weight': w_prior4, 'fc4.bias': b_prior4}
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
        """
        w_mu2 = Variable(torch.randn(hidden_nodes2, hidden_nodes).cuda(), requires_grad=True)
        w_log_sig2 = Variable((-3.0 * torch.ones(hidden_nodes2, hidden_nodes) + 0.05 * torch.randn(hidden_nodes2, hidden_nodes)).cuda(), requires_grad=True)
        b_mu2 = Variable(torch.randn(1, hidden_nodes2).cuda(), requires_grad=True)
        b_log_sig2 = Variable((-3.0 * torch.ones(1, hidden_nodes2) + 0.05 * torch.randn(1, hidden_nodes2)).cuda(), requires_grad=True)
        
        w_mu3 = Variable(torch.randn(hidden_nodes3, hidden_nodes2).cuda(), requires_grad=True)
        w_log_sig3 = Variable((-3.0 * torch.ones(hidden_nodes3, hidden_nodes2) + 0.05 * torch.randn(hidden_nodes3, hidden_nodes2)).cuda(), requires_grad=True)
        b_mu3 = Variable(torch.randn(1, hidden_nodes3).cuda(), requires_grad=True)
        b_log_sig3 = Variable((-3.0 * torch.ones(1, hidden_nodes3) + 0.05 * torch.randn(1, hidden_nodes3)).cuda(), requires_grad=True)
        """
        w_mu4 = Variable(torch.randn(output_nodes, hidden_nodes).cuda(), requires_grad=True)
        w_log_sig4 = Variable((-3.0 * torch.ones(output_nodes, hidden_nodes) + 0.05 * torch.randn(output_nodes, hidden_nodes)).cuda(), requires_grad=True)
        b_mu4 = Variable(torch.randn(1, output_nodes).cuda(), requires_grad=True)
        b_log_sig4 = Variable((-3.0 * torch.ones(1, output_nodes) + 0.05 * torch.randn(1, output_nodes)).cuda(), requires_grad=True)

    else:
        w_mu1 = Variable(torch.randn(hidden_nodes, p), requires_grad=True)
        w_log_sig1 = Variable((-3.0 * torch.ones(hidden_nodes, p) + 0.05 * torch.randn(hidden_nodes, p)), requires_grad=True)
        b_mu1 = Variable(torch.randn(1, hidden_nodes), requires_grad=True)
        b_log_sig1 = Variable((-3.0 * torch.ones(1, hidden_nodes) + 0.05 * torch.randn(1, hidden_nodes)), requires_grad=True)
        """
        w_mu2 = Variable(torch.randn(hidden_nodes2, hidden_nodes), requires_grad=True)
        w_log_sig2 = Variable((-3.0 * torch.ones(hidden_nodes2, hidden_nodes) + 0.05 * torch.randn(hidden_nodes2, hidden_nodes)), requires_grad=True)
        b_mu2 = Variable(torch.randn(1, hidden_nodes2), requires_grad=True)
        b_log_sig2 = Variable((-3.0 * torch.ones(1, hidden_nodes2) + 0.05 * torch.randn(1, hidden_nodes2)), requires_grad=True)
        
        w_mu3 = Variable(torch.randn(hidden_nodes3, hidden_nodes2), requires_grad=True)
        w_log_sig3 = Variable((-3.0 * torch.ones(hidden_nodes3, hidden_nodes2) + 0.05 * torch.randn(hidden_nodes3, hidden_nodes2)), requires_grad=True)
        b_mu3 = Variable(torch.randn(1, hidden_nodes3), requires_grad=True)
        b_log_sig3 = Variable((-3.0 * torch.ones(1, hidden_nodes3) + 0.05 * torch.randn(1, hidden_nodes3)), requires_grad=True)
        """
        w_mu4 = Variable(torch.randn(output_nodes, hidden_nodes), requires_grad=True)
        w_log_sig4 = Variable((-3.0 * torch.ones(output_nodes, hidden_nodes) + 0.05 * torch.randn(output_nodes, hidden_nodes)), requires_grad=True)
        b_mu4 = Variable(torch.randn(1, output_nodes), requires_grad=True)
        b_log_sig4 = Variable((-3.0 * torch.ones(1, output_nodes) + 0.05 * torch.randn(1, output_nodes)), requires_grad=True)


    # register learnable params in the param store
    mw_param1 = pyro.param("guide_mean_weight1", w_mu1)
    sw_param1 = softplus(pyro.param("guide_log_sigma_weight1", w_log_sig1))
    mb_param1 = pyro.param("guide_mean_bias1", b_mu1)
    sb_param1 = softplus(pyro.param("guide_log_sigma_bias1", b_log_sig1))
    # gaussian guide distributions for w and b
    w_dist1 = Normal(mw_param1, sw_param1)
    b_dist1 = Normal(mb_param1, sb_param1)
    """
    # register learnable params in the param store
    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))
    # gaussian guide distributions for w and b
    w_dist2 = Normal(mw_param2, sw_param2)
    b_dist2 = Normal(mb_param2, sb_param2)
    
    # register learnable params in the param store
    mw_param3 = pyro.param("guide_mean_weight3", w_mu3)
    sw_param3 = softplus(pyro.param("guide_log_sigma_weight3", w_log_sig3))
    mb_param3 = pyro.param("guide_mean_bias3", b_mu3)
    sb_param3 = softplus(pyro.param("guide_log_sigma_bias3", b_log_sig3))
    # gaussian guide distributions for w and b
    w_dist3 = Normal(mw_param3, sw_param3)
    b_dist3 = Normal(mb_param3, sb_param3)
    """
    # register learnable params in the param store
    mw_param4 = pyro.param("guide_mean_weight4", w_mu4)
    sw_param4 = softplus(pyro.param("guide_log_sigma_weight4", w_log_sig4))
    mb_param4 = pyro.param("guide_mean_bias4", b_mu4)
    sb_param4 = softplus(pyro.param("guide_log_sigma_bias4", b_log_sig4))
    # gaussian guide distributions for w and b
    w_dist4 = Normal(mw_param4, sw_param4)
    b_dist4 = Normal(mb_param4, sb_param4)
    

    #dists = {'fc1.weight': w_dist1, 'fc1.bias': b_dist1, 'fc2.weight': w_dist2, 'fc2.bias': b_dist2, 'fc3.weight': w_dist3, 'fc3.bias': b_dist3, 'fc4.weight': w_dist4, 'fc4.bias': b_dist4}
    dists = {'fc1.weight': w_dist1, 'fc1.bias': b_dist1, 'fc4.weight': w_dist4, 'fc4.bias': b_dist4}
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", bnn_model, dists)
    # sample a regressor
    return lifted_module()



    


@ex.automain
def train_and_compress(batch_size,sensor_dimensions,teacher_criterion,teacher_optimizer,CUDA,teacher_model,epochs,out_train_filename,out_test_filename,bnn_batch_size,bnn_model_,bnn_learning_rate,num_particles,rec_step,bnn_epochs,filename_valid,filename_test,point_pdf_file,log_directory,hidden_nodes_,output_nodes_,parameter_samples_no,batch_training):

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
    global bnn_model
    bnn_model = bnn_model_
      
    """
    Set up data loaders for teacher training.
    """
    train_loader = utils.data.DataLoader(WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=True,sensor_dimensions=sensor_dimensions),batch_size=batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(WallNavDataset.WallNavDataset(root_dir='../data/Wall', train=False,sensor_dimensions=sensor_dimensions),batch_size=batch_size, shuffle=True)


    """
    Train teacher network and evaluate accuracy on test set.
    """
    for epoch in xrange(epochs):
        train(teacher_model, CUDA, train_loader, teacher_optimizer, teacher_criterion)
    print('Finished teacher training\n')
    print('Accuracy:{}\n'.format(test(teacher_model, test_loader, teacher_criterion, CUDA)))


    """
    Get teacher data for bayesian compression.
    """
    train_teacher,test_teacher = get_teacher_dataset(teacher_model,train_loader,test_loader,out_train_filename,out_test_filename)
    global N
    N = train_teacher.shape[0]
    if batch_training==False:
        bnn_batch_size = N

    """
    Bayesian compression.
    """

    """
    Load validation and test sets.
    """
    valid_data = []
    with open(filename_valid,'rb') as f: 
        valid_data = Variable(torch.Tensor(torch.load(f))) 
    test_data = []
    with open(filename_test,'rb') as f: 
        test_data = Variable(torch.Tensor(torch.load(f)))  


    """
    Train and log Bayesian Neural Network.
    """
    log = dict()
    gradient_norms = defaultdict(list)

    bnn_main(model,guide,custom_step,rec_step,num_particles,log,gradient_norms,CUDA,bnn_epochs,bnn_batch_size, Variable(torch.Tensor(train_teacher)),valid_data,test_data,softplus,bnn_model_,sensor_dimensions,train_teacher.shape[0],debug=False,logdir=log_directory,point_pdf_file=point_pdf_file,learning_rate=bnn_learning_rate,n_samples_=parameter_samples_no)

