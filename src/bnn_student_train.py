import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize 
from torch.autograd import Variable
import pyro
from pyro.distributions import Normal, Bernoulli 
from pyro.infer import SVI
from pyro.optim import Adam, SGD, ClippedAdam
import cPickle
from tensorboardX import SummaryWriter
import json
import time
import datetime
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from debug_bnn import wdecay, create_dataset, create_grid
from debug_visualisation import visualise_and_debug,plot_confusion_matrix

"""
hidden_nodes  = 7


#hidden_nodes2 = 1500

#hidden_nodes3 = 100000

output_nodes  = 4
feature_num   = 24
softplus      = nn.Softplus()
p             = feature_num
learning_rate = 0.1
num_particles = 1
rec_step = 1000

log = dict()
gradient_norms = defaultdict(list)


def custom_step(svi,data,cuda,epoch,rec_step,gradient_norms):
#Custom step() for monitoring gradients
    
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


# NN 
class RegressionModel(nn.Module):
    def __init__(self, p, hidden_nodes,output_nodes):
        super(RegressionModel, self).__init__()
        self.hidden_nodes = hidden_nodes
#        self.hidden_nodes2 = hidden_nodes2
"""
#        self.hidden_nodes3 = hidden_nodes3
"""
        self.output_nodes = output_nodes
        self.fc1 = nn.Linear(p, self.hidden_nodes)
#        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes2)
"""
#        self.fc3 = nn.Linear(self.hidden_nodes2, self.hidden_nodes3)
"""
        self.fc4 = nn.Linear(self.hidden_nodes, self.output_nodes)

    def forward(self, x):
        x = self.fc1(x)
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
#        x = F.log_softmax(x,dim=-1) 

        return x



def model(data):
    # Create unit normal priors over the parameters
    
    if CUDA_:
        mu1 = Variable(torch.zeros(hidden_nodes, p)).cuda()
        sigma1 = Variable(torch.ones(hidden_nodes, p)).cuda()
        bias_mu1 = Variable(torch.zeros(1, hidden_nodes)).cuda()
        bias_sigma1 = Variable(torch.ones(1, hidden_nodes)).cuda()
"""
#        mu2 = Variable(torch.zeros(hidden_nodes2, hidden_nodes)).cuda()
#        sigma2 = Variable(torch.ones(hidden_nodes2, hidden_nodes)).cuda()
#        bias_mu2 = Variable(torch.zeros(1, hidden_nodes2)).cuda()
#        bias_sigma2 = Variable(torch.ones(1, hidden_nodes2)).cuda()
        
#        mu3 = Variable(torch.zeros(hidden_nodes3, hidden_nodes2)).cuda()
#        sigma3 = Variable(torch.ones(hidden_nodes3, hidden_nodes2)).cuda()
#        bias_mu3 = Variable(torch.zeros(1, hidden_nodes3)).cuda()
#        bias_sigma3 = Variable(torch.ones(1, hidden_nodes3)).cuda()
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
 #       mu2 = Variable(torch.zeros(hidden_nodes2, hidden_nodes))
 #       sigma2 = Variable(torch.ones(hidden_nodes2, hidden_nodes))
 #       bias_mu2 = Variable(torch.zeros(1, hidden_nodes2))
 #       bias_sigma2 = Variable(torch.ones(1, hidden_nodes2))
        
 #       mu3 = Variable(torch.zeros(hidden_nodes3, hidden_nodes2))
 #       sigma3 = Variable(torch.ones(hidden_nodes3, hidden_nodes2))
 #       bias_mu3 = Variable(torch.zeros(1, hidden_nodes3))
 #       bias_sigma3 = Variable(torch.ones(1, hidden_nodes3))
"""
        mu4 = Variable(torch.zeros(output_nodes, hidden_nodes))
        sigma4 = Variable(torch.ones(output_nodes, hidden_nodes))
        bias_mu4 = Variable(torch.zeros(1, output_nodes))
        bias_sigma4 = Variable(torch.ones(1, output_nodes))


    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)
"""
 #   w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)
    
 #   w_prior3, b_prior3 = Normal(mu3, sigma3), Normal(bias_mu3, bias_sigma3)
"""
    w_prior4, b_prior4 = Normal(mu4, sigma4), Normal(bias_mu4, bias_sigma4)

    #priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc2.weight': w_prior2, 'fc2.bias': b_prior2, 'fc3.weight': w_prior3, 'fc3.bias': b_prior3, 'fc4.weight': w_prior4, 'fc4.bias': b_prior4}
    priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc4.weight': w_prior4, 'fc4.bias': b_prior4}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
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
 #       w_mu2 = Variable(torch.randn(hidden_nodes2, hidden_nodes).cuda(), requires_grad=True)
 #       w_log_sig2 = Variable((-3.0 * torch.ones(hidden_nodes2, hidden_nodes) + 0.05 * torch.randn(hidden_nodes2, hidden_nodes)).cuda(), requires_grad=True)
 #       b_mu2 = Variable(torch.randn(1, hidden_nodes2).cuda(), requires_grad=True)
 #       b_log_sig2 = Variable((-3.0 * torch.ones(1, hidden_nodes2) + 0.05 * torch.randn(1, hidden_nodes2)).cuda(), requires_grad=True)
        
 #       w_mu3 = Variable(torch.randn(hidden_nodes3, hidden_nodes2).cuda(), requires_grad=True)
 #       w_log_sig3 = Variable((-3.0 * torch.ones(hidden_nodes3, hidden_nodes2) + 0.05 * torch.randn(hidden_nodes3, hidden_nodes2)).cuda(), requires_grad=True)
 #       b_mu3 = Variable(torch.randn(1, hidden_nodes3).cuda(), requires_grad=True)
 #       b_log_sig3 = Variable((-3.0 * torch.ones(1, hidden_nodes3) + 0.05 * torch.randn(1, hidden_nodes3)).cuda(), requires_grad=True)
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
  #      w_mu2 = Variable(torch.randn(hidden_nodes2, hidden_nodes), requires_grad=True)
  #      w_log_sig2 = Variable((-3.0 * torch.ones(hidden_nodes2, hidden_nodes) + 0.05 * torch.randn(hidden_nodes2, hidden_nodes)), requires_grad=True)
  #      b_mu2 = Variable(torch.randn(1, hidden_nodes2), requires_grad=True)
  #      b_log_sig2 = Variable((-3.0 * torch.ones(1, hidden_nodes2) + 0.05 * torch.randn(1, hidden_nodes2)), requires_grad=True)
        
  #      w_mu3 = Variable(torch.randn(hidden_nodes3, hidden_nodes2), requires_grad=True)
  #      w_log_sig3 = Variable((-3.0 * torch.ones(hidden_nodes3, hidden_nodes2) + 0.05 * torch.randn(hidden_nodes3, hidden_nodes2)), requires_grad=True)
  #      b_mu3 = Variable(torch.randn(1, hidden_nodes3), requires_grad=True)
  #      b_log_sig3 = Variable((-3.0 * torch.ones(1, hidden_nodes3) + 0.05 * torch.randn(1, hidden_nodes3)), requires_grad=True)
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
#    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
#    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
#    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
#    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))
    # gaussian guide distributions for w and b
#    w_dist2 = Normal(mw_param2, sw_param2)
#    b_dist2 = Normal(mb_param2, sb_param2)
    
    # register learnable params in the param store
#    mw_param3 = pyro.param("guide_mean_weight3", w_mu3)
#    sw_param3 = softplus(pyro.param("guide_log_sigma_weight3", w_log_sig3))
#    mb_param3 = pyro.param("guide_mean_bias3", b_mu3)
#    sb_param3 = softplus(pyro.param("guide_log_sigma_bias3", b_log_sig3))
    # gaussian guide distributions for w and b
#    w_dist3 = Normal(mw_param3, sw_param3)
#    b_dist3 = Normal(mb_param3, sb_param3)
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
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()


"""
# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main(model,guide,custom_step,rec_step,num_particles,log,gradient_norms,args_cuda,args_num_epochs,args_batch_size,data,valid_data,softplus,regression_model,feature_num,N,debug,logdir='/tmp/runs_icha/',point_pdf_file='/tmp/PredictionPDF.pt',learning_rate=None,n_samples_=10,log_directory="./",fname_student="./student_confMat.pdf"):

    #Initialize summary writer for Tensorboard
    writer = SummaryWriter(logdir)
    x_data, y_data = valid_data[:,0:feature_num], valid_data[:,feature_num:feature_num+4] #feature_num+4 : change if not 1-hot

    global CUDA_    
    if args_cuda:
        # make tensors and modules CUDA
        CUDA_ = True
        data = data.cuda()
        valid_data = valid_data.cuda()
        x_data, y_data = x_data.cuda(), y_data.cuda()
        softplus.cuda()
        regression_model.cuda()

    #Monitor model graph
    writer.add_graph(regression_model, data[:, 0:feature_num], verbose=True)

    # instantiate optim and inference objects
    optim = Adam({"lr":learning_rate})
    
    svi = SVI(model, guide, optim, loss="ELBO", num_particles=num_particles)
    
    log["epochs"] = args_num_epochs
    log["batch_size"] = args_batch_size
    log["lr"] = learning_rate
    log["num_particles"] = num_particles

    param_ctrl = defaultdict()
    p = 0
    for j in xrange(args_num_epochs):
        if args_batch_size == N:
            # use the entire data set
            epoch_loss = custom_step(svi,data,args_cuda,j,rec_step,gradient_norms)
        else:
            # mini batch
            epoch_loss = 0.0
            perm = torch.randperm(N) if not args_cuda else torch.randperm(N).cuda()
            # shuffle data
            data = data[perm]
            # get indices of each batch
            all_batches = get_batch_indices(N, args_batch_size)
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_data = data[batch_start: batch_end]
                epoch_loss += custom_step(svi,batch_data,args_cuda,j,rec_step,gradient_norms)
            
        epoch_loss_valid = svi.evaluate_loss(valid_data)   
        if j % rec_step == 0:
            print("Training set: epoch avg loss {}".format(epoch_loss/float(N)))
            writer.add_scalar('data/train_loss_avg', epoch_loss/float(N), j/rec_step)
            print("Validation set: epoch avg loss {}".format(epoch_loss_valid/len(valid_data)))
            writer.add_scalar('data/valid_loss_avg', epoch_loss_valid/float(len(valid_data)), j/rec_step)
            #Monitor gradient norms
            for name, grad_norms in gradient_norms.items():
                writer.add_scalar("gradients/"+name, grad_norms[0], j)
                param_ctrl["gradients_"+name] = grad_norms[0]
            for name, param in regression_model.named_parameters():
                if "bias" in name:
                    pass
                else:
                    param_ctrl[name] = param.clone().cpu().data.numpy()
            with open(log_directory + "paramVIS_"+str(p)+".pt","wb") as f:
                torch.save(param_ctrl,f)
            p = p + 1
            
            param_ctrl.clear()    
        
        #Clear dict for next epoch
        gradient_norms.clear() 

        writer.add_scalar('data/train_loss', epoch_loss, j)
        writer.add_scalar('data/valid_loss', epoch_loss_valid, j)


        
        #Monitor weights and biases
        for name, param in regression_model.named_parameters():
            if 'weight' in name:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
                writer.add_scalar("norm_"+name, param.norm().clone().cpu().data.numpy(), j)
        
        

    """
    Save model for evaluation
    """
    PATH = log_directory + "/bnn_student_model.pt"
    #print(pyro.get_param_store().named_parameters())        
    pyro.get_param_store().save(PATH)

    
    # Validate - test model
    print("Validate trained model...")
    #Number of parameter sampling steps
    n_samples = n_samples_
    y_preds = [] 
    avg_pred = 0.0
    #Create list of float tensors each containing the evaluation of the test data by a sampled BNN
    if debug:
       tst_data, X, Y = create_grid(-12, 12, 50) 
    else:
       if args_cuda:
         minx = np.amin(x_data.cpu().data.numpy())
         maxx = np.amax(x_data.cpu().data.numpy())
       else:
         minx = np.amin(x_data.data.numpy())
         maxx = np.amax(x_data.data.numpy())
       tst_data, X, Y = create_grid(minx-0.001,maxx+0.001, 50)

    if args_cuda:
       tst_data = Variable(torch.Tensor(tst_data)).cuda()


    avg_pred = []
    for i in range(n_samples):
    # guide does not require the data, get "samplepredictions" as in example
        sampled_reg_model = guide(None)
        if debug:
            y_preds.append(sampled_reg_model(Variable(torch.Tensor(tst_data))).data.numpy())            
        else:
           if args_cuda:
              y_preds.append(sampled_reg_model(x_data).cpu().data.numpy())
              #avg_pred.append(sampled_reg_model(tst_data).cpu().data.numpy())
           else:
              y_preds.append(sampled_reg_model(x_data).data.numpy())
              #avg_pred.append(sampled_reg_model(Variable(torch.Tensor(tst_data))).data.numpy())


    y_pred_np = np.asarray(y_preds)
    #avg_pred_np = np.asarray(avg_pred)


    """    
    # Needed for decision surface visualisation
    ap_tst = []
    prob_distr_perPoint_tst = np.zeros((len(tst_data),output_nodes))
    for i in xrange(len(tst_data)):
         prob_distr_perPoint_tst[i,:] = np.sum(avg_pred_np[:,i,:],axis=0)/float(n_samples)
    majority_class_tst = np.argmax(prob_distr_perPoint_tst,axis=1)
    """


    ap = []
    prob_distr_perPoint = [] #np.zeros((len(x_data),output_nodes))
    for i in xrange(len(x_data)):
        prob_distr_perPoint.append(np.sum(y_pred_np[:,i,:],axis=0)/float(n_samples)) 
    with open(point_pdf_file,'wb') as f:
         torch.save(prob_distr_perPoint,f)

    # Accuracy on valid set     
    # Use max of softmax output to get accuracy on valid data    
    majority_class = np.argmax(prob_distr_perPoint,axis=1)
    if args_cuda:
       accuracy = len(np.where(majority_class==np.argmax(y_data.cpu().data.numpy(),axis=1))[0])/float(len(majority_class))*100    
    else:
       accuracy = len(np.where(majority_class==np.argmax(y_data.data.numpy(),axis=1))[0])/float(len(majority_class))*100
    print("( Micro ) Prediction accuracy on validation set is {}".format(accuracy,"%"))

    """
    bins=[0,1,2,3,4]
    names = ['Slight-Right-Turn','Sharp-Right-Turn','Move-Forward','Slight-Left-Turn']
    fig1 = plt.figure(figsize=(15,6))
    ax = fig1.add_subplot(111)
    _, _, histogram = plt.hist(majority_class, bins=bins, density=False, align='left',rwidth=0.3)
    ax.set_xticks(bins)
    ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
    plt.savefig("./micro.pdf")
    """
    
    # Use average accuracy (max of softmax for each sample prediction -> get accuracy -> avg accuracies for all points)  to get accuracy on valid data 
    acc_samples = np.argmax(y_pred_np,axis=2)
    accuracy = 0.0
    for i in xrange(n_samples):
        if args_cuda:
            accuracy += len(np.where(acc_samples[i,:]==np.argmax(y_data.cpu().data.numpy(),axis=1))[0])/float(acc_samples.shape[1])*100    
        else:
            accuracy += len(np.where(acc_samples[i,:]==np.argmax(y_data.data.numpy(),axis=1))[0])/float(acc_samples.shape[1])*100

    print("( Macro ) Prediction accuracy on validation set is {}".format(accuracy/n_samples,"%"))
    """
    fig22 = plt.figure(figsize=(15,6))
    ax = fig22.add_subplot(111)
    _, _, histogram = plt.hist(acc_samples[0,:], bins=bins, density=False, align='left',rwidth=0.3)
    ax.set_xticks(bins)
    ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
    plt.savefig("./macro1.pdf")
    fig3 = plt.figure(figsize=(15,6))
    ax = fig3.add_subplot(111)
    _, _, histogram = plt.hist(acc_samples[100,:], bins=bins, density=False, align='left',rwidth=0.3)
    ax.set_xticks(bins)
    ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
    plt.savefig("./macro2.pdf")
    """
    
    names = ['Slight-Right-Turn','Sharp-Right-Turn','Move-Forward','Slight-Left-Turn']
    cnf_matrix = confusion_matrix(np.argmax(y_data.cpu().data.numpy(),axis=1), majority_class)  
    fig2 = plt.figure(2, figsize=(15, 7))
    plot_confusion_matrix(cnf_matrix, classes=names,normalize=False,title='Student confusion matrix, without normalization')
    plt.savefig(fname_student)
    

    """
    base_pred = np.argmax(sampled_reg_model(Variable(torch.Tensor(tst_data))).data.numpy(),axis=1)
    visualise_and_debug(np.argmax(y_pred_np,axis=2),majority_class_tst,base_pred,data,n_samples,X,Y,predictionPDF='/tmp/PredictionPDF.pt',PDFcenters='/tmp/PDFCenters.pt',debug=debug)    
    """

    writer.close()

"""
def initialize(filename_train,filename_valid,filename_test,feature_num,debug=False):

   data = []
   with open(filename_train,'rb') as f:
"""
#        logits train
"""
        a = torch.load(f)[:,0:32] #!!!!!!!!!!!!!!!!!! for gold standard
        data = Variable(torch.Tensor(a))

   valid_data = []
   with open(filename_valid,'rb') as f: 
        valid_data = Variable(torch.Tensor(torch.load(f))) 
   test_data = []
   with open(filename_test,'rb') as f: 
        test_data = Variable(torch.Tensor(torch.load(f))) 
     
   N = len(data)  # size of data
   regression_model = RegressionModel(feature_num,hidden_nodes,output_nodes)

   if debug:
      xs1, ys1 = create_dataset()
      data1 = np.hstack((xs1,ys1.reshape(len(xs1),1)))
      with open('./tst_data.pt','wb') as f:
          torch.save(data1,f)
      data = Variable(torch.Tensor(data1))


   return data,valid_data,test_data,N,hidden_nodes,output_nodes,softplus,regression_model



if __name__ == '__main__':
    CUDA_ = False

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    log["timestamp"] = st
    log["hidden_nodes"] = hidden_nodes
    xs = None    
    ys = None
    filename_train = '../data/Wall/train_data_24sensors_1hot_scaled_teacherLabels_goldstandard.pt'  #!!!!!!!!!!!!!!!!!!!!1
    filename_valid =  '../data/Wall/valid_data_24sensors_1hot_scaled_70pc.pt'
    filename_test =  '../data/Wall/test_data_24sensors_1hot_scaled_30pc.pt'
    logdir = './runs_icha_n24/'
    
    # Currently no CUDA on debug mode
    debug = False
    data,valid_data,test_data,N,hidden_nodes,output_nodes,softplus,regression_model = initialize(filename_train,filename_valid,filename_test,feature_num,debug)

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    
    main(model,guide,custom_step,rec_step,num_particles,log,gradient_norms,args.cuda,args.num_epochs,args.batch_size,data,valid_data,test_data,softplus,regression_model,feature_num,N,debug,logdir,point_pdf_file="./runs_icha_n24/"+st+"_point_PDF.pt",learning_rate=learning_rate)
    with open("./runs_icha_n24/log_"+st+".json", 'w') as f:
         json.dump(log, f)
"""
