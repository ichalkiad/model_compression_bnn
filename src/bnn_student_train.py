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
from pyro.optim import Adam, SGD
import cPickle
from tensorboardX import SummaryWriter

from debug_bnn import wdecay, create_dataset, create_grid
from debug_visualisation import visualise_and_debug


hidden_nodes = 100
hidden_nodes2 = 100
hidden_nodes3 = 100
output_nodes = 4
feature_num = 2
softplus = nn.Softplus()
p = feature_num


# NN 
class RegressionModel(nn.Module):
    def __init__(self, p, hidden_nodes,hidden_nodes2,hidden_nodes3,output_nodes):
        super(RegressionModel, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.hidden_nodes2 = hidden_nodes2
        self.hidden_nodes3 = hidden_nodes3
        self.output_nodes = output_nodes
        self.fc1 = nn.Linear(p, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes2)
        self.fc3 = nn.Linear(self.hidden_nodes2, self.hidden_nodes3)
        self.fc4 = nn.Linear(self.hidden_nodes3, self.output_nodes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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

        mu2 = Variable(torch.zeros(hidden_nodes2, hidden_nodes)).cuda()
        sigma2 = Variable(torch.ones(hidden_nodes2, hidden_nodes)).cuda()
        bias_mu2 = Variable(torch.zeros(1, hidden_nodes2)).cuda()
        bias_sigma2 = Variable(torch.ones(1, hidden_nodes2)).cuda()

        mu3 = Variable(torch.zeros(hidden_nodes3, hidden_nodes2)).cuda()
        sigma3 = Variable(torch.ones(hidden_nodes3, hidden_nodes2)).cuda()
        bias_mu3 = Variable(torch.zeros(1, hidden_nodes3)).cuda()
        bias_sigma3 = Variable(torch.ones(1, hidden_nodes3)).cuda()

        mu4 = Variable(torch.zeros(output_nodes, hidden_nodes3)).cuda()
        sigma4 = Variable(torch.ones(output_nodes, hidden_nodes3)).cuda()
        bias_mu4 = Variable(torch.zeros(1, output_nodes)).cuda()
        bias_sigma4 = Variable(torch.ones(1, output_nodes)).cuda()
    else:
        mu1 = Variable(torch.zeros(hidden_nodes, p))
        sigma1 = Variable(torch.ones(hidden_nodes, p))
        bias_mu1 = Variable(torch.zeros(1, hidden_nodes))
        bias_sigma1 = Variable(torch.ones(1, hidden_nodes))

        mu2 = Variable(torch.zeros(hidden_nodes2, hidden_nodes))
        sigma2 = Variable(torch.ones(hidden_nodes2, hidden_nodes))
        bias_mu2 = Variable(torch.zeros(1, hidden_nodes2))
        bias_sigma2 = Variable(torch.ones(1, hidden_nodes2))

        mu3 = Variable(torch.zeros(hidden_nodes3, hidden_nodes2))
        sigma3 = Variable(torch.ones(hidden_nodes3, hidden_nodes2))
        bias_mu3 = Variable(torch.zeros(1, hidden_nodes3))
        bias_sigma3 = Variable(torch.ones(1, hidden_nodes3))

        mu4 = Variable(torch.zeros(output_nodes, hidden_nodes3))
        sigma4 = Variable(torch.ones(output_nodes, hidden_nodes3))
        bias_mu4 = Variable(torch.zeros(1, output_nodes))
        bias_sigma4 = Variable(torch.ones(1, output_nodes))


    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)
    w_prior3, b_prior3 = Normal(mu3, sigma3), Normal(bias_mu3, bias_sigma3)
    w_prior4, b_prior4 = Normal(mu4, sigma4), Normal(bias_mu4, bias_sigma4)

    priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc2.weight': w_prior2, 'fc2.bias': b_prior2, 'fc3.weight': w_prior3, 'fc3.bias': b_prior3, 'fc4.weight': w_prior4, 'fc4.bias': b_prior4}
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

        w_mu2 = Variable(torch.randn(hidden_nodes2, hidden_nodes).cuda(), requires_grad=True)
        w_log_sig2 = Variable((-3.0 * torch.ones(hidden_nodes2, hidden_nodes) + 0.05 * torch.randn(hidden_nodes2, hidden_nodes)).cuda(), requires_grad=True)
        b_mu2 = Variable(torch.randn(1, hidden_nodes2).cuda(), requires_grad=True)
        b_log_sig2 = Variable((-3.0 * torch.ones(1, hidden_nodes2) + 0.05 * torch.randn(1, hidden_nodes2)).cuda(), requires_grad=True)

        w_mu3 = Variable(torch.randn(hidden_nodes3, hidden_nodes2).cuda(), requires_grad=True)
        w_log_sig3 = Variable((-3.0 * torch.ones(hidden_nodes3, hidden_nodes2) + 0.05 * torch.randn(hidden_nodes3, hidden_nodes2)).cuda(), requires_grad=True)
        b_mu3 = Variable(torch.randn(1, hidden_nodes3).cuda(), requires_grad=True)
        b_log_sig3 = Variable((-3.0 * torch.ones(1, hidden_nodes3) + 0.05 * torch.randn(1, hidden_nodes3)).cuda(), requires_grad=True)

        w_mu4 = Variable(torch.randn(output_nodes, hidden_nodes3).cuda(), requires_grad=True)
        w_log_sig4 = Variable((-3.0 * torch.ones(output_nodes, hidden_nodes3) + 0.05 * torch.randn(output_nodes, hidden_nodes3)).cuda(), requires_grad=True)
        b_mu4 = Variable(torch.randn(1, output_nodes).cuda(), requires_grad=True)
        b_log_sig4 = Variable((-3.0 * torch.ones(1, output_nodes) + 0.05 * torch.randn(1, output_nodes)).cuda(), requires_grad=True)

    else:
        w_mu1 = Variable(torch.randn(hidden_nodes, p), requires_grad=True)
        w_log_sig1 = Variable((-3.0 * torch.ones(hidden_nodes, p) + 0.05 * torch.randn(hidden_nodes, p)), requires_grad=True)
        b_mu1 = Variable(torch.randn(1, hidden_nodes), requires_grad=True)
        b_log_sig1 = Variable((-3.0 * torch.ones(1, hidden_nodes) + 0.05 * torch.randn(1, hidden_nodes)), requires_grad=True)

        w_mu2 = Variable(torch.randn(hidden_nodes2, hidden_nodes), requires_grad=True)
        w_log_sig2 = Variable((-3.0 * torch.ones(hidden_nodes2, hidden_nodes) + 0.05 * torch.randn(hidden_nodes2, hidden_nodes)), requires_grad=True)
        b_mu2 = Variable(torch.randn(1, hidden_nodes2), requires_grad=True)
        b_log_sig2 = Variable((-3.0 * torch.ones(1, hidden_nodes2) + 0.05 * torch.randn(1, hidden_nodes2)), requires_grad=True)

        w_mu3 = Variable(torch.randn(hidden_nodes3, hidden_nodes2), requires_grad=True)
        w_log_sig3 = Variable((-3.0 * torch.ones(hidden_nodes3, hidden_nodes2) + 0.05 * torch.randn(hidden_nodes3, hidden_nodes2)), requires_grad=True)
        b_mu3 = Variable(torch.randn(1, hidden_nodes3), requires_grad=True)
        b_log_sig3 = Variable((-3.0 * torch.ones(1, hidden_nodes3) + 0.05 * torch.randn(1, hidden_nodes3)), requires_grad=True)

        w_mu4 = Variable(torch.randn(output_nodes, hidden_nodes3), requires_grad=True)
        w_log_sig4 = Variable((-3.0 * torch.ones(output_nodes, hidden_nodes3) + 0.05 * torch.randn(output_nodes, hidden_nodes3)), requires_grad=True)
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

    # register learnable params in the param store
    mw_param4 = pyro.param("guide_mean_weight4", w_mu4)
    sw_param4 = softplus(pyro.param("guide_log_sigma_weight4", w_log_sig4))
    mb_param4 = pyro.param("guide_mean_bias4", b_mu4)
    sb_param4 = softplus(pyro.param("guide_log_sigma_bias4", b_log_sig4))
    # gaussian guide distributions for w and b
    w_dist4 = Normal(mw_param4, sw_param4)
    b_dist4 = Normal(mb_param4, sb_param4)
    

    dists = {'fc1.weight': w_dist1, 'fc1.bias': b_dist1, 'fc2.weight': w_dist2, 'fc2.bias': b_dist2, 'fc3.weight': w_dist3, 'fc3.bias': b_dist3, 'fc4.weight': w_dist4, 'fc4.bias': b_dist4}
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()



# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main(args,data,test_data,softplus,regression_model,feature_num,N,debug=True):

    #Initialize summary writer for Tensorboard
    writer = SummaryWriter()

    x_data, y_data = test_data[:,0:feature_num], test_data[:,feature_num:feature_num+4] #feature_num+4 : change if not 1-hot
    global CUDA_    
    if args.cuda:
        # make tensors and modules CUDA
        CUDA_ = True
        data = data.cuda()
        test_data = test_data.cuda()
        x_data, y_data = x_data.cuda(), y_data.cuda()
        softplus.cuda()
        regression_model.cuda()


    #Monitor model graph
    writer.add_graph(regression_model, data[:, 0:feature_num], verbose=True)

    # instantiate optim and inference objects
    optim = Adam({"lr":0.01})
    svi = SVI(model, guide, optim, loss="ELBO")


    for j in range(args.num_epochs):
        if args.batch_size == N:
            # use the entire data set
            epoch_loss = svi.step(data)
        else:
            # mini batch
            epoch_loss = 0.0
            perm = torch.randperm(N) if not args.cuda else torch.randperm(N).cuda()
            # shuffle data
            data = data[perm]
            # get indices of each batch
            all_batches = get_batch_indices(N, args.batch_size)
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_data = data[batch_start: batch_end]
                epoch_loss += svi.step(batch_data)
        if j % 10 == 0:
            print("epoch avg loss {}".format(epoch_loss/float(N)))
            writer.add_scalar('data/epoch_loss_avg', epoch_loss/float(N), j/100)

        
        for name, param in regression_model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
        
        writer.add_scalar('data/epoch_loss', epoch_loss, j)



    # Validate - test model
    print("Validate trained model...")
         
    #Number of parameter sampling steps
    n_samples = 10000
    y_preds = [] 
    avg_pred = 0.0
    #Create list of float tensors each containing the evaluation of the test data by a sampled BNN

    if debug:
       tst_data, X, Y = create_grid(-12, 12, 50) 
    else:
       minx = np.amin(x_data.data.numpy())
       maxx = np.amax(x_data.data.numpy())
       tst_data, X, Y = create_grid(minx-0.001,maxx+0.001, 50)

    avg_pred = []
    for i in range(n_samples):
    # guide does not require the data
        sampled_reg_model = guide(None)
        if debug:
           y_preds.append(np.argmax(sampled_reg_model(Variable(torch.Tensor(tst_data))).data.numpy(),axis=1))
            
        else:
           y_preds.append(np.argmax(sampled_reg_model(x_data).data.numpy(),axis=1))
        avg_pred.append(np.argmax(sampled_reg_model(Variable(torch.Tensor(tst_data))).data.numpy(),axis=1))
    
    y_pred_np = np.asarray(y_preds)      
    avg_pred_np = np.asarray(avg_pred)
    
    # Needed for decision surface visualisation
    ap_tst = []
    for i in xrange(len(tst_data)):
         ap_tst.append(np.argmax(np.bincount(avg_pred_np[:,i])))
    majority_class_tst = np.asarray(ap_tst)

    ap = []
    for i in xrange(len(x_data)):
        ap.append(np.argmax(np.bincount(y_pred_np[:,i])))
    # Use majority_class to get accuracy on test data    
    majority_class = np.asarray(ap)
    print("Prediction accuracy on test set is",len(np.where(majority_class==np.argmax(y_data.data.numpy(),axis=1))[0])/float(len(majority_class))*100,"%")
    
    base_pred = np.argmax(sampled_reg_model(Variable(torch.Tensor(tst_data))).data.numpy(),axis=1)
    
    visualise_and_debug(y_pred_np,majority_class_tst,base_pred,data,n_samples,X,Y,predictionPDF='/tmp/PredictionPDF.pt',PDFcenters='/tmp/PDFCenters.pt',debug=debug)    

    writer.close()


def initialize(filename_train,filename_test,feature_num,debug=False):

   data = []
   with open(filename_train,'rb') as f:
        """
        logits train
        """
        a = torch.load(f)[:,0:6]
        data = Variable(torch.Tensor(a))

   test_data = []
   with open(filename_test,'rb') as f: 
        test_data = Variable(torch.Tensor(torch.load(f))) 

     
   N = len(data)  # size of data
   regression_model = RegressionModel(feature_num,hidden_nodes,hidden_nodes2,hidden_nodes3,output_nodes)

   if debug:
      xs1, ys1 = create_dataset()
      data1 = np.hstack((xs1,ys1.reshape(len(xs1),1)))
      with open('./tst_data.pt','wb') as f:
          torch.save(data1,f)
      data = Variable(torch.Tensor(data1))


   return data,test_data,N,hidden_nodes,hidden_nodes2,hidden_nodes3,output_nodes,softplus,regression_model



if __name__ == '__main__':
    CUDA_ = False
  
    xs = None    
    ys = None
    filename_train = '../data/Wall/train_data_2sensors_logits.pt'
    filename_test =  '../data/Wall/test_data_2sensors_1hot_scaled.pt'

    # Currently no CUDA on debug mode
    debug = False
    data,test_data,N,hidden_nodes,hidden_nodes2,hidden_nodes3,output_nodes,softplus,regression_model = initialize(filename_train,filename_test,feature_num,debug)

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    main(args,data,test_data,softplus,regression_model,feature_num,N,debug)
