import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize 
from torch.autograd import Variable
import pyro
from pyro.distributions import Normal, Bernoulli 
from pyro.infer import SVI
from pyro.optim import Adam
import cPickle

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


data = []
with open('../data/Wall/train_data_2sensors_logits.pt','rb') as f:
     data1 = cPickle.load(f)
     data = Variable(torch.Tensor(data1[:,0:6]))
     #data = Variable(torch.Tensor(torch.load(f)))
     
    

N = len(data)  # size of data
p = 2  # number of features
hidden_nodes = 6
output_nodes = 4
softplus = nn.Softplus()
regression_model = RegressionModel(p,hidden_nodes,output_nodes)


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
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.iarange("map", N, subsample=data):
        x_data = data[:, 0:2]
        y_data = data[:, 2:6]
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
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()


# instantiate optim and inference objects
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss="ELBO")


# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main(args,data):
    global CUDA_    
    if args.cuda:
        # make tensors and modules CUDA
        CUDA_ = True
        data = data.cuda()
        softplus.cuda()
        regression_model.cuda()
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
        if j % 100 == 0:
            print("epoch avg loss {}".format(epoch_loss/float(N)))
        
    print("Validate trained model...")
    test_data = []
    with open('../data/Wall/test_data_2sensors_logits.pt','rb') as f: 
         test_data1 = cPickle.load(f)
         test_data = Variable(torch.Tensor(test_data1[:,0:6]))
         #test_data = Variable(torch.Tensor(torch.load(f))) 


    loss = nn.MSELoss()
    x_data, y_data = test_data[:,0:2], test_data[:,2:6]
    y_preds = Variable(torch.zeros(len(test_data), 1))

    if args.cuda:
        test_data = test_data.cuda()
        x_data, y_data = x_data.cuda(), y_data.cuda()
        y_preds = y_preds.cuda()

    for i in range(len(test_data)):
    # guide does not require the data
        sampled_reg_model = guide(None)
        # run the regression model and add prediction to total
        y_preds = y_preds + sampled_reg_model(x_data)
        # take the average of the predictions
    y_preds = y_preds / len(test_data)
    
    print ("Loss: ", loss(y_preds, y_data).data[0])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=5000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    CUDA_ = False

    main(args,data)
