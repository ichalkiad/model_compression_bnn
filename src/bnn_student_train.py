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


# generate toy dataset
def build_linear_dataset(N, p, noise_std=0.01):
    X = np.random.rand(N, p)
    # use random integer weights from [0, 7]
    w = np.random.randint(8, size=p)
    # set b = 1
    y = np.matmul(X, w) + np.repeat(1, N) + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))

    return torch.cat((X, y), 1)


# NN 
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(p, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x


N = 100  # size of toy data
p = 2  # number of features

softplus = nn.Softplus()
regression_model = RegressionModel(p)


def model(data):
    # Create unit normal priors over the parameters
    mu1 = Variable(torch.zeros(3, p)).type_as(data)
    sigma1 = Variable(torch.ones(3, p)).type_as(data)
    
    bias_mu1 = Variable(torch.zeros(1, 3)).type_as(data)
    bias_sigma1 = Variable(torch.ones(1, 3)).type_as(data)
    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)

    mu2 = Variable(torch.zeros(1, 3)).type_as(data)
    sigma2 = Variable(torch.ones(1, 3)).type_as(data)
    bias_mu2 = Variable(torch.zeros(1, 1)).type_as(data)
    bias_sigma2 = Variable(torch.ones(1, 1)).type_as(data)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)
   
    priors = {'fc1.weight': w_prior1, 'fc1.bias': b_prior1, 'fc2.weight': w_prior2, 'fc2.bias': b_prior2}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.iarange("map", N, subsample=data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze()
        pyro.sample("obs",
                    Normal(prediction_mean, Variable(torch.ones(data.size(0))).type_as(data)),
                    obs=y_data.squeeze())


def guide(data):
    w_mu1 = Variable(torch.randn(3, p).type_as(data.data), requires_grad=True)
    w_log_sig1 = Variable((-3.0 * torch.ones(3, p) + 0.05 * torch.randn(3, p)).type_as(data.data), requires_grad=True)
    b_mu1 = Variable(torch.randn(1, 3).type_as(data.data), requires_grad=True)
    b_log_sig1 = Variable((-3.0 * torch.ones(1, 3) + 0.05 * torch.randn(1, 3)).type_as(data.data), requires_grad=True)
    # register learnable params in the param store
    mw_param1 = pyro.param("guide_mean_weight1", w_mu1)
    sw_param1 = softplus(pyro.param("guide_log_sigma_weight1", w_log_sig1))
    mb_param1 = pyro.param("guide_mean_bias1", b_mu1)
    sb_param1 = softplus(pyro.param("guide_log_sigma_bias1", b_log_sig1))
    # gaussian guide distributions for w and b
    w_dist1 = Normal(mw_param1, sw_param1)
    b_dist1 = Normal(mb_param1, sb_param1)
    
    w_mu2 = Variable(torch.randn(1, 3).type_as(data.data), requires_grad=True)
    w_log_sig2 = Variable((-3.0 * torch.ones(1, 3) + 0.05 * torch.randn(1, 3)).type_as(data.data), requires_grad=True)
    b_mu2 = Variable(torch.randn(1, 1).type_as(data.data), requires_grad=True)
    b_log_sig2 = Variable((-3.0 * torch.ones(1, 1) + 0.05 * torch.randn(1, 1)).type_as(data.data), requires_grad=True)
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
optim = Adam({"lr": 0.001})
svi = SVI(model, guide, optim, loss="ELBO")


# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main(args):
    data = build_linear_dataset(N, p)
    if args.cuda:
        # make tensors and modules CUDA
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
