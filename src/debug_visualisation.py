import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

def visualise_sample_ds(base_pred,xs,ys): 
    """
    Visualize decision surface based on a sample of the network
    """

    # plot the base prediction surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')                                                                                                                                                                                 
    Z = base_pred.reshape(list(X.shape))                                                                                                                                                        
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)                                                                                                                                 
    ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)                                                                                                                                                         
    ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)                                                                                                                                                       
    ax.view_init(elev=90, azim=-90)                                                                                                                                                           
    plt.xlabel('x1')                                                                                                                                                                                             
    plt.ylabel('x2')                                                                                                                                                                              
    plt.axis('equal')                                                                                                                                                                                          
    ax.axis([-12, 12, -12, 12])                                                                                                                                                                                  
    fig.suptitle('Prediction surface using average weights')                                                                                                                               

    return fig


def visualise_avg_ds(avg_pred,xs,ys): 
    """
    Visualize decision surface based on an average of network samples
    """
                                                                                                                                                                                                                  
    # plot the average prediction surface                                                                                                                                         
    fig = plt.figure()                                                                                                                                                  
    ax = fig.gca(projection='3d')                                                                                                                              
    Z = avg_pred.reshape(list(X.shape))                                                                                       
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)                                                                        
    ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)                                                                                                                           
    ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)                                                                                                                     
    ax.view_init(elev=90, azim=-90)                                                                                                                                                  
    plt.xlabel('x1')                                                                                                                                                              
    plt.ylabel('x2')           
    plt.axis('equal')                                                                                                                                                                              
    ax.axis([-12, 12, -12, 12])
    fig.suptitle('Bayesian prediction surface')                                                                                                                                                                    

    return fig



def visualise_and_debug(y_preds,avg_pred,base_pred,data,n_samples,predictionPDF='/tmp/PredictionPDF.pt',PDFcenters='/tmp/PDFCenters.pt'):

    xs = data[:,0:2]
    ys = data[:,2]

    #Create histograms of sample predictions to form the final posterior predictive distribution                                                                                         
    probs = [] 
    centers = []
    for i in y_preds:
        histogram = np.histogram(i.detach().cpu().data.numpy(), bins=20)
        probs.append(histogram[0] / float(n_samples))
        delta = histogram[1][1] - histogram[1][0]
        centers.append([np.float32(a + delta / 2) for a in histogram[1][:-1]])
    with open(predictionPDF,'wb') as f:
         torch.save(probs,f)
    with open(PDFcenters,'wb') as f:
         torch.save(centers,f)

    fig1 = visualise_sample_ds(base_pred,xs,ys)
    fig2 = visualise_avg_ds(avg_pred,xs,ys)

    plt.show()
    
