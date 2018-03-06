import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#003CB3', '#E62E00', '#006622', '#FFFF1A'])

def visualise_sample_ds(base_pred,xs,ys,X,Y): 
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


def visualise_avg_ds(avg_pred,xs,ys,X,Y): 
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


def visualise_navigation_ds(avg_pred,base_pred,xs,ys,X,Y): 
    """
    Visualize decision surface based on a sample of the network
    """
    y = np.argmax(ys,axis=1)
    bp = np.argmax(base_pred,axis=1)
    
    # plot the base prediction surface
    fig1 = plt.figure()
    ax = fig1.gca()
    Z = bp.reshape(list(X.shape))
    """
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    """
    plt.pcolormesh(X, Y, Z, cmap=cmap_light)
    """
    ax.plot(xs[y == 0, 0], xs[y == 0, 1], 'b.', ms=12)                                                                                                                                                         
    ax.plot(xs[y == 1, 0], xs[y == 1, 1], 'r.', ms=12)
    ax.plot(xs[y == 2, 0], xs[y == 2, 1], 'g.', ms=12)                                                                                                                                                         
    ax.plot(xs[y == 3, 0], xs[y == 3, 1], 'y.', ms=12)                                                                                                                                                       
    ax.view_init(elev=90, azim=-90)                                                                                                                                                           
    """
    plt.xlabel('x1')                                                                                                                                                                                             
    plt.ylabel('x2')                                                                                                                                                                              
    plt.axis('equal')                                                                                                                                                                                          
    ax.axis([np.amin(xs), np.amax(xs), np.amin(xs), np.amax(xs)])                                                                                                                            
    fig1.suptitle('Prediction surface using sample weights')                                                                                                                               

    """
    Visualize decision surface based on an average of network samples
    """

    ap = np.argmax(avg_pred,axis=1)
    
    # plot the average prediction surface                                                                                                                                         
    fig = plt.figure()                                                                                                                                                  
    ax = fig.gca()  #(projection='3d')                                                                                                                              
    Z = ap.reshape(list(X.shape))
    plt.pcolormesh(X, Y, Z, cmap=cmap_light)
    """
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)                                                                        
    ax.plot(xs[y == 0, 0], xs[y == 0, 1], 'b.', ms=12)                                                                                                                                                         
    ax.plot(xs[y == 1, 0], xs[y == 1, 1], 'r.', ms=12)
    ax.plot(xs[y == 2, 0], xs[y == 2, 1], 'g.', ms=12)                                                                                                                                                         
    ax.plot(xs[y == 3, 0], xs[y == 3, 1], 'y.', ms=12)
    ax.view_init(elev=90, azim=-90)                                                                                                                                                 
    """
    plt.xlabel('x1')                                                                                                                                                              
    plt.ylabel('x2')           
    plt.axis('equal')                                                                                                                                                                              
    ax.axis([np.amin(xs), np.amax(xs), np.amin(xs), np.amax(xs)])
    fig.suptitle('Bayesian prediction surface')
    
    return fig1,fig



def visualise_and_debug(y_preds,avg_pred,base_pred,data,n_samples,X,Y,predictionPDF='/tmp/PredictionPDF.pt',PDFcenters='/tmp/PDFCenters.pt',debug=True):

    xs = data[:,0:2].data.numpy()
    ys = data[:,2].data.numpy()

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

    if debug:
       fig1 = visualise_sample_ds(base_pred,xs,ys,X,Y)
       fig2 = visualise_avg_ds(avg_pred,xs,ys,X,Y)
    else:
       ys = data[:,2:6].data.numpy()      #FIX FOR DIFFERENT FEATURE NUMBER
       visualise_navigation_ds(avg_pred,base_pred,xs,ys,X,Y)
       
    plt.show()
    
