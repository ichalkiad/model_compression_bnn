import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import itertools

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
    
    # plot the base prediction surface
    fig1 = plt.figure()
    ax = fig1.gca()
    Z = base_pred.reshape(list(X.shape))
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

    # plot the average prediction surface                                                                                                                                         
    fig = plt.figure()                                                                                                                                                  
    ax = fig.gca()  #(projection='3d')                                                                                                                              
    Z = avg_pred.reshape(list(X.shape))
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
    #centers = []
    for i in xrange(y_preds.shape[1]):
        histogram = np.histogram(y_preds[:,i],bins=4,density=False)
        print(histogram[0])
        probs.append(histogram[0] / float(n_samples))
        #delta = histogram[1][1] - histogram[1][0]
        #centers.append([np.float32(a + delta / 2) for a in histogram[1][:-1]])
    with open(predictionPDF,'wb') as f:
         torch.save(probs,f)
    #with open(PDFcenters,'wb') as f:
    #     torch.save(centers,f)

    if debug:
       fig1 = visualise_sample_ds(base_pred,xs,ys,X,Y)
       fig2 = visualise_avg_ds(avg_pred,xs,ys,X,Y)
    else:
       ys = data[:,2:6].data.numpy()      #FIX FOR DIFFERENT FEATURE NUMBER
       visualise_navigation_ds(avg_pred,base_pred,xs,ys,X,Y)
       
    plt.show()
    

#Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
  
