import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
from scipy import stats

#  KNN function
def knn_predict(X_test, X_train, y_train, k=1):
    n_X_test = X_test.shape[0]
    decision = np.zeros((n_X_test, 1))    #column vector
    for i in range(n_X_test):
        point = X_test[[i],:]    #ith row all columns

        #  compute euclidan distance from the point to all training data
        dist = cdist(X_train, point)   #dist stores in an array all distances

        #  sort the distance, get the index
        idx_sorted = np.argsort(dist, axis=0)

        #  find the most frequent class among the k nearest neighbour
        pred = stats.mode( y_train[idx_sorted[0:k]] )

        decision[i] = pred[0]
    return decision


np.random.seed(1)

# Setup data
def switch_classes(m):
    D = np.genfromtxt('iris.csv', delimiter=',')
    a=np.arange(len(D))
    a_rand=np.random.choice(a,size=m)
    for i in a_rand:                    ## assigning random classes to the train set.
            if D[i][2]==1.0: D[i][2]=np.random.choice([2,3],size=1)[0]; continue
            if D[i][2]==2.0: D[i][2]=np.random.choice([1,3],size=1)[0]; continue
            if D[i][2]==3.0: D[i][2]=np.random.choice([1,2],size=1)[0]; continue
    
    return D

D10=switch_classes(10)
D20=switch_classes(20)
D30=switch_classes(30)
D50=switch_classes(50)


def meshandplot(Dd):
    X_train = Dd[:, 0:2]   # feature
    y_train = Dd[:, -1]    # label
    # Setup meshgrid
    x1, x2 = np.meshgrid(np.arange(2,5,0.01), np.arange(0,3,0.01))
    X12 = np.c_[x1.ravel(), x2.ravel()]
    
    # Compute 1NN decision
    k = 3
    decision = knn_predict(X12, X_train, y_train, k)
    
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    #  plot decisions in the grid
    decision = decision.reshape(x1.shape)
    plt.figure()
    plt.pcolormesh(x1, x2, decision, cmap=cmap_light)
    
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s=25)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    
    plt.show()

#function for calculating leave one out validation error.
def val(Dx):
    
    X_train = Dx[:, 0:2]   # feature
    y_train = Dx[:, -1]    # label
    detected=np.zeros(len(y_train))
    for i in range(150):
        X_train_mod=np.delete(X_train,i,axis=0)     #deleting the left out element for validation
        x12=np.array(X_train[i]).reshape(1,2)
        k = 3
        decision = knn_predict(x12, X_train_mod, y_train, k)
        detected[i]=decision
    errorRate = (y_train != detected).mean()      #mean takes avg of number of true values for the given condition
    print("Error rate=",errorRate)   #error rate of loov

val(D10)
val(D20)
val(D30)
val(D50)

meshandplot(D10)
meshandplot(D20)
meshandplot(D30)
meshandplot(D50)