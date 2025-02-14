import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import gzip
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def read(path):
    X_all_labels, y_all_labels=load_mnist(path, kind='train')
    indexLabel1 = np.where((y_all_labels==1))
    xLabel1 =  X_all_labels[indexLabel1][:1000,:].astype('float64')
    yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

    indexLabel5 = np.where((y_all_labels==5))
    xLabel5 =  X_all_labels[indexLabel5][:1000,:].astype('float64')
    yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')

    X_data = np.concatenate((xLabel1, xLabel5), axis=0) 
    Y_data = np.concatenate((yLabel1, yLabel5)) 

    
    Y_data = np.where(Y_data == 5, -1, Y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2,shuffle=True, random_state=1933718)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test) 
    
    return X_train,X_test,y_train, y_test

def kernel_function(x,y,gamma,k_type):  
    if k_type=='Polynomial':
        return ((np.dot(x,y.T)+1)**gamma) 
    
def Q_function(X1,X2,Y1,Y2,gamma,k_type): #Q=YKY
    K=kernel_function(X1,X2,gamma,k_type)
    Q=np.dot(np.dot(np.diag(Y1),K),np.diag(Y2)).astype('float64')
    return Q

def classifier(alfa,X_train,X,y,gamma,C,tolerance,k_type):
    free_Support_vectors=np.where(np.logical_and(alfa>tolerance,alfa<C-tolerance))[0]
    y_diagonal=np.diag(y)
    if free_Support_vectors.shape[0]!=0:
        
        bias=np.mean(y[free_Support_vectors] - np.dot(np.dot(alfa.T,y_diagonal),kernel_function(X_train,X_train[free_Support_vectors,:],gamma,k_type)))

    else:
        Q=Q_function(X_train,X_train,y,y,gamma,k_type)
        grad=(np.dot(Q, alfa) - 1)
        R,S=R_S_finder(alfa,y,C,tolerance)
        M,m=M_m_finder(grad,y,R,S)
        bias=np.mean([M,m])

    f_x=np.dot(np.dot(alfa.T,y_diagonal),kernel_function(X_train,X,gamma,k_type))+bias
    f_x=np.sign(f_x)
    return f_x


def M_m_finder(grad,Y,R,S):


    Y=Y.reshape(-1,1)

    M=np.min(-grad[S]*Y[S]) 
    
    m=np.max(-grad[R]*Y[R]) 


    return M,m

def R_S_finder(alfa,Y,C,tolerance):

    Y=Y.reshape(-1,1)

    condition1_S=(alfa<C-tolerance)&(Y==-1)
    condition2_S=(alfa>tolerance)&(Y==1)
    condition1_R=(alfa<C-tolerance)&(Y==1)
    condition2_R=(alfa>tolerance)&(Y==-1)

    S = np.where(condition1_S|condition2_S)[0]

    R = np.where(condition1_R|condition2_R)[0]

    return R,S

def working_set(Y,grad,R,S):
    Y=Y.reshape(-1,1)
    M_grad=(-grad[S]*Y[S])
    m_grad=(-grad[R]*Y[R])

    i=np.array(R[np.argmax(m_grad)]).reshape(1,) 
    j=np.array(S[np.argmin(M_grad)]).reshape(1,)

    W=np.concatenate((i,j),axis=0)
    return W,i,j


def t_finder(d,gradient,W,Q_W):
            c1=-np.dot(d.T,gradient[W])
            c2=np.dot(np.dot(d.T,Q_W),d)
            return c1/c2


def decomposition(X,Y,C,gamma,tolerance,max_iter,k_type):
    P=len(X)
    alfa=np.zeros((P,1))
    grad=-np.ones((P,1))
    iter=0

    R,S=R_S_finder(alfa,Y,C,tolerance)

    M,m=M_m_finder(grad,Y,R,S)

    Q_cache={} #cache for Q columns

    while (m-M>1e-10) & (iter<max_iter):
        W,i,j=working_set(Y,grad,R,S)
        d=np.array([Y[i],-Y[j]]).reshape(2,1)

        t_max1=[np.inf]
        t_max2=[np.inf]

        alfa_partial=np.concatenate((alfa[i],alfa[j])) #subset of alfa associated with i and j
        for h in range(2):
            if d[h]>0:   #tolerance is not neccessary since it is always d[h]=1 or d[h]=-1   
                t_max1.append((C-alfa_partial[h])/d[h])

            if d[h]<0:
                t_max2.append(alfa_partial[h]/(np.abs(d[h])))

        t_max_feas=min(min(t_max1),min(t_max2))
        #cache update
        if i.item() not in Q_cache:
            X_i=X[i,:] 
            Y_i=Y[i]
            Q_cache[i.item()]=Q_function(X,X_i,Y,Y_i,gamma,k_type)

        if j.item() not in Q_cache:
            X_j=X[j,:] 
            Y_j=Y[j]
            Q_cache[j.item()]=Q_function(X,X_j,Y,Y_j,gamma,k_type) 

        Q_columns=np.hstack((Q_cache[i.item()],Q_cache[j.item()])) 

        Q_2x2=Q_columns[W,:] #submatrix of Q associated with i and j

        if t_max_feas<1e-16:
            t_star=0
        elif np.dot(np.dot(d.T,Q_2x2),d)<1e-16:
            t_star=t_max_feas
        else:
            if np.dot(np.dot(d.T,Q_2x2),d)>1e-16:
                t_nv=t_finder(d,grad,W,Q_2x2)
                t_star=min(t_max_feas,t_nv)

        alfa_star=alfa[W]+t_star*d

        #gradient update
        grad=grad+np.dot(Q_columns,(alfa_star-alfa[W]))

        #alfa update
        alfa[W]=alfa_star

        R,S=R_S_finder(alfa,Y,C,tolerance)
        M,m=M_m_finder(grad,Y,R,S)

        iter+=1
    return alfa, m, M, iter, Q_cache


def f_objective(alfa,X,Y,gamma,k_type):
    Q=Q_function(X,X,Y,Y,gamma,k_type)
    quadratic_term=0.5*np.dot(np.dot(alfa.T,Q),alfa).item()
    linear_term=-np.sum(alfa)
    return (quadratic_term+linear_term)
  