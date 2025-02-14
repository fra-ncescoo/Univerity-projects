import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cvxopt import matrix, solvers
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


def M_m_finder(grad,Y,R,S): #from R and S find M and m

    Y=Y.reshape(-1,1)

    M=np.min(-grad[S]*Y[S]) 
    
    m=np.max(-grad[R]*Y[R]) 

    return M,m

def R_S_finder(alfa,Y,C,tolerance): #computes R and S sets

    Y=Y.reshape(-1,1)

    condition1_S=(alfa<C-tolerance)&(Y==-1)
    condition2_S=(alfa>tolerance)&(Y==1)
    condition1_R=(alfa<C-tolerance)&(Y==1)
    condition2_R=(alfa>tolerance)&(Y==-1)

    S = np.where(condition1_S|condition2_S)[0]

    R = np.where(condition1_R|condition2_R)[0]
    return R,S

def working_set(Y,grad,R,S,q): #computes working set
    
    Y=Y.reshape(-1,1)
    
    M_grad=(-grad[S]*Y[S])
    m_grad=(-grad[R]*Y[R])

    q1_set=np.argsort(-m_grad.flatten()) 
    q2_set=np.argsort(M_grad.flatten()) 


    I=R[q1_set[:q // 2]]
    J=S[q2_set[:q // 2]]

    W=np.concatenate((I,J),axis=0)
    return W


def decomposition(X,Y,C,gamma,q,tolerance,max_iter,k_type):

    P=len(X)
    alfa=np.zeros((P,1))
    grad=-np.ones((P,1))
    iter=0

    R,S=R_S_finder(alfa,Y,C,tolerance)
    M,m=M_m_finder(grad,Y,R,S)

    while (m-M>1e-10) & (iter<max_iter):
        solvers.options['show_progress']=False
        R,S=R_S_finder(alfa,Y,C,tolerance)
        
        W=working_set(Y,grad,R,S,q)

        #Quadratic term
        Q=Q_function(X[W],X[W],Y[W],Y[W],gamma,k_type) 
        Q_term=matrix(Q) 

        #Linear term
        not_W_indices=np.delete(np.arange(P),W)

        Q_W_not_W=Q_function(X[W],X[not_W_indices],Y[W],Y[not_W_indices],gamma,k_type)

        alfa_not_W=(np.delete(alfa,W)).reshape(-1,1)

        e_term=matrix(np.dot(Q_W_not_W,alfa_not_W ) - 1)

        #Inequality constraints
        G_term=matrix(np.vstack((-np.eye(q),np.eye(q)))) 

        H_term=matrix(np.vstack((np.zeros((q,1)),C*np.ones((q,1)))))

        #equality constraint
        A=matrix(Y[W]).T 

        b=matrix(-np.dot((np.delete(Y,W)),(np.delete(alfa,W))), tc = 'd')

        #solver
        solvers.options['abstol'] = 1e-10
        sol=solvers.qp(Q_term,e_term,G_term,H_term,A,b)
        alfa_star=np.array(sol['x'])
   
        Y_W=Y[W]

        kernel=kernel_function(X,X[W],gamma,k_type)

        factor1=np.dot(np.diag(Y),kernel)
        Q_W=np.dot(factor1,np.diag(Y_W))

        #gradient update
        grad=grad+np.dot(Q_W,(alfa_star-alfa[W]))

        #alfa update
        alfa[W]=alfa_star


        R,S=R_S_finder(alfa,Y,C,tolerance)
        M,m=M_m_finder(grad,Y,R,S)


        iter+=1
    
    return alfa, m, M, iter, sol


def f_objective(alfa,X,Y,gamma,k_type): #computes the value of the objective function
    Q=Q_function(X,X,Y,Y,gamma,k_type)
    quadratic_term=0.5*np.dot(np.dot(alfa.T,Q),alfa).item()
    linear_term=-np.sum(alfa)
    return (quadratic_term+linear_term)
