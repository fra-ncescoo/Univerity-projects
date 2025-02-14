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
from itertools import combinations


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

    indexLabel7 = np.where((y_all_labels==7))
    xLabel7 =  X_all_labels[indexLabel7][:1000,:].astype('float64')
    yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')
            

    X_data = np.concatenate((xLabel1, xLabel5,xLabel7), axis=0)
    Y_data = np.concatenate((yLabel1, yLabel5,yLabel7), axis=0)

    
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2,shuffle=True, random_state=1933718)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
   
    x1_train = X_train[np.where(y_train == 1)]  
    x5_train = X_train[np.where(y_train == 5)]
    x7_train = X_train[np.where(y_train == 7)]

    y1_train = y_train[np.where(y_train == 1)]
    y5_train = y_train[np.where(y_train == 5)]
    y7_train = y_train[np.where(y_train == 7)]


    x_train=[x1_train,x5_train,x7_train] #train data are divided in classes
    y_train=[y1_train,y5_train,y7_train]

    return x_train,X_test,y_train,y_test


def kernel_function(x,y,gamma,k_type): 
    if k_type=='Polynomial':
        return ((np.dot(x,y.T)+1)**gamma) 
    
def Q_function(X1,X2,Y1,Y2,gamma,k_type): 
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
    f_x=np.sign(f_x).T
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

    Q_cache={}

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
    return alfa,iter,m,M


def f_objective(alfa,X,Y,gamma,k_type):
    Q=Q_function(X,X,Y,Y,gamma,k_type)
    quadratic_term=0.5*np.dot(np.dot(alfa.T,Q),alfa).item()
    linear_term=-np.sum(alfa)
    return (quadratic_term+linear_term)

def set_splitter(pair,X_train,y_train): #given a pair of classes, it returns the corresponding training set

    X_train_i=X_train[pair[0]]
    X_train_j=X_train[pair[1]]

    y_train_i=np.ones((y_train[pair[0]].shape[0],1))   #the same as substituting all y==i with label 1
    y_train_j=-np.ones((y_train[pair[1]].shape[0],1))   #the same as substituting all y==j with label -1

    X_train_current=np.concatenate((X_train_i,X_train_j),axis=0)
    
    y_train_current=np.concatenate((y_train_i,y_train_j),axis=0).flatten()
    
    return X_train_current,y_train_current


def OAO_optimization(X_train,y_train,C,gamma,tolerance,max_iter,k_type): #for each pair of classes, it solves a binary classification problem returning the optimal alfa_star
    
    number_of_classes=len(X_train)
    classes=list(range(number_of_classes))
    combos=list(combinations(classes, 2))
    alfa_opt_dict={}
    obj_f_tracker={}
    KKT_tracker={}
    tot_iter=0
    for pair in combos:
        X_train_pair,y_train_pair=set_splitter(pair,X_train,y_train)

        alfa_star,iter_pair,m,M=decomposition(X_train_pair,y_train_pair,C,gamma,tolerance,max_iter,k_type)
        alfa_opt_dict[pair]=alfa_star
        obj_f_tracker[pair]=f_objective(alfa_star,X_train_pair,y_train_pair,gamma,k_type)
        KKT_tracker[pair]=m-M
        tot_iter+=iter_pair
  
    return alfa_opt_dict,tot_iter, KKT_tracker,obj_f_tracker

def OAO_classification(alfa_opt_dict,X_train,y_train,X_test,gamma,C,tolerance,k_type):   #Computes the prediction of y using a voting procedure
    number_of_classes=len(X_train)
    classes=list(range(number_of_classes))
    combos=list(combinations(classes, 2))
    f_class=np.zeros((X_test.shape[0],3))
    for pair in combos:
        h=pair[0]
        m=pair[1]
        X_train_pair,y_train_pair=set_splitter(pair,X_train,y_train)

        f_class_h=classifier(alfa_opt_dict[pair],X_train_pair,X_test,y_train_pair,gamma,C,tolerance,k_type)  #f(x) in respect to class h

        f_class[:,h]+=0.5*((f_class_h+1).flatten()) 
        f_class[:,m]+=0.5*((-f_class_h+1).flatten())

        
    f_assigned_class=np.argmax(f_class,axis=1)  #assign the class with the highest score (keep in mind that class indices are 0,1,2)
    f_assigned_class=np.array(f_assigned_class).reshape(-1,1)
    y_pred=label_converter(f_assigned_class)

    return y_pred

def label_converter(y): #converts the class indices 0,1,2 into the original labels 1,5,7
    return np.where(y==0,1,np.where(y==1,5,7))  