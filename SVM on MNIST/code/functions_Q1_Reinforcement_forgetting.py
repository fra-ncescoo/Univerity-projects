import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import gzip
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from itertools import product


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


def read(path): #read the dataset, split it in training and test set and applay the standard scaler
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

def kernel_function(x,y,gamma,k_type):  #if k_type==Gaussian->gaussian kernel if k_type==polinomial-> polynomilal kernel
    if k_type=='Polynomial':
        return ((np.dot(x,y.T)+1)**gamma)
    
    if k_type=='Gaussian':
        sq_x= np.sum(x**2, axis=1).reshape(-1,1) 
        sq_y= np.sum(y**2, axis=1).reshape(1,-1)
        dist_sq= sq_x+sq_y-2*np.dot(x,y.T)
        return np.exp(-gamma*dist_sq)
    
def Q_function(X,Y,gamma,k_type): #Q=YKY
    K=kernel_function(X,X,gamma,k_type)
    Q=np.dot(np.dot(np.diag(Y),K),np.diag(Y)).astype('float64')
    return Q

def dual_solver(X,y,C,gamma,k_type): #solves the dual problem through cvxopt
    P=len(X) 
    
    #Quadratic term
    Q_term=matrix(Q_function(X,y,gamma,k_type)) 

    #Linear term
    e_term=matrix(-np.ones((X.shape[0],1)))       

    #Inequality constraints
    G_term=matrix(np.vstack((-np.eye(P),np.eye(P)))) 

    H_term=matrix(np.vstack((np.zeros((P,1)),C*np.ones((P,1)))))

    #equality constraint
    A=matrix(y).T 
    
    b=matrix(0.0)

    #solver
    solvers.options['abstol'] = 1e-10
    sol=solvers.qp(Q_term,e_term,G_term,H_term,A,b)
    alfa_star = np.array(sol['x'])

    return sol,alfa_star


def classifier(alfa,X_train,X,y,gamma,C,tollerance,k_type):  #given an alpha, returns the predicted labels 
    Free_Support_vectors=np.where(np.logical_and(alfa>tollerance,alfa<C-tollerance))[0]
    y_diagonal=np.diag(y)
    if Free_Support_vectors.shape[0]!=0:
        bias=np.mean(y[Free_Support_vectors] - np.dot(np.dot(alfa.T,y_diagonal),kernel_function(X_train,X_train[Free_Support_vectors,:],gamma,k_type)))

    else: #if there are no support vectors chung-jin (a library for SVM) method is used
        M,m=set_finder(alfa,X_train,y,C,gamma,tollerance,k_type)
        bias=np.mean([M,m])

    f_x=np.dot(np.dot(alfa.T,y_diagonal),kernel_function(X_train,X,gamma,k_type))+bias
    f_x=np.sign(f_x)
    return f_x

def set_finder(alfa,X_train,Y,C,gamma,tolerance,k_type): 
    
    Q=Q_function(X_train,Y,gamma,k_type)

    grad=(np.dot(Q, alfa) - 1)

    Y=Y.reshape(-1,1)

    #finds the set S and R
    condition1_S=(alfa<C-tolerance)&(Y==-1)
    condition2_S=(alfa>tolerance)&(Y==1)
    condition1_R=(alfa<C-tolerance)&(Y==1)
    condition2_R=(alfa>tolerance)&(Y==-1)

    S = np.where(condition1_S|condition2_S)[0]
    R = np.where(condition1_R|condition2_R)[0]

    # Compute M and m
    M=np.min(-grad[S]*Y[S]) 
    m=np.max(-grad[R]*Y[R]) 

    return M,m


#grid search functions
def cross_validation(k,x,y,C,gamma,tolerance,k_type): #it also keeps track of the number of iterations
    kf=KFold(k,random_state=1933718,shuffle=True)
    accuracy_val_list=[]
    accuracy_train_list=[]
    iter_list=[]
    for train_index, test_index in kf.split(x):
        X_train, X_validation = x[train_index,:], x[test_index,:] 
        y_train, y_validation = y[train_index], y[test_index]  
        sol,alfa_opt=dual_solver(X_train,y_train,C,gamma,k_type)
        y_pred_train=classifier(alfa_opt,X_train,X_train,y_train,gamma,C,tolerance,k_type)
        y_pred_test=classifier(alfa_opt,X_train,X_validation,y_train,gamma,C,tolerance,k_type)
        train_accuracy=acc(y_train,y_pred_train.flatten())
        val_accuracy=acc(y_validation,y_pred_test.flatten())
        accuracy_train_list.append(train_accuracy)
        accuracy_val_list.append(val_accuracy)
        iter_list.append(sol['iterations'])

    mean_accuracy_train=np.mean(accuracy_train_list)
    mean_accuracy_val=np.mean(accuracy_val_list)
    mean_iter=np.mean(iter_list)
    return mean_accuracy_train,mean_accuracy_val,mean_iter


def grid_search(param_grid,k,X,Y,tolerance,k_type):
    maximum=0 
    param_combinations = list(product(param_grid["C"],param_grid["gamma"]))
    C_list=[]
    gamma_list=[]
    training_accuracy_list=[]
    validation_accuracy_list=[]
    iter_list=[]
    for C,gamma in param_combinations:
        training_accuracy,validation_accuracy,mean_iter=cross_validation(k,X,Y,C,gamma,tolerance,k_type)
        C_list.append(C)
        gamma_list.append(gamma)
        training_accuracy_list.append(training_accuracy)
        validation_accuracy_list.append(validation_accuracy)
        iter_list.append(mean_iter)
        if validation_accuracy>maximum:
            maximum=validation_accuracy
            C_opt=C
            gamma_opt=gamma
    
    plot_fitting_C(C_list,gamma_list,gamma_opt,training_accuracy_list,validation_accuracy_list)
    plot_fitting_gamma(C_list,gamma_list,C_opt,training_accuracy_list,validation_accuracy_list)
    
    return maximum, C_opt, gamma_opt,iter_list


def plot_fitting_C(C_list,gamma_list,gamma_opt,training_list,validation_list):
    indices = [i for i, gamma in enumerate(gamma_list) if gamma == gamma_opt]
    
    C_of = [np.log(C_list[i]) for i in indices]
    training_error = [training_list[i] for i in indices]
    validation_error = [validation_list[i] for i in indices]

    plt.figure(figsize=(8,4))
    plt.plot(C_of, training_error, label='Training Error', marker='o')
    plt.plot(C_of, validation_error, label='Validation Error', marker='s')
    plt.title(f'Training and Validation Error over C (gamma={gamma_opt})')
    plt.xlabel('log(C)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fitting_gamma(C_list,gamma_list,C_opt,training_list,validation_list):
    indices = [i for i, C in enumerate(C_list) if C == C_opt]
    
    gamma_of = [np.log(gamma_list[i]) for i in indices]
    training_error = [training_list[i] for i in indices]
    validation_error = [validation_list[i] for i in indices]

    plt.figure(figsize=(8,4))
    plt.plot(gamma_of, training_error, label='Training Error', marker='o')
    plt.plot(gamma_of, validation_error, label='Validation Error', marker='s')
    plt.title(f'Training and Validation Error over gamma (C={C_opt})')
    plt.xlabel('log(Gamma)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()