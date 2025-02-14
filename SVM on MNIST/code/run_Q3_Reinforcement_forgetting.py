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

from functions_Q3_Reinforcement_forgetting import *

cwd = os.getcwd()
X_train,X_test,y_train, y_test=read(cwd)

k_type='Polynomial'
if k_type=='Polynomial':
    gamma=2
    C=0.0001

tolerance=1e-5
max_iter=1000


t0=time.time()
alfa_star,m,M,iter,Q_cache=decomposition(X_train,y_train,C,gamma,tolerance,max_iter,k_type)
t1=time.time()
opt_time=t1-t0


y_pred_train=classifier(alfa_star,X_train,X_train,y_train,gamma,C, tolerance,k_type)
train_accuracy=acc(y_train,y_pred_train.flatten()) 

y_pred_test=classifier(alfa_star,X_train,X_test,y_train,gamma,C, tolerance,k_type)
test_accuracy=acc(y_test,y_pred_test.flatten())


conf_matrix_train = confusion_matrix(y_train,y_pred_train.flatten())
conf_matrix_test = confusion_matrix(y_test,y_pred_test.flatten()) 

print('Used kernel type:',k_type) #1
print('C, gamma: ',C,gamma) #2
print('Classification rate on traing set: ',(train_accuracy*100),'%') #3
print('Classification rate on test set: ',(test_accuracy*100),'%') #4 
print('Confusion matrix train:',conf_matrix_train) #5
print('Confusion matrix test:',conf_matrix_test) #5
print('Time necessary for optimization: ',opt_time) #6
print('Number of optimization iterations:',iter) #7
print('Difference between m(alfa) and M(alfa):',m-M) #8
print('Final value of the dual SVM objective function:',f_objective(alfa_star,X_train,y_train,gamma,k_type)) #9

#print('Solver status: ', sol['status']) #10

n_columns=len(set(Q_cache.keys()))
print('Number of columns of the matrix in the quadratic objective funciton computed to solve the dual SVM: ',n_columns) #11

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_train, display_labels=[1,5])    
disp.plot(cmap='Greens')   
plt.title('Confusion Matrix Train')
plt.show()  


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=[1,5])    
disp.plot(cmap='Blues') 
plt.title('Confusion Matrix Test')  
plt.show()  