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

from functions_Q4_Reinforcement_forgetting import *

cwd = os.getcwd()
X_train,X_test,y_train, y_test=read(cwd)


k_type='Polynomial'
if k_type=='Polynomial':
    gamma=2
    C=0.0001

tolerance=1e-5
max_iter=1000
k=5



t0=time.time()
alfa_opt_dict,iter,KKT_tracker,obj_f=OAO_optimization(X_train,y_train,C,gamma,tolerance,max_iter,k_type)
t1=time.time()
opt_time=t1-t0


y_pred_train=OAO_classification(alfa_opt_dict,X_train,y_train,np.concatenate(X_train,axis=0),gamma,C,tolerance,k_type)
y_pred_test=OAO_classification(alfa_opt_dict,X_train,y_train,X_test,gamma,C,tolerance,k_type)

train_accuracy=acc(np.concatenate(y_train,axis=0),y_pred_train.flatten()) 

test_accuracy=acc(y_test,y_pred_test.flatten())


conf_matrix_train = confusion_matrix(np.concatenate(y_train,axis=0),y_pred_train.flatten())
conf_matrix_test = confusion_matrix(y_test,y_pred_test.flatten()) 

print('Used kernel type:',k_type) #1
print('C, gamma: ',C,gamma) #2
print('Classification rate on traing set: ',(train_accuracy*100),'%') #3
print('Classification rate on test set: ',(test_accuracy*100),'%') #4 
print('Confusion matrix train',conf_matrix_train) #5
print('Confusion matrix test',conf_matrix_test) #5bis
print('Time necessary for optimization: ',opt_time) #6
print('Number of TOTAL optimization iterations:',iter) #7
for i in KKT_tracker:
    i_array=np.array(i)
    correct_class_pair=label_converter(i_array)
    print('Difference between m(alfa) and M(alfa) associated with',correct_class_pair,': ',KKT_tracker[i]) #8
    print('Final value of the dual SVM objective function associated with',correct_class_pair,': ',obj_f[i]) #9

print('Multiclass strategy implemented: One against One (OAO)') #11



disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_train, display_labels=[1,5,7])    
disp.plot(cmap='Oranges')   
plt.title('Confusion Matrix Train')
plt.show()  


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=[1,5,7])    
disp.plot(cmap='Purples') 
plt.title('Confusion Matrix Test')  
plt.show()  