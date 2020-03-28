import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np

data=pd.read_csv("ads.csv")
print(data.head())

real_x=data.iloc[:,[2,3]].values
real_y=data.iloc[:,4].values

train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)

sv=SVC(kernel="linear",random_state=0)
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.fit_transform(test_x)
sv.fit(train_x,train_y)
pred_y=sv.predict(test_x)

cf=confusion_matrix(test_y,pred_y)
print(cf)


# plotting of data
#traning data
from matplotlib.colors import ListedColormap
X_set,Y_set=train_x,train_y
X1 , X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1,X2,sv.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate (np.unique(Y_set)):
   plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1], c=ListedColormap(('red','green'))(i),label=j)
plt.title("SVM")
plt.xlabel("Age")
plt.ylabel("Estimated_sal")
plt.legend()
plt.show()

#testing  data
from matplotlib.colors import ListedColormap
X_set,Y_set=test_x,test_y
X1 , X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1,X2,sv.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate (np.unique(Y_set)):
   plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1], c=ListedColormap(('red','green'))(i),label=j)
plt.title("SVM")
plt.xlabel("Age")
plt.ylabel("Estimated_sal")
plt.legend()
plt.show()
