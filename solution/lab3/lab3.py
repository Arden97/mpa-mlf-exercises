from sklearn.datasets import (
    load_iris,
    make_blobs
)
from sklearn.svm import (
    SVC,
    OneClassSVM
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy

iris=load_iris()
iris.feature_names
print(iris.feature_names)
x=iris.data
y=iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
plt.figure(1)
plt.scatter(x[y==0,0],x[y==0,1],color='green')
plt.scatter(x[y==1,0],x[y==1,1],color='blue')
plt.scatter(x[y==2,0],x[y==2,1],color='cyan')

# choose only first two features (columns) of iris.data
# SVM is in its basic form a 2-class classifier, so eliminate iris.target =2 from the data
x=iris.data[iris.target!=2,0:2]
y=iris.target[iris.target!=2]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(x.shape)

# Plot scatterplots of targets 0 and 1 and check the separability of the classes
plt.figure(2)
plt.scatter(x[y==0,0],x[y==0,1],color='green')
plt.scatter(x[y==1,0],x[y==1,1],color='blue')

#  Train and test the SVM classifier, play with regularization parameter C (either use the default value or try e.g. 200)
SVMmodel=SVC(C=200, kernel='linear')
SVMmodel.fit(x_train,y_train)
SVMmodel.get_params()
SVMmodel.score(x_test,y_test)
print(SVMmodel.score(x_test,y_test))

# Show support vectors in the 2D plot, plot the decision line from equation [w0 w1]*[x0 x1] + b = 0
supvectors=SVMmodel.support_vectors_
# Plot the support vectors here
plt.figure(3)
plt.scatter(x[y==0,0],x[y==0,1],color='green')
plt.scatter(x[y==1,0],x[y==1,1],color='blue')
plt.scatter(supvectors[:,0],supvectors[:,1],color='red')
#Separating line coefficients
w=SVMmodel.coef_
b=SVMmodel.intercept_
x0 = numpy.linspace(min(x[:,0]),max(x[:,0]),101)
x1 = -w[:,0]/w[:,1]*x0 - b/w[:,1]
plt.scatter(x0,x1,s=5)

# Import one-class SVM and generate data (Gaussian blobs in 2D-plane)
numpy.random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

# Train one-class SVM and plot the outliers (outputs of prediction being equal to -1)
SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = numpy.where(pred==-1)
values = x[anom_index]

# Plot the support vectors
supvectors=SVMmodelOne.support_vectors_
plt.figure(4)
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.scatter(supvectors[:,0],supvectors[:,1],color='magenta')
plt.axis('equal')

# What if we want to have a control what is outlier? Use e.g. 5% "quantile" to mark the outliers. Every point with lower score than threshold will be an outlier.
scores = SVMmodelOne.score_samples(x)
thresh = numpy.quantile(scores, 0.05)
print(thresh)
index = numpy.where(scores<=thresh)
values = x[index]
plt.figure(5)
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')

plt.show()
