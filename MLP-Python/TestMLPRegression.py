from sklearn import datasets
from MLPRegression import MultiHiddenLayerNN
import numpy as np

diabetes = datasets.load_diabetes()

network = MultiHiddenLayerNN((10, 80, 50, 1), 0.001, 10, 200)


print diabetes.keys()
print diabetes["DESCR"]
print diabetes["data"]
print "Target : "
print diabetes["target"]
X = diabetes["data"]
Y = diabetes["target"]
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

mean = np.mean(Y)
std = np.std(Y)




print X.shape
print Y.shape
print Y

S = diabetes["target"]
range_ = S.max() - S.min()
label = (S-mean)/(S.max()-S.min())
print "label max : ", label.max()

network.train(X, (Y-mean)/(Y.max()-Y.min()))
prediction = network.predict(X)

for i in range(len(X)):
    print "Network predicted : ", prediction[i], " - True val : ", (Y[i]-mean)/range_



exit()
