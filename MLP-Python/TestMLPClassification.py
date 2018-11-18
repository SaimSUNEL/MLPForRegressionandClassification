import math
import pickle

import cv2
import numpy as np
from sklearn import datasets


from MLPClassification import MultiHiddenLayerNN


neural_network = MultiHiddenLayerNN((784, 100,  10), 0.001, 10, 750)

print "Vector : ", neural_network.binarization(2, 3)
#exit(0)
#minist_data_load()
X, Y = [], []

print("Tamam")
renkler = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
           3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255), 6: (255, 255, 255)}

numbers = {"sifir": 0, "bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7,
           "sekiz": 8, "dokuz": 9, "A": 'A', "B": 'B', "C": "C", "D": 'D', "E": 'E', "F": 'F',
           "G": 'G', 'H': 'H', "I": 'I', "K": 'K', "M": "M", "N": "N", "P": "P", "S": "S", "T": "T",
           "V": "V", "Y": "Y", "Z": "Z", "R": "R", "CH": "CH", "SH": "SH", "U": "U", "J": "J", "L": "L"}

resim_directory = "CharacterCreater/"
rakamlar = ["sifir", "bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz", "dokuz", "A",
            "B", "C", "D", "E", "F", "G", "H", "I", "K", "M", "N", "P", "S", "T", "V", "Y", "Z",
            "R", "CH", "SH", "U", "J", "L"]
print "The pictures are being loaded..."

for kategory in rakamlar[0:10]:

    for number in range(0, 35):
        yol = resim_directory + kategory + str(number) + ".jpg"
        #print yol

        resim = cv2.imread(yol)
        arr = []

        for a in range(0, 28):
            for b in range(0, 28):

                if resim[a, b].all() > 0:
                    arr.append(1)
                else:
                    arr.append(0)

        X.append(np.array(arr, dtype=np.float32))
        Y.append(kategory)

print "data set has been loaded..."

#0 1 2
hot_vector = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
print "The pictures are being loaded..."

print "data set has been loaded..."
print "Network is being trained..."

iris = datasets.load_iris()
y = iris.target
x = iris.data
x_original = x.copy()
y_original = y.copy()

y = [hot_vector[b] for b in y]


print "Our neural network is getting trained...."
#neural_network.back_propagate( x , y )
neural_network.train(np.array(X), np.array(Y))

print("NN2 de tamam")
print ("Result NN: ", neural_network.predict([X[2]]))
print ("Result : ", Y[2])
print ("Result NN: ", neural_network.predict([X[len(X)-2]]))
print ("Result : ", Y[len(X)-2])
correct = 0
false = 0
pickle.dump(neural_network, open("MyNeuralNetwork.nn", "w"))
print "Neural network has been saved..."
for i, a in enumerate(X):

    res = neural_network.predict([a])
    print ("Result NN: ", res)
    print ("Result Y: ", Y[i])
    if Y[i] == res[0]:
        correct += 1
    else:
        false += 1

print ("Accuracy : %f", (float(correct)/(correct + false) * 100.0))
