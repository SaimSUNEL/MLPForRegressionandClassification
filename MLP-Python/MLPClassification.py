import numpy as np
from scipy.special import expit
import math

class MultiHiddenLayerNN:

    def create_weight_matrix(self, input_size, output_size):

        init_range = math.sqrt(3.0/input_size)
        return np.random.uniform(-init_range, init_range, size=(input_size, output_size))


        pass

    def __init__(self, neural_network_structure, learning_rate, batch_size, max_iteration=200):
        #We are creating our class variables...
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.neural_network_structure = list(neural_network_structure)
        self.learning_rate = learning_rate
        self.neural_network_parameters = [[x] for x in range(0, len(neural_network_structure)-1)]
        self.bias_parameters = [[x] for x in range(0, len(neural_network_structure)-1)]

        for layer_index in range(len(neural_network_structure)-1):
            self.neural_network_parameters[layer_index] = \
                self.create_weight_matrix(neural_network_structure[layer_index],
                                          neural_network_structure[layer_index+1])
            self.bias_parameters[layer_index] = self.create_weight_matrix(1, neural_network_structure[layer_index+1])

        self.immediate_result = [[0] for x in range(0, len(neural_network_structure)-1)]
        self.weight_updates = [[0] for x in range(len(neural_network_structure)-1)]
        self.bias_updates = [[0] for x in range(len(neural_network_structure)-1)]

        #Before training, we need to convert the discrete labels into hot vector representation
        #for each discrete label and their corresponding hot vector representation are kept as a dictionary
        self.label_hot_vector = {}
        #In this variable, unique discrete labels that are used in training are kept...
        #This is used when the final result is extracted...
        #The max valued index of the result is used in this array and the corresponding result is returned...
        self.unique_labels = None

    #This function creates hot vector.
    #value parameter indicates which index will be 1.0
    #size parameter indicates the size of the array...
    #example (value=2, size=5) => [0.0, 0.0, 1.0, 0.0,0.0]
    @staticmethod
    def binarization(value, size):
        #The values will be held in this array..
        array = [0 for i in range(size)]
        array[value] = 1.0

        return array

    #This function trains nn by given train set and train label
    def train(self, train_set, train_label):

        #We are getting the unique discrete labels from the train labels
        #Those labels are converted to hot vector...
        labels = np.unique(train_label)
        self.unique_labels = labels

        #Input and output size's of the nn are extracted from entered arrays...
        inputvectorsize = len(train_set[0])
        outputvectorsize = len(labels)


        #corresponding hot vector representation for each unique labels is extracted..
        #Index of the discrete label and the hot vector's max index is the same
        for val in range(0, len(labels)):
            self.label_hot_vector[labels[val]] = self.binarization(val, outputvectorsize)




        #We are creating a copy of the train labels not to effect the original one...
        train_label_copy = [aa for aa in train_label]
        #And we are replacing the hot vector representation with the corresponding label
        for val in range(0, len(train_label)):
            train_label_copy[val] = self.label_hot_vector[train_label[val]]

        train_label_copy = np.array(train_label_copy)

        indices = [a for a in range(len(train_set))]
        np.random.shuffle(indices)

        train_set = train_set[indices]
        train_label_copy = train_label_copy[indices]


        for it in range(0, self.max_iteration):

            #After each iteration over the train set, we are calculating the overall error on the train set...
            total_loss = 0.0
            re = self.feed_forward(train_set)
            error = train_label_copy - re
            total_loss = -np.sum(train_label_copy*np.log(re))
            print( "Iteration ", it, " loss : ", total_loss)
            for batch_index in range(0, len(train_set), self.batch_size):
                X_batch = train_set[batch_index: batch_index+self.batch_size]
                Y_batch = train_label_copy[batch_index: batch_index+self.batch_size]
                self.back_propagate(X_batch, Y_batch)

    #This function takes an array of sample and returns an array consisting of results of each individual
    #sample value...
    def predict(self, test_samples):
        results = []
        for sample in test_samples:
            #The max valued index will correspond to result of the nn
            results.append(self.unique_labels[np.argmax(self.feed_forward(sample))])

        return results

    #This function takes and array and for each element it applies softmax and returns array of results...
    @staticmethod
    def softmax(output):
        scorematexp = np.exp(output)

        return scorematexp / np.sum(scorematexp, axis=1).reshape((scorematexp.shape[0], 1))

    #This functions takes an input sample and performs required paramater multiplications
    #Returns a result by applying a softmax at the end...
    def feed_forward(self, sample):


        result = None
        ones = np.ones((sample.shape[0], 1))

        for layer_index in range(0, len(self.neural_network_parameters)-1):
            try:
                result = np.matmul(sample, self.neural_network_parameters[layer_index])
            except Exception as err:
                print("Error : ", err)
                print("Layer index ", layer_index)
                exit()


            result = result + np.matmul(ones, self.bias_parameters[layer_index])
            result = expit(result)


            self.immediate_result[layer_index+1] = result.copy()
            sample = result



        result = np.matmul(result, self.neural_network_parameters[len(self.neural_network_parameters)-1])
        result +=  np.matmul(ones, self.bias_parameters[len(self.neural_network_parameters) -1])

        return self.softmax(result)

    #this function is not used in the program, instead expit function is used...
    #to give information it is kept...
    @staticmethod
    def sigmoid(result):
        return 1.0 / (1.0 + np.exp(-result))

    def back_propagate(self, x_, y_):


        re = self.feed_forward(x_)
        error = y_ - re

        ones = np.ones((1, x_.shape[0]))

        self.immediate_result[0] =  x_

        for layer_index in range(len(self.neural_network_parameters) - 1, -1, -1):
            if(layer_index == len(self.neural_network_parameters)-1):
                f = np.transpose(self.immediate_result[layer_index])

                self.weight_updates[layer_index] = np.matmul(self.learning_rate * f,
                                                       error)
                self.bias_updates[layer_index] = np.matmul(ones*self.learning_rate, error)
                continue
            error = np.matmul(error, np.transpose(self.neural_network_parameters[layer_index+1]))

            error = error * self.immediate_result[layer_index+1] * \
                    (1-np.array(self.immediate_result[layer_index+1]))

            f = np.transpose(self.immediate_result[layer_index])
            self.weight_updates[layer_index] = np.matmul(self.learning_rate*f, error)
            self.bias_updates[layer_index] = np.matmul(ones*self.learning_rate, error)

        # All the parameter updates must be performed at the end...
        for layer_index in range(0, len(self.weight_updates)):
            self.neural_network_parameters[layer_index] += self.weight_updates[layer_index]
            self.bias_parameters[layer_index] += self.bias_updates[layer_index]




