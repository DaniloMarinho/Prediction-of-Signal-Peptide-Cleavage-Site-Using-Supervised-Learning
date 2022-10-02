from sklearn import svm
import numpy as np
import pandas as pd
import os
from statistical_model import Classifier
from svm_base import SVMBase
from sklearn.utils import shuffle

class SVM(SVMBase):
    """
    Correct usage in the main function is a call of the  fit function with arguments (Kernel, parameters = [])
    Supported Kernels and their respective expected parameters:
    
        - "Linear"        | with parameter array []
    
        - "Polynomial"    | with parameter array [c, d]  - c and d being real numbers in the formula kernel(x,y) = (x*y.t+c)^d
        
        - "Similarity"    | with parameter array [file]  - file being a string containing the path of the Matrix file to be used as Similarity Matrix
            - Matrix file should be formated according to the Matrix.txt file in the original Project Folder
        
        - "Probabilistic" | with parameter array []

        - "Combination"   | with parameter array [alpha] - 

    Parameters of the class:
    - path: path of the original dataset
    - N: number of sequences (not words) of the dataset to be used in training + testing
    - split: proportion of the dataset to be used as training set
    """
    def __init__(self, Kernel_type, a, p=13, q=2, N=1000, split=0.8, path="dataset.txt"):
        super().__init__(path)
        self.type = Kernel_type
        self.args = a
        self.p = p
        self.q = q
        self.N = N  # number of sequences (not words!) of the dataset to be used (training + testing)
        self.split = split  # split proportion between training and testing
        # initialize classifier wth training set for probabilistic kernel
        # takes  the same part of the dataset as svm.fit
        self.classifier = Classifier(self.p, self.q, N=int(split*N), path=path)

        if not os.path.exists('proc_data.csv'):
            self.make_DF()

    def fit(self):

        df = pd.read_csv('proc_data.csv')

        # Shuffle and balance data - improves F-Score
        # df = self.shuffle_and_balance(df, count_pos=100, count_neg=2000)

        X_data = np.array(df.drop('cleavages', axis=1))
        y_data = np.array(df['cleavages'])

        # split
        X_train, X_test = self.train_test_split(X_data, self.split)
        y_train, y_test = self.train_test_split(y_data, self.split)

        model = svm.SVC(kernel = self.Op, C=1, gamma = 'scale', shrinking = False)
        model.fit(X_train,y_train)

        # predict
        predictions = model.predict(X_test)
        accuracy, precision, recall, f1 = self.statistics(predictions, y_test)

        print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF-Score: {f1}")

        return

    def Op(self, X, Y):

        if(self.type == "RBF"):
            X_norm = np.sum(X ** 2, axis=-1)
            Y_norm = np.sum(Y ** 2, axis=-1)
            K = np.exp(-self.args[0] * (np.tile(X_norm[:, None], (1, Y.shape[0])) + np.tile(Y_norm[None, :], (X.shape[0], 1)) - 2 * np.dot(X, Y.T)))

            return K

        elif(self.type == "Linear"):
            return np.dot(X,Y.T)
        
        elif (self.type == "Polynomial"):
            return np.power(np.dot(X,Y.T)+self.args[0]*np.ones((X.shape[0],Y.T.shape[1])),self.args[1])

        elif (self.type == "Similarity"):
            return self.K_sim(X,Y)

        elif (self.type == "Probabilistic"):
            return self.K_prob(X,Y)
        
        elif (self.type == "Combination"):
            return self.K_comb(X,Y)

    def cleavage_sites(self, sequence):
        '''
        Identify cleavage sites of a given sequence
        input: sequence of amino acids (String)
        output: array with positions of cleavage sites (pairs of amino acids)
        '''

        # trains our best classifier (Linear)
        df = pd.read_csv('proc_data.csv')
        df = self.shuffle_and_balance(df, count_pos=500, count_neg=2000)

        X_data = np.array(df.drop('cleavages', axis=1))
        y_data = np.array(df['cleavages'])
        X_train, _ = self.train_test_split(X_data, self.split)
        y_train, _ = self.train_test_split(y_data, self.split)
        model = svm.SVC(kernel = self.Op, gamma = 'scale', shrinking = False)
        model.fit(X_train, y_train)

        # tests on a given sequence
        cleavage_sites = []
        for i in range(self.p, len(sequence)-self.q):
            word = sequence[i-self.p:i+self.q]
            prediction = model.predict(self.encoding(word).reshape(1,-1))
            print(prediction)
            if(prediction[0] == 1):
                cleavage_sites.append([i,i+1])
        
        return cleavage_sites

    def find_cleavage_sites(self, path):
        with open(path) as f:
            lines = f.readlines()
        sites = []
        for seq in lines:
            sites.append(self.cleavage_sites(seq))
        return sites
