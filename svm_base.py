import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class SVMBase:
    def __init__(self, path):
        self.path = path

    # functions for treating data

    def make_DF(self):
        data = self.pre_proc(13, 2)
        (X_train, y_train) = list(zip(*data))
        X_train = np.array(X_train)
        df = pd.DataFrame(X_train)
        df['cleavages']=y_train
        df.to_csv('proc_data.csv', index=False)

    def encoding(self, word):
        enc = np.array([])
        for l in word:
            letter_i = ord(l) - ord('A')
            enc = np.concatenate((enc, np.eye(1, 26, letter_i)[0]))

        return enc

    def decoding(self, arr):
        dec = ""
        n = arr.shape[0]//26
        for i in range(n):
            for j in range(26*i, 26*(i+1)):
                if(arr[j] == 1):
                    dec += chr(65+j%26)
                    break
        return dec
    
    def pre_proc(self, p, q):
        with open(self.path) as f:
            lines = f.readlines()

        sequences = []
        for i in range(self.N):     # takes only the specified lines
            word = lines[3*i+1]
            site = lines[3*i+2].find('C')
            for j in range(len(word)-(p+q)):
                sequences.append((self.encoding(word[j:j+p+q]),(1 if site == j+p else 0)))

        return sequences

    def shuffle_and_balance(self, df, count_pos, count_neg):
        '''
        Given a dataframe, returns shuffled and balanced data
        '''

        # shuffles data
        df = shuffle(df)

        X_train = np.array(df.drop('cleavages', axis=1))
        y_train = np.array(df['cleavages'])

        count_pos = count_pos
        count_neg = count_neg
        pos = []    # array of positions to be kept

        for i in range(len(X_train)):
            if y_train[i] == 1 and count_pos > 0:
                pos.append(i)
                count_pos -= 1
            if y_train[i] == 0 and count_neg > 0:
                pos.append(i)
                count_neg -= 1

        # re-shuffle
        df = shuffle(df.iloc[pos])

        return df

    # functions for similarity kernel

    def parseMatrix(self, file):
        with open('Matrix.txt') as f:
            lines = f.readlines()
            lines = [lines[i][2:] for i in range(len(lines))]

        Letters = lines[0].split()
        M = [[int(n) for n in l.split()] for l in lines[1:]]

        return M, Letters

    def train_test_split(self, dataset, p):
        t = int(len(dataset)*p)
        return dataset[:t], dataset[t:]

    def sim(self, a, b , M, Letters):
        sum=0
        dict_ = dict(zip(Letters, list(range(len(Letters)))))
        for i in range(len(a)):
            sum += M[dict_[a[i]]][dict_[b[i]]]

        return sum

    def K_sim(self, X, Y):
        A = [self.decoding(a) for a in Y]
        B = [self.decoding(b) for b in X]
        M, Letters = self.parseMatrix(self.args[0])
        R = [[self.sim(a,b,M,Letters) for a in A] for b in B]

        return np.array(R)

    # functions for probabilistic kernel

    def s(self, a,i):
        return self.classifier.s(a,i)

    def phi(self, i,x,y):
        if x != y:
            return self.s(x,i) + self.s(y,i)
        return self.s(x,i) + math.log(1+math.exp(self.s(x,i)))

    def K_vector(self,x,y):
        a = self.decoding(x)
        b = self.decoding(y)
        return math.exp(sum([self.phi(i,a[self.p+i],b[self.p+i]) for i in range(-self.p,self.q)]))

    def K_prob(self,X,Y):
        '''
        Creates Kernel Matrix by setting each entry at a time
        A parallellized calculation could improve speed
        '''
        row_X, _ = X.shape
        row_Y, _ = Y.shape
        V = np.zeros((row_X, row_Y))
        for i in range(row_X):
            for j in range(row_Y):
                V[i][j] = self.K_vector(X[i].astype(int), Y[j].astype(int))
        return V

    # Combination of similarity and probabilistic kernels

    def K_comb(self,X,Y):
        return self.K_prob(X,Y) + self.K_sim(X,Y)

    # Function for returning accuracy, precision, recall and F-Score

    def statistics(self, prediction, test):
        total_p= np.sum(prediction == np.ones(prediction.shape))
        total_n = np.sum(prediction == np.zeros(prediction.shape))

        tp = np.dot(prediction, test)
        tn = np.dot(np.ones(prediction.shape)-prediction, np.ones(test.shape)-test)
        fp = total_p - tp
        fn = total_n - tn

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        if(tp+fp != 0):
            precision = tp/(tp+fp)
        else:
            precision = -1
        if(tp+fn != 0):
            recall = tp/(tp+fn)
        else:
            recall = -1
        if(tp+fn !=0 and tp+fp !=0 and precision+recall != 0):
            f1 = 2*(precision*recall)/(precision+recall)
        else:
            f1 = -1

        return accuracy, precision, recall, f1




