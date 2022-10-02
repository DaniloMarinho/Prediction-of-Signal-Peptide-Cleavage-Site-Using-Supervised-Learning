import numpy as np
from sklearn import mixture
import math

class Classifier:
    def __init__(self, p, q, N=None, path="dataset.txt"):
        with open(path) as f:
            words = list(map(str.strip, f))
            self.N = N
            if N == None: self.N = len(words)//3
            self.sequences = [words[1+3*i] for i in range(self.N)]
            self.sites = [words[2+3*i] for i in range(self.N)]
            self.p = p
            self.q = q
            self.frequencies = {}
            self.background_frequencies = {}
            self.set_frequencies()
            self.set_background_frequency()

    def set_frequencies(self):
        # total of counts
        counts = {}

        # initialize all with zeros
        for character in range(65,91):
            l = [0 for i in range(self.p+self.q)]
            counts[chr(character)] = l

        # update counts
        for i in range(self.N):
            seq = self.sequences[i]
            site = self.sites[i].find('C')
            for l in range(-self.p, self.q):
                letter = seq[site+l]
                counts[letter][self.p+l] += 1

        # divide by N
        for character in range(65,91):
            self.frequencies[chr(character)] = [counts[chr(character)][i]/self.N for i in range(self.p+self.q)]

    def set_background_frequency(self):
        # total of counts
        counts = {}
        total_length = 0

        # initialize all with zeros
        for character in range(65,91):
            counts[chr(character)] = 0

        # update counts, not taking first letter into account to avoid bias
        for i in range(self.N):
            seq = self.sequences[i][1:]
            for l in seq:
                counts[l] += 1
                total_length += 1

        # divide by the total length
        for character in range(65,91):
            self.background_frequencies[chr(character)] = counts[chr(character)]/total_length

    def freq(self, a, i):
        return self.frequencies[a][self.p+i]

    def g(self, a):
        return self.background_frequencies[a]

    def s(self, a, i):
        # Pseudocount
        if(self.freq(a,i) == 0):
            d = 1/(self.p+self.q)   # uniform distribution in i
            alpha = 0.5
            return math.log(self.freq(a,i) + alpha)-math.log(self.g(a) + d*alpha)
        return math.log(self.freq(a,i))-math.log(self.g(a))
    
    def score(self, word):
        return sum([self.s(word[i], i-self.p) for i in range(self.p+self.q)])