from svm import SVM
from statistical_model import Classifier
import matplotlib.pyplot as plt

# Testing Statistical Model Classifier

p,q = 13, 2
N = 800    # size of training set
T = 200    # size of testing set
classifier = Classifier(p, q, N)

# Testing set
with open("dataset.txt") as f:
    words = list(map(str.strip, f))
    sequences = [words[1+3*i] for i in range(N,N+T)]
    sites = [words[2+3*i] for i in range(N,N+T)]

positives, negatives = [], []
for i in range(T):
    seq = sequences[i]
    site = sites[i]
    for l in range(p+1, len(seq)-q+1):  # first letter discarded
        window = seq[l-p:l+q]
        if(site[l] == 'C'):
            positives.append(classifier.score(window))
        else:
            negatives.append(classifier.score(window))

# plot histograms

print("Plotting histograms of scores")

plt.subplot(2, 2, 1)
plt.ylabel('Number of words')
plt.xlabel('Score')
plt.title("Score of positive words")
plt.hist(positives, color="blue")

plt.subplot(2, 2, 2)
plt.ylabel('Number of words')
plt.xlabel('Score')
plt.title("Score of negative words")
plt.hist(negatives, color="red")

plt.subplots_adjust(bottom=0.1, right=1.8, top=1.5)
plt.show()

# Thresholding

eps_list = []
acc_list = []
fs_list = []

for i in range(-16,16):
  eps = 0.5*i
  tp = list(filter(lambda x: (x >= eps), positives))
  fn = list(filter(lambda x: (x < eps), positives))
  tn = list(filter(lambda x: (x < eps), negatives))
  fp = list(filter(lambda x: (x >= eps), negatives))
  total = len(tp)+len(fp)+len(tn)+len(fn)

  acc = (len(tp)+len(tn))/total
  ppv = len(tp)/(len(tp)+len(fp))
  tpr = len(tp)/(len(tp)+len(fn))
  fs = 2*ppv*tpr/(ppv+tpr)

  eps_list.append(eps)
  fs_list.append(fs)
  acc_list.append(acc)

# Plots

print("Plotting statistics according to choice of epsilon")

plt.subplot(2, 2, 1)
plt.ylabel('Accuracy')
plt.xlabel('Epsilon')
plt.plot(eps_list, acc_list, color="blue")

plt.subplot(2, 2, 2)
plt.ylabel('F-Score')
plt.xlabel('Epsilon')
plt.plot(eps_list, fs_list, color="red")

plt.subplots_adjust(bottom=0.1, right=1.8, top=1.5)
plt.show()

# Testing kernels

# Linear kernel
svm = SVM("Linear", [])
print("\nLinear:")
svm.fit()

# Polynomial kernel
print("\nPolynomial:")
svm = SVM("Polynomial", [1, 2])
svm.fit()

# RBF kernel
print("\nRBF:")
svm = SVM("RBF", [0.1])
svm.fit()

# Similarity kernel
print("\nSimilarity:")
svm = SVM("Similarity", ["Matrix.txt"])
svm.fit()

# Probabilistic kernel
print("\nProbabilistic:")
svm = SVM("Probabilistic", [])
svm.fit()

# Combination of similarity and probabilistic kernels
print("\nCombination:")
svm = SVM("Combination", ["Matrix.txt"])
svm.fit()

# Finding cleavage sites in sequencies with best classifier (SVM with linear kernel)

svm = SVM("Linear", [], N=800)
print("Finding cleavage sites for words in sequencies.txt using linear kernel")
print(svm.find_cleavage_sites("sequencies.txt"))