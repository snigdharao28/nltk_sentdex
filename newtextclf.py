#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:24:41 2019

@author: snigdharao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:40:35 2019

@author: snigdharao
"""
import random
import pickle

import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()


documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )
    
for r in short_neg.split('\n'):
    documents.append( (r, "neg") )
    

#random.shuffle(documents)

#print(documents[1])

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

    
all_words = nltk.FreqDist(all_words)

# =============================================================================
# print(all_words.most_common(15))
# print(all_words["stupid"])
# =============================================================================

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

#print((find_features(movie_reviews.words('neg/cv350_22139.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)


training_set = featuresets[:10000]
test_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features(15)
print("Original Classifier accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)


# =============================================================================
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()
# =============================================================================





# =============================================================================
# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
# =============================================================================

#Multinomial NB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier,test_set)*100)

#Bernoulli NB
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, test_set)*100)

#Logistic Regression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",nltk.classify.accuracy(LogisticRegression_classifier, test_set)*100)

#SGDClassifier ( stochastic gradient descent)
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:",nltk.classify.accuracy(SGDClassifier_classifier, test_set)*100)

#Support Vector classifier
# =============================================================================
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:",nltk.classify.accuracy(SVC_classifier, test_set)*100)
# =============================================================================

#Linear SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:",nltk.classify.accuracy(LinearSVC_classifier, test_set)*100)

#NuSVC (number of units)
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:",nltk.classify.accuracy(NuSVC_classifier, test_set)*100)

#new voted classifier
voted_classifier = VoteClassifier(NuSVC_classifier, 
                                  LinearSVC_classifier, 
                                  MNB_classifier, 
                                  BNB_classifier, 
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)

#print("Classification:", voted_classifier.classify(test_set[0][0]), "Confidence %:", voted_classifier.confidence(test_set[0][0])*100)
#print("Classification:", voted_classifier.classify(test_set[1][0]), "Confidence %:", voted_classifier.confidence(test_set[1][0])*100)
#print("Classification:", voted_classifier.classify(test_set[2][0]), "Confidence %:", voted_classifier.confidence(test_set[2][0])*100)
#print("Classification:", voted_classifier.classify(test_set[3][0]), "Confidence %:", voted_classifier.confidence(test_set[3][0])*100)
#print("Classification:", voted_classifier.classify(test_set[4][0]), "Confidence %:", voted_classifier.confidence(test_set[4][0])*100)
#print("Classification:", voted_classifier.classify(test_set[5][0]), "Confidence %:", voted_classifier.confidence(test_set[5][0])*100)
