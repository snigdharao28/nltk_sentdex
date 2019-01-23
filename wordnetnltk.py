#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:27:49 2019

@author: snigdharao
"""

# =============================================================================
# from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
# from nltk.corpus import gutenberg
# 
# sample = gutenberg.raw("bible-kjv.txt")
# tok = sent_tokenize(sample)
# 
# for x in range(5):
#     print(tok[x])
# =============================================================================

from nltk.corpus import wordnet

syns = wordnet.synsets("programs")

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())

#examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
#        print("l:",l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

# =============================================================================
# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("car.n.01")
# print(w1.wup_similarity(w2))
# 
# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("cat.n.01")
# print(w1.wup_similarity(w2))
# =============================================================================

