#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:56:24 2019

@author: snigdharao
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ex_sentence = "this is an example showing off stop word filtration."
stop_words = set(stopwords.words("english"))

#print(stop_words)

words = word_tokenize(ex_sentence)

# =============================================================================
# filtered_sentence = []
# 
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)
#         
# print(filtered_sentence)
# =============================================================================

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)