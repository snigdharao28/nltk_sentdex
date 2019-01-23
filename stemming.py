#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:10:06 2019

@author: snigdharao
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

# =============================================================================
# for w in example_words:
#     print(ps.stem(w))
# =============================================================================

new_text = "it is very important to be pythonly while you are pythoning with python. All pythoners have pythoned atleast once"
words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))