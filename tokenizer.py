
#import nltk

from nltk.tokenize import sent_tokenize, word_tokenize


#nltk.download()

#tokenizing - word tokenizers and sentence tokenizers
#lexicons and corporas
#corpora - body of text. ex: medical journals, speeches etc
#lexicon - words and their meanings [nouns verbs adj etc]

ex_text = "Hello Ms. Snigdha, how are you doing today? The weather is great and python is awesome. The sky is blue. You shouldn't eat junk."

print(sent_tokenize(ex_text))

print(word_tokenize(ex_text))

for i in word_tokenize(ex_text):
    print(i)