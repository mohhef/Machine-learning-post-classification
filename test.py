import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from multiprocessing import pool 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

word = "show-hn hello mohamed xd talks hello-hn"
word = word.translate(str.maketrans('', '', string.punctuation))
print(word)
words = nltk.word_tokenize(word)
print(words)
