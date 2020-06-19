# -------------------------------------------------------
# Assignment (2)
# Written by (Mohamed Hefny, 40033382)
# For COMP 472 Section (ABIX) – Summer 2020
# --------------------------------------------------------
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
import string
import const
import re

#download nltk dependencies
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

removed_words = set()
remove_punc = set('\"#$%&\'()*+,./:;<=>?@[\]^-`{|}~”,') 
def get_wordnet_pos(word):
    """
    Classifies the words to correct vocabulary mapping, e.g('mohamed->noun, play->verb)
    param word: word to be classified
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def remove_vocab():
    """
    prints all the removed vocabulary to an external file
    """
    fl = open(const.REMOVEDWORDS, 'w+', encoding='utf-8')
    for word in removed_words:
        fl.write('%s\n' %word)
    fl.close()

def corpus_lemmatization(corpus_string, stopwords=None, word_length_filtering=False,baseline=False):
    """
    sanatizes the data by removing punctuation and then tokenizing it, it then applies
    lemmatization to each word in the vocabulary, this improves the performance of the classfier
    param corpus_string: string to be tokenized
    param stopwords: set of words to not include in the vocabulary
    param word_length_filtering: if true applies word length filtering
    param baseline: if true outputs the remove_wrds and vocabulary text files
    """
    words = list()
    lemmatizor = WordNetLemmatizer()

    if stopwords is None:
        stopwords = set()

    #output the removed punctiuation to remove_words
    if baseline:
        punc_list = [removed_words.add(char) for char in remove_punc if char in corpus_string]

    #remove punctuation except for '-'
    no_punc = corpus_string.translate(str.maketrans('', '', "!\"#$%'()*+,./:;<=>?@[\]^-`{|}~”,…"))

    #To keep the '-' punctioation
    """
    ## replace '-' with '_' because word_tokenize method doesnt tokenize '_'
    ## no_punc = no_punc.replace("-","_")
    """

    #tokenize the sentence to words
    for word in nltk.word_tokenize(no_punc):
        #replace '_' back to its original form
        #word = word.replace('_','-')
        #check if the word is only alphabetical and not in stopwords
        if (word.isalpha() or re.match("\w+(?=\S*[_])([a-zA-Z_]+)",word)):

            #lemmatize each word
            word = lemmatizor.lemmatize(word, get_wordnet_pos(word))

            #check if valid length, if using the word length filtering experiment
            if valid_length(word, word_length_filtering) and word not in stopwords:
                words.append(word)
        else:
            #if word is to be removed 
            if baseline:
                removed_words.add(word)

    #lemma = [lemmatizor.lemmatize(w, get_wordnet_pos(w)) for w in words]
    return words
    #return [word for word in lemma if valid_length(word, word_length_filtering)]

def valid_length(word, word_length_filtering):
    """
    checks if a word is within the length specified in the assignment
    param word: word to be checked
    word_length_filtering: if true applies word length filtering to the word
    """
    if word_length_filtering:
        if len(word)<=2 or len(word)>=9:
            return False
    return True
