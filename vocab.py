# -------------------------------------------------------
# Assignment (2)
# Written by (Mohamed Hefny, 40033382)
# For COMP 472 Section (ABIX) – Summer 2020
# --------------------------------------------------------
import const
import pandas as pd
import json
import string
import nltk_functions
import nltk as nl

posts = pd.read_csv(const.INPUT_FILE)
def buildModel(output_file=None, stopwords=None, word_length_filtering=False, smoothing=0,baseline=False):
    """
    Builds the vocabulary on which the model will be built on
    param output_file: model text file to be outptted depending on the experiment
    param stopwords: list of stopwords to not consider in the vocabulary
    parm word_length_filtering: if using the word length filtering experiment
    param smoothing: smoothing variable to be applied
    param baseline: if using the baseline experiment 
    """
    post_types={}
    print("Loading data from file %s ..." % "csv file")
    posts["Title"]= posts["Title"].str.lower()

    #Training set from 2018
    posts["Created At"] = pd.to_datetime(posts["Created At"])
    trainingSet = posts[posts["Created At"].dt.year == 2018]
    #trainingSet = posts

    #Get unique post types
    unique_posts = posts["Post Type"].unique()

    #Get the unique types fro from post types
    for post in unique_posts:
        post_data = trainingSet[trainingSet["Post Type"]==post]
        post_types[post] = post_data
        
    
    #Start the lemmatization process for the titles
    print("Lemmatization...")
    story_length = 0 
    titlesStory= list()
    ask_hn_length = 0
    titleAskhn = list()
    show_hn_length = 0
    titleShowhn = list()
    poll_length=0
    titlePoll = list()
    #lemmatize for story type if story exists 
    if "story" in post_types:
        story_length = len(post_types.get("story"))
        titlesStory = lemmatization(post_types.get("story"), stopwords, word_length_filtering,baseline)

    #lemmatize for ask_hn type if ask_hn exists
    if "ask_hn" in post_types:
        ask_hn_length = len(post_types.get("ask_hn"))
        titleAskhn = lemmatization (post_types.get("ask_hn"), stopwords, word_length_filtering,baseline)

    #lemmatize for show_hn type of show_hn exists
    if "show_hn" in post_types:
        show_hn_length = len(post_types.get("show_hn"))
        titleShowhn = lemmatization (post_types.get("show_hn"), stopwords, word_length_filtering,baseline)
    
    #lemmatize for poll type if poll  exists
    if "poll" in post_types:
        poll_length = len(post_types.get("poll"))
        titlePoll = lemmatization (post_types.get("poll"), stopwords, word_length_filtering,baseline)
    
    #save the lenghts of each vocab type in a json file
    export_post_count(story_length, ask_hn_length, show_hn_length, poll_length, len(trainingSet))

    #save the removed vocablary
    if baseline:
        nltk_functions.remove_vocab()

    #create a set and of the vocabulary
    vocabulary_set = set(titlesStory)
    vocabulary_set.update(titleAskhn) 
    vocabulary_set.update(titleShowhn)
    vocabulary_set.update(titlePoll)

    sorted_vocab = list(vocabulary_set)
    sorted_vocab.sort()

    #output the vocablary to a file
    with open(const.VOCABULARY, "w",encoding='utf-8') as outfile:
        outfile.write("\n".join(sorted_vocab))

    outputModel(output_file,sorted_vocab,titlesStory,titleAskhn,titleShowhn,titlePoll,smoothing)


def outputModel(output_file, sorted_vocab, titlesStory, titlesAskhn, titlesShowhn, titlesPoll, smoothing=0):
    """
    outputs the model to a text file formatted
    param output_file: file to output the data
    param sorted_vocab: vocabulary of the model sorted
    param titlesStory: all words in the story post type
    param titlesAskhn: all words in the askhn post type
    param titlesShowhn: all words in the showhn post type
    param titlesPoll: all words in the poll post type
    param smoothing: the value of smoothing to apply to the model
    """
    fl = open(output_file, 'w+', encoding='utf-8')

    #get the length of vocablary in the tiles of each type
    totalStory = len(titlesStory)
    totalAskhn = len(titlesAskhn)
    totalShowhn = len(titlesShowhn)
    totalTitlesPoll = len(titlesPoll)
    total = len(sorted_vocab)

    export_wordcount(totalStory,totalAskhn,totalShowhn,totalTitlesPoll,total)

    #create a frequency distribution of each unique title
    storyFreq = nl.FreqDist(titlesStory)
    askhnFreq = nl.FreqDist(titlesAskhn)
    showhnFreq = nl.FreqDist(titlesShowhn)
    titlespollFreq = nl.FreqDist(titlesPoll)

    #calculate probabilities of each word in the vocabulary, using smoothing
    for i in range(total):
        w=sorted_vocab[i]
        freq_story = 0
        freq_askhn = 0
        freq_showhn = 0
        freq_poll = 0
        if storyFreq.__contains__(w):
            freq_story = storyFreq.get(w)
        prob_story = (freq_story + smoothing) / (totalStory + total*smoothing) if totalStory != 0 else float('-inf')
        
        if askhnFreq.__contains__(w):
            freq_askhn = askhnFreq.get(w)
        prob_askhn = (freq_askhn + smoothing) / (totalAskhn + total*smoothing) if totalAskhn != 0 else float('-inf')
       
        if showhnFreq.__contains__(w):
            freq_showhn = showhnFreq.get(w)
        prob_showhn = (freq_showhn + smoothing) / (totalShowhn + total*smoothing) if totalShowhn != 0 else float('-inf')

        if titlespollFreq.__contains__(w):
            freq_poll = titlespollFreq.get(w)
        prob_titlespoll = (freq_poll + smoothing) / (totalTitlesPoll + total*smoothing) if totalTitlesPoll != 0 else float('-inf')

        line = "%d  %s  %d  %.10f  %d  %.10f  %d  %.10f  %d  %.10f\n" % (i + 1, w,
                                                                                 freq_story, prob_story,
                                                                                 freq_askhn, prob_askhn,
                                                                                 freq_showhn, prob_showhn,
                                                                                 freq_poll, prob_titlespoll)
        fl.write(line)
    fl.close()

def lemmatization(trainingSet, stopwords=None, word_length_filtering=False,baseline=False):
    """
    lemmatizes all the words in a title type
    param trainingSet: pandas dataframe of the training set
    param stopwords: set of stopwords to execlude from the vocabulary
    parm word_length_filtering: if using the word length filtering experiment
    param baseline: if using the baseline experiment 
    """
    allTitles = trainingSet["Title"].str.cat(sep=' ')
    return nltk_functions.corpus_lemmatization(allTitles,stopwords,word_length_filtering,baseline)

def export_post_count(total_story, total_ask, total_show, total_poll, total):
    """
    exports the number of titles in each post type
    param total_story: total number of story titles
    param total_ask: total number of ask titles
    param total_show: total number of show titles
    param total: total of all the titles
    """
    stat = {
        "story": total_story,
        "ask": total_ask,
        "show": total_show,
        "poll": total_poll,
        "total": total
    }
    with open('output/post_count.json', 'w') as outfile:
        json.dump(stat, outfile)

def export_wordcount(totalStory,totalAskhn,totalShowhn,totalTitlesPoll,total):
    """
    exports the number of words in each post type
    param  totalStory: total number of words in story
    param totalAskhn: total number of words in ask
    param totalShowhn: total number of words in show
    param totalTitlesPoll: total number of words in poll
    param total: total number of all the words
    """
    stat = {
       "story":totalStory,
       "ask":totalAskhn,
       "show":totalShowhn,
       "poll":totalTitlesPoll,
       "total":total    
    }
    with open('output/word_count.json','w') as outfile:
        json.dump(stat, outfile)

def get_stopwords(filename):
    """
    gets all the stopwords from a given file
    param filename: file name to get the stopwords from
    """
    a_file = open(filename)
    file_contents = a_file.read()
    return file_contents.splitlines()

def build_model(stopwords_removal=False,word_length_filter=False,frequency_filter=False,baseline=False):
    """
    sets the output file and other constants for each experiment
    param stopwords_removal: if stopwords removal experiment
    param word_length_filter: if word length filter experiment
    param frequency_filter: if frequency filter experiment
    param baseline: if baseline experiment
    """
    output_file = const.MODEL_FILE
    smoothing_delta = 0.5
    frequency_built = False

    stopwords = set()

    #if using stopword removal model, use the stopword model file
    if stopwords_removal:
        print("Outputting model with stopword removal...")
        output_file = const.STOPWORD_MODEL
        stopwords = get_stopwords(const.STOPWORDS)
    
    #if using the word length filtering model, using the word length filter model file
    if word_length_filter:
        print("Outputting model with word length filter")
        output_file = const.WORDLENGTH_MODEL

    #if using the frequency filter, use the diffirent frequencies and use 10 diffirent model files + baseline model files
    if frequency_filter:
        frequency = [1,5,10,15,20,0.05,0.1,0.15,0.2,0.25]
        output_file = ["freq_filter/frequencyfilter-model_1.txt", "freq_filter/frequencyfilter-model_5.txt","freq_filter/frequencyfilter-model_10.txt",
                        "freq_filter/frequencyfilter-model_15.txt","freq_filter/frequencyfilter-model_20.txt","freq_filter/frequencyfilter-model-top_5.txt", 
                        "freq_filter/frequencyfilter-model-top_10.txt","freq_filter/frequencyfilter-model-top_15.txt",
                        "freq_filter/frequencyfilter-model-top_20.txt","freq_filter/frequencyfilter-model-top_25.txt"]
        stopwords=set()
        for i in range(10):
            print("Outputting model with %s" %output_file[i])
            buildFrequencyModel(output_file[i], frequency[i])
        frequency_built=True

    if not frequency_built:
        buildModel(output_file,stopwords, word_length_filter,smoothing_delta,baseline)

def buildFrequencyModel(output_file,frequency):
    """
    builds the vocabulary model for the last experiment (infrequent word filtering)
    param output_file: file to output the model to
    param frequency: frequency removal of words (e.g. 1 for words with frequency 1, 0.25 for the 25 most frequent words)
    """
    INPUT_COLUMNS = ["counter", "word", "freq_story", "prob_story", "freq_ask", "prob_ask", "freq_show", "prob_show", "freq_poll", "prob_poll"]
    wordsRemove = list()
    vocab = pd.read_csv(const.MODEL_FILE , sep="  ", header=None, names = INPUT_COLUMNS, engine="python")

    #create a new coloumn that adds up all post type frequencies
    vocab['freq_total'] = vocab["freq_story"] + vocab["freq_ask"] + vocab["freq_show"] + vocab["freq_poll"]

    #determine the new vocabulary for frequency numbers
    if frequency >= 1:
        for i,row in vocab.iterrows():
            #prob = [row["freq_story"], row["freq_ask"], row["freq_show"], row["freq_poll"]]
            if frequency == 1 and (row['freq_total'] ==1):
                wordsRemove.append(i)

            if frequency > 1 and (row['freq_total'] <=frequency):
                wordsRemove.append(i)
        vocab = vocab.drop(vocab.index[wordsRemove])
    else:
    #determine the new vocabulary for frequency percentage
        vocab.sort_values(by="freq_total", ascending=False,inplace=True)
        top_freq = int(len(vocab)*frequency)
        drop_indexes= list(range(0,top_freq+1))
        vocab = vocab.drop(vocab.index[drop_indexes])
        vocab.sort_values(by="word", inplace=True)

    #output each frequency variation to a model file
    f = open(output_file, "w+")
    for i in range(len(vocab)):
        record = vocab.iloc[i]
        record_line = "%d  %s  %d  %.10f  %d  %.10f  %d  %.10f  %d  %.10f\n" % (i + 1, record["word"],
                                                                             record["freq_story"], record["prob_story"],
                                                                             record["freq_ask"], record["prob_ask"],
                                                                             record["freq_show"], record["prob_show"],
                                                                             record["freq_poll"], record["prob_poll"])
        f.write(record_line)
    f.close()