# -------------------------------------------------------
# Assignment (2)
# Written by (Mohamed Hefny, 40033382)
# For COMP 472 Section (ABIX) – Summer 2020
# --------------------------------------------------------
import pandas
import nltk_functions
import os
import json
import math
import vocab
import const
import matplotlib.pyplot as plt
from sklearn import metrics

import time

INPUT_COLUMNS = ["counter", "word", "freq_story", "prob_story", "freq_ask", "prob_ask", "freq_show", "prob_show", "freq_poll", "prob_poll"]

#intialize x,y variables for drawing frequency graphs
xticks = list()
xticks_top = list()
yticks_fmeasure = list()
yticks_accuracy = list()
yticks_recall = list()
yticks_precision = list()
yticks_top_fmeasure = list()
yticks_top_accuracy = list()
yticks_top_recall = list()
yticks_top_precision = list()

def draw_graph():
    """
    draws two subplots for each frequency methods, one for number 
    frequency and the other for percentage frequency
    """
    #create 2 subplots
    fig, axs = plt.subplots(1, 2)
    #scatter points for the first graph
    axs[0].plot(xticks,yticks_fmeasure, label = "F-measure",marker='x')
    axs[0].plot(xticks,yticks_accuracy, label= "accuracy",marker='D')
    axs[0].plot(xticks,yticks_precision,label= "precision",marker='*')
    axs[0].plot(xticks,yticks_recall, label= "recall",marker='P')

    #scatter points for the second graphs
    axs[1].plot(xticks_top,yticks_top_fmeasure, label = "F-measure",marker='x')
    axs[1].plot(xticks_top,yticks_top_accuracy,label = "accuracy",marker='D')
    axs[1].plot(xticks_top,yticks_top_precision, label = "precision",marker='*')
    axs[1].plot(xticks_top,yticks_top_recall, label = "recall",marker='P')

    #set the plot titles and legends
    plt.setp(axs[0], ylabel='F-measure')
    axs[0].set_title('Frequency number')
    axs[1].set_title('Frequency percentage')
    fig.text(0.5, 0.04, 'Words left in vocab', ha='center', va='center')
    axs[0].legend()
    axs[1].legend()

    plt.show()

def load(input):
    """
    loads the data model text
    param input: csv file to be loaded
    """
    #return the file as a pandas data frame
    return pandas.read_csv(input, sep="\s+", header=None, names=INPUT_COLUMNS)

def load_test(input):
    """
    loads the data csv file to get only the testing data
    param input: csv file
    """
    data = pandas.read_csv(input)
    data["Created At"] = pandas.to_datetime(data["Created At"])
    data["Year"] = data["Created At"].dt.year
    data["Title"] = data["Title"].replace('\s+',' ',regex=True).str.lower()
    data["Title"] = data["Title"].str.strip()
    return data[data["Year"] == 2019] 


def classify(csv, input, output,stop_words,frequency_filter=False,word_length_filter=False,top_freq=False,baseline=False):
    """
    classifys the testing data
    param csv: csv file to extract the data from 
    param input: input model to classify upon
    param output: file to output the classification
    param stop_words: set of stop words
    param frequency_filter: if frequency filter experiment
    param word_length_filter: if word length filter experiment
    param top_freq: if top frequency from the infrequent filter  experiment
    param baseline: if baseline experiment
    """
    #load the model data to classify based on
    model_data = load(input)

    # if proccessed lemmatized testing titles doesnt exist, lemmatize titles and save a h5 file
    ##if not exists:
    print('processing, classification files')        
    test_set = load_test(csv)
    test_set['Titlelemma'] = test_set.apply(lambda row: nltk_functions.corpus_lemmatization(corpus_string=row['Title'],stopwords=stop_words,
    
                                                                                          word_length_filtering=word_length_filter,baseline=baseline),axis=1)
    #To store the proccessed data in a h5 file, not supported by older pandas version
    """
    ## exists = os.path.isfile('processed_data.h5')
    ## data_store = pandas.HDFStore('processed_data.h5')
    ## data_store['preprocessed_df'] = test_set  
    ## data_store.close()
    ## else:
    ##     print('File, pre-processed')
    ##     data_store = pandas.HDFStore('processed_data.h5')
    ##     test_set = data_store['preprocessed_df']
    ##     data_store.close()
    """
    count = 0
    voc_size = len(model_data)
    with open("output/post_count.json") as json_file:
        all_post = json.load(json_file)

    #get the length of post counts
    post_story = all_post["story"]
    post_ask = all_post["ask"]
    post_show = all_post["show"]
    post_poll = all_post["poll"]
    total_post = all_post["total"]

    #get a list of the actual post types
    true_list = test_set["Post Type"].tolist()

    if not frequency_filter:
        f = open(output, "w+",encoding="utf-8")

    prediction_list = list()

    #iterate through the test data tiles
    for i,row in test_set.iterrows():
        count+=1
        title = row["Titlelemma"]
        title_plain = row["Title"] 
        post_type = row["Post Type"]
        #word_list = nltk_functions.corpus_lemmatization(title,stop_words,word_length_filter,baseline)

        #get the words from the model that exist in the title
        word_probab = model_data[model_data["word"].isin(title)]

        #set the propability of story types if exists, otherwise -inf
        p_story = float("-inf") if post_story == 0 else math.log(post_story/total_post, 10)
        p_ask = float("-inf") if post_ask == 0 else math.log(post_ask/total_post, 10)
        p_show = float("-inf") if post_show == 0 else math.log(post_show /total_post, 10)
        p_poll =  float("-inf") if post_poll == 0 else math.log(post_show /total_post, 10)
        #print(title)
        #check if the title consists of certain words, assign a higher probabilty for these words
        if ("ask hn" in title_plain  and post_type=="ask_hn")  or ('ask_hn' in title_plain and post_type == 'ask_hn'):
            p_ask = 100
        if ("show hn" in title_plain and post_type=="show_hn")  or ('show_hn' in title_plain and post_type == 'show_hn'):
            p_show = 100
        if ("poll" in title and post_type=="poll"):
            p_poll = 100

        #calculate the total post type score for each word in the title
        for word in title:
            word_prob = word_probab[word_probab["word"]==word]
            if not word_prob.empty:
                p_story += math.log(word_prob["prob_story"],10) if post_story != 0 else 0
                p_ask += math.log(word_prob["prob_ask"],10) if post_ask != 0 else 0
                p_show += math.log(word_prob["prob_show"],10) if post_show != 0 else 0
                p_poll += math.log(word_prob["prob_poll"],10) if post_poll != 0 else 0
        end1 =time.time()
        scores = [p_story,p_ask,p_show,p_poll]
        types = ["story","ask_hn","show_hn","poll"]

        #get the most probable type
        max_index = scores.index(max(scores))
        prediction = types[max_index]

        if not frequency_filter:
            result = "right" if prediction == post_type else "wrong"
            line ="%d  %s  %s  %f  %f  %f  %f  %s  %s\n" % (count, title_plain, prediction,
                                                            p_story, p_ask, p_show, p_poll,
                                                            post_type, result)
            f.write(line)
        prediction_list.append(prediction)

    if frequency_filter:
        #get performance metrics for the  frequency filter classification
        fScore, accuracy, recall, precision = get_metrics(true_list,prediction_list)

        #if generating the percentage frequency graph
        if top_freq:
            xticks_top.append(voc_size)
            yticks_top_fmeasure.append(fScore)
            yticks_top_accuracy.append(accuracy)
            yticks_top_precision.append(precision)
            yticks_top_recall.append(recall)
        #if generating the numerical frequency graph
        else:
            xticks.append(voc_size)
            yticks_fmeasure.append(fScore)
            yticks_accuracy.append(accuracy)
            yticks_precision.append(precision)
            yticks_recall.append(recall)
    

def get_metrics(true_list,prediction_list):
    """
    provides the performance metrics from two equally sized lists of data
    true_list: list of the true classifications
    prediction_list: list of the predicted classifications
    """
    #mcm = metrics.confusion_matrix(true_list,prediction_list, labels=["story","show_hn","ask_hn","poll"])
    #print(mcm)
    #print(metrics.classification_report(true_list,prediction_list, digits=7))

    #generate 4 main metrics, fscore, accuracy, recall, precision
    fScore = metrics.f1_score(true_list,prediction_list,average="weighted")
    accuracy = metrics.accuracy_score(true_list,prediction_list,normalize=True)
    recall = metrics.recall_score(true_list,prediction_list,average='weighted')
    precision = metrics.precision_score(true_list,prediction_list,average='weighted')
    return (fScore,accuracy,recall,precision)


def predict_classifier(stopword_removal=False, word_length_filter=False, frequency_filter=False, input_filename="", output_filename="",top_freq=False,baseline=True):
    """
    sets constants for each experiment, such as model file names and prediction file names
    param stopword_removal: if stopword experiment
    param word_length_filter: if word length filter experiment
    param frequency_filter: if frequency filter experiment
    param input_file: the name of the model file to be analyzed 
    param output_filename: the name of the outputted classification file
    param top_freq: if using the infrequent word filtering experiment and analyzing the top % frequency
    param baseline: if using the baseline experiment
    """
    print("Classifying...")
    model_file = const.MODEL_FILE
    prediction_file = const.BASELINE_RESULT
    stop_words = set()
    
    #if stopword experiment, generate the stopword classification
    if stopword_removal:
        print("Classifying using experiment 1 (stop-word)")
        model_file = const.STOPWORD_MODEL
        prediction_file = const.STOPWORD_RESULT
        stop_words = vocab.get_stopwords(const.STOPWORDS)
    
    #if word length filter experiment, generate the word length filter classification
    if word_length_filter:
        print("classifying using experiment 2 (word-length)")
        model_file= const.WORDLENGTH_MODEL
        prediction_file= const.WORDLENGTH_RESULT
        stop_words = set()
    
    #if frequency filter experiment, generate the frequency filter classification
    if frequency_filter:
        print("clasifying using experiment 3 (frequency-filter)")
        model_file = input_filename
        prediction_file = output_filename
        stop_words = set()

    classify(const.INPUT_TEST,model_file,prediction_file,frequency_filter=frequency_filter,word_length_filter=word_length_filter,stop_words=stop_words,top_freq=top_freq,baseline=baseline)

