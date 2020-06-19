# -------------------------------------------------------
# Assignment (2)
# Written by (Mohamed Hefny, 40033382)
# For COMP 472 Section (ABIX) – Summer 2020
# --------------------------------------------------------
import classifier
import vocab

#baseline experiment
def baseline():
    vocab.build_model(baseline=True)
    classifier.predict_classifier(baseline=True)

#stop word experiment
def remove_stopword():
    vocab.build_model(stopwords_removal=True)
    classifier.predict_classifier(stopword_removal=True)

#filter word length experiment
def filter_wordlength():
    vocab.build_model(word_length_filter=True)
    classifier.predict_classifier(word_length_filter=True)

#frequency filtering experiment
def frequency_filtering():
    vocab.build_model(frequency_filter=True)
    # #model_filename = []
    model_file = ["txtOutput/model-2018.txt","freq_filter/frequencyfilter-model_1.txt", "freq_filter/frequencyfilter-model_5.txt","freq_filter/frequencyfilter-model_10.txt",
                        "freq_filter/frequencyfilter-model_15.txt","freq_filter/frequencyfilter-model_20.txt"]
    for i in range(6):
        classifier.predict_classifier(frequency_filter=True, input_filename=model_file[i])
    
    model_file = ["txtOutput/model-2018.txt","freq_filter/frequencyfilter-model-top_5.txt", "freq_filter/frequencyfilter-model-top_10.txt","freq_filter/frequencyfilter-model-top_15.txt",
                         "freq_filter/frequencyfilter-model-top_20.txt","freq_filter/frequencyfilter-model-top_25.txt"]
    for i in range(6):    
        classifier.predict_classifier(frequency_filter=True, input_filename= model_file[i],top_freq=True)

    classifier.draw_graph()


prompt = input("-Enter 0 to run all the experiments \n-Enter 1 to run the baseline experiment \n-Enter 2 to run the remove word experiment \n-Enter 3 to run the word length filter experiment \n-Enter 4 to run the infrequent filtering experiment \n")
prompt= int(prompt)
if prompt==0:
    baseline()
    remove_stopword()
    filter_wordlength()
    frequency_filtering()
elif prompt==1:
    baseline()
elif prompt==2:
    remove_stopword()
elif prompt==3:
    filter_wordlength()
elif prompt==4:
    frequency_filtering()
print('Thank you for using the program!')
#frequency_filtering()