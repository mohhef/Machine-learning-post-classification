# Post type classification

This is a python script that classifys a dataset to a  following post type post(ask_hn, show_hn, story, poll). It has over 80% accuracy for a test set of 5000 post types.
The classification has been tested on Hacker News dataset fetched form kaggle.

**Information about the dataset:**
- Hacker News posts from 2018 to 2019 
- Each post includes the following columns:
 Object ID | Title | Post Type | Author | Created At | URL | Points | Number of  Comments | year

**Classifier specifications:**
- Builds a probabilistic model from the training set using Naïve Bays Classifier
- Data extrated from "Created At"column of value 2019 is used as a testing dataset.
- Posts are tokenized and the resulting word set is used as vocabulary.
- Each word in the vocablary set  its frequency and its conditional probability are calculated and a smoothing of value 0.5 is used.

**Classifier experiments:**
- Baseline:  
`Access the data and calculates the score of story, ask-hn, show-hn, poll.`
`Select the correct post kind based on the scores`
`Generate a label to indicated if the accessment is correct`
`Student's Guide poll 0.002 0.03 0.007 0.12 story wrong`
- Stop-word Filtering:  
`Remove specific  words from the vocabulary which are accessible in stopwords.txt`
- Word Length Filtering:  
`remove all words with length ≤2 and all words with length ≥ 9`
- Infrequent Word Filtering:  
`Use the baseline experiment, and gradually remove from the vocabulary words with frequency= 1, frequency ≤ 5, frequency ≤ 10, frequency ≤ 15 and frequency ≤ 20. Then gradually remove the top 5% most frequent words, the 10% most frequent words, 15%, 20% and 25% most frequent words. Plot both performance of the classifiers against the number of words left in your vocabulary`

**Steps to run the program:**

-- To change the csv file name, change the first two variables in const.py
-- To disable a predictor that increases performance, change 'HEURSTIC' to 'False'
in const.py

1. install all the libraries
2. navigate to the folder where main.py exists
3. open the command line prompt
4. Type 'py main.py' in the command line
5. Instruction are written in the terminal
6. To run all the experiments, enter 0
7. To run the baseline experiment, enter 1
8. To run the stopword experiment, enter 2
9. To run the word length filter experiment, enter 3
10. To run the infrequency filter experiment, enter 4
11. Output text files exist in 'txtOutput'

**libraries used:**
pandas
matplotlib
sklearn
nltk
json
string
math
re
