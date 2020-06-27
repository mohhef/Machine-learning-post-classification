# Post type classification

This is a python script that classifys a testing set to a post type (ask_hn, show_hn, story, poll). It has over 80% accuracy for a test set of 5000 post types.

Steps to run the program:
-- To change the csv file name, change the first two variables in const.py
-- To disable a predictor that increases performance, change 'HEURSTIC' to 'False'
in const.py

1. install all the libraries above
2. navigate to the assignment folder where main.py exists
3. open the command line prompt
4. Type 'py main.py' in the command line
5. Instruction are written in the terminal
6. To run all the experiments, enter 0
7. To run the baseline experiment, enter 1
8. To run the stopword experiment, enter 2
9. To run the word length filter experiment, enter 3
10. To run the infrequency filter experiment, enter 4
11. Output text files exist in 'txtOutput'

libraries used:

pandas
matplotlib
sklearn
nltk
json
string
math
re
