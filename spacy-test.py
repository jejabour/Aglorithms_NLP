# Joseph Jabour
# CS8833 Algorithms
# Prof. Ioana Banicescu


# refereces
# https://spacy.io/usage/spacy-101#features
# https://blog.ekbana.com/nlp-for-beninners-using-spacy-6161cf48a229
# https://www.digitalocean.com/community/tutorials/python-counter-python-collections-counter


#####################################################################
#####################################################################
#####################################################################


import time
begin = time.time()

import os
import spacy
from collections import Counter
import pandas as pd
import csv

# python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")


# Dictionary to contain files and their frequency of a specified word counts
freq_dict = {}

word_freq = Counter()

# Variable to contain a new line for use when writing to a file because Python is weird like that
new_line = '\n'

# Path to my small subset of txt's to test
working_path = 'journals/100_txts/'

# Path to where I want to store all the txt outputs containing finalized parts of speech
pos_output_path = "spaCy_outputs/pos_output_spacy/"

freq_output_path = "spaCy_outputs/spaCy_Frequency_of_coat.csv"

time_output = "spaCy_outputs/spaCy_times.csv"


# Make the output path if it doesn't exist
try:
    os.mkdir(pos_output_path)
except OSError as error:
    print(error)

# Replace my NLTK Freq output if it already exists
if os.path.exists(freq_output_path):
    os.remove(freq_output_path)

if os.path.exists(time_output):
    os.remove(time_output)

# This is just a file I know contains a few instances of coating
# 1-s2.0-S8756328205000050-main.txt

for i in range(50):
    loop_begin = time.time()
    for file in os.listdir(working_path):

        # if os.path.exists(pos_output_path + file):
        #     # print("file is already there")
        #     pass
        # else:
        with open (working_path + file, encoding='utf-8') as f:
            # Turns the doc into a "spacy.tokens.doc.Doc"
            doc = nlp(f.read())
            
            # Put the "Lemmatized" words into for later looking for coat
            # In spaCy, Lemma takes the base form of a word
            # I'm gonna keep calling it stemming because it's easier
            stemmed_list = []

            pos_output_file = open(pos_output_path + file, "w")

            for token in doc:
                if not token.is_stop and not token.is_punct:
                    
                    pos_output = f"{token}, {token.pos_} {new_line}"
                    pos_output_file.write(pos_output)
                    stemmed_list.append(token.lemma_.lower())
                    # print(token.text, token.pos_, token.lemma_)
                    # print(token.text)
            
            pos_output_file.close()
            word_freq = word_freq.clear()
            word_freq = Counter(stemmed_list)
            

            freq_dict.update({file: word_freq['coat']})

            f.close()
    time.sleep(1)
    loop_end = time.time()
    loop_time = (f"{loop_end - loop_begin} {new_line}")
    print(loop_time)
    with open(time_output, 'a', newline = '') as csvfile:
        csvfile.write(loop_time)


sorted_dict = dict(sorted(freq_dict.items(), key = lambda x:x[1], reverse=True))
pd.DataFrame.from_dict(data=sorted_dict, orient='index').to_csv(freq_output_path, header=False)


time.sleep(1)
end = time.time()
print(f"Time is {end - begin}")