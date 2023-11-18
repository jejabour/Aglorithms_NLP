import time
begin = time.time()

import os
import spacy
from collections import Counter


nlp = spacy.load("en_core_web_sm")


# Dictionary to contain files and their frequency of a specified word counts
freq_dict = {}

word_freq = Counter()

# Variable to contain a new line for use when writing to a file because Python is weird like that
new_line = '\n'

# Path to my small subset of txt's to test
test_path = 'test-txt-search/'

# Path to where I want to store all the txt outputs containing finalized parts of speech
pos_output_path = "./Output-spacy/"

freq_output_path = "spaCy_Frequency_of_coat.txt"




# Make the output path if it doesn't exist
try:
    os.mkdir(pos_output_path)
except OSError as error:
    print(error)

# Replace my NLTK Freq output if it already exists
if os.path.exists(freq_output_path):
    os.remove(freq_output_path)

# This is just a file I know contains a few instances of coating
# 1-s2.0-S8756328205000050-main.txt

# Temporarily, file is my short test file
file = "test-doc.txt"

with open (file, encoding='utf-8') as f:
    # Turns the doc into a "spacy.tokens.doc.Doc"
    doc = nlp(f.read())
    
    # Put the "Lemmatized" words into for later looking for coat
    # In spaCy, Lemma takes the base form of a word
    # I'm gonna keep calling it stemming because it's easier
    stemmed_list = []

    pos_output_file = open(pos_output_path + file)

    for token in doc:
        if not token.is_stop and not token.is_punct:
            
            pos_output = f"{token.pos_} {new_line}"
            pos_output_file.write(pos_output)
            stemmed_list.append(token.lemma_.lower())
            # print(token.text, token.pos_, token.lemma_)
            # print(token.text)
    
    pos_output_file.close()
    word_freq = Counter.clear()
    word_freq = Counter(stemmed_list)

    freq_dict.update({file: word_freq['coat']})











time.sleep(1)
end = time.time()
print(f"Time is {end - begin}")