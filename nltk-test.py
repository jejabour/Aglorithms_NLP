# Joseph Jabour
# CS8833 Algorithms
# Prof. Ioana Banicescu

# references
# https://dev.to/thedevtimeline/compare-documents-similarity-using-python-nlp-4odp
# https://stackoverflow.com/questions/30829382/check-the-similarity-between-two-words-with-nltk-with-python
# https://www.nltk.org/howto/wordnet.html
# https://www.youtube.com/watch?v=X2vAabgKiuM
# https://www.nltk.org/howto/tokenize.html
# https://courses.cs.duke.edu/spring14/compsci290/assignments/lab02.html
# https://stackoverflow.com/questions/64248850/sort-simmilarity-matrix-according-to-plot-colors
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/#
# https://www.nltk.org/api/nltk.probability.FreqDist.html
# https://www.geeksforgeeks.org/python-measure-time-taken-by-program-to-execute/
# https://blog.devgenius.io/time-complexity-with-examples-from-nlp-60c4feb9f31e




import time
begin = time.time()

from nltk.corpus import wordnet
import nltk
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os
from nltk.tokenize import word_tokenize
import re
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')

# Grabs the nltk stop words corpora
stop_words = set(stopwords.words('english'))

# Create a frequency distribution
fdist = FreqDist()

# Create a Stemmer   
pst = PorterStemmer()

# Create a list of punctuation to exclude
punctuation = re.compile(r'[-.?!,:;()*+@–|0-9]')

# Dictionary to contain files and their frequency of a specified word counts
freq_dict = {}

# Variable to contain a new line for use when writing to a file because Python is weird like that
new_line = '\n'

# Path to my small subset of txt's to test
test_path = 'test-txt-search/'

# Path to where I want to store all the txt outputs containing finalized parts of speech
pos_output_path = "./pos_output_nltk/"

# Make the output path if it doesn't exist
try:
    os.mkdir(pos_output_path)
except OSError as error:
    print(error)

# Replace my NLTK Freq output if it already exists
if os.path.exists("./NLTK_Frequency_of_coat.txt"):
    os.remove("./NLTK_Frequency_of_coat.txt")

# This is just a file I know contains a few instances of coating
# 1-s2.0-S8756328205000050-main.txt

# iterate over my directory of files
for file in os.listdir(test_path):
    
    # Open the file, set encoder because these journals use big words
    with open(test_path + file, encoding='utf-8') as f:
        # Read the text
        text = f.read()
        # Tokenize every individual word by space
        text_tokens = word_tokenize(text)
        # List for containing all the words that are stemmed
        stemmed_list = []
        # Create and open a new file in the output path to write to
        pos_output_file = open(pos_output_path + file, "a")
        
        # Iterate over every word in the tokenized list
        for word in text_tokens:
            
            # Only if it's not included in the stop words
            if not word in stop_words:
                # Remove most punctuations
                no_punct = punctuation.sub("", word)
                # Create a string of parts of speech out of the words with punctuation remove, then new line
                pos_output = f"{nltk.pos_tag([no_punct])} {new_line}"
                # Write the string to the file
                pos_output_file.write(pos_output)
                # If no_punct basically exists, stem the word and add it to our list
                if len(no_punct) > 0:
                    stemmed_list.append(pst.stem(no_punct.lower()))
        # close the pos_output_file
        pos_output_file.close()
        
        # Clear the freq distribution for a new file, then add every word in stemmed list to it
        fdist.clear()
        for word in stemmed_list:
            fdist[word] += 1
        
        # Search freq_dist for the word 'coat'
        freq_dict.update({file: fdist['coat'] })
    # close the open file.
    f.close()

# replace/open our frequence output
with open("NLTK_Frequency_of_coat.txt", "a") as f:
    # create a dictionary out of a sorted string by highest key value, aka most instances of 'coat'
    sorted_dict = dict(sorted(freq_dict.items(), key = lambda x:x[1], reverse=True))
    # I'm writing this way to include a new line in the txt per dictionary element
    for k in sorted_dict.keys():
        f.write("'{}':'{}'\n".format(k, sorted_dict[k]))
    # Close the file
    f.close()


# At the very top of the python file, I started a clock. Here is where it ends and prints to the console.
time.sleep(1)
end = time.time()
print(f"Time is {end - begin}")
