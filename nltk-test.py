# credit of code goes to
# https://dev.to/thedevtimeline/compare-documents-similarity-using-python-nlp-4odp
# https://stackoverflow.com/questions/30829382/check-the-similarity-between-two-words-with-nltk-with-python
# https://www.nltk.org/howto/wordnet.html
# https://www.youtube.com/watch?v=X2vAabgKiuM

import time
begin = time.time()

from nltk.corpus import wordnet
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os
from nltk.tokenize import word_tokenize
import re
from nltk.probability import FreqDist

from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')


stop_words = set(stopwords.words('english'))



fdist = FreqDist()
pst = PorterStemmer()
punctuation = re.compile(r'[-.?!,:;()|0-9]')
path = 'test-txt-search/'

if os.path.exists("output-nltk.txt"):
    os.remove("output-nltk.txt")

# 1-s2.0-S8756328205000050-main.txt

for file in os.listdir(path):
    with open(path + file, encoding='utf-8') as f:
        text = f.read()
        text_tokens = word_tokenize(text)
        stemmed_list = []
        for word in text_tokens:
            if not word in stop_words:
                stemmed_word = pst.stem(word)
                # stemmed_list.append(stemmed_word)
                no_punct = punctuation.sub("", stemmed_word)
                if len(no_punct) > 0:
                    stemmed_list.append(no_punct)
        fdist.clear()
        for word in stemmed_list:
            fdist[word] += 1
        
        f = open("output-nltk.txt", "a")
        new_line = '\n'
        output = f"Instance of 'coat' in {file}: {fdist['coat']} {new_line}"
        f.write(output)
        f.close()






time.sleep(1)
end = time.time()

print(f"Time is {end - begin}")

    



# for file in tqdm(os.listdir(path)):
#     with open(path + file, encoding='utf-8') as f:
#         text = f.read()
#         text_tokens = word_tokenize(text)

#         stemmed_list = []
#         for word in text:
#             stemmed_word = pst.stem(word)
#             stemmed_list.append(stemmed_word)
#         print(stemmed_list)

        











# path = 'test-txt-search/'
# token_dict = {}
# stemmer = PorterStemmer()

# def stem_tokens(tokens, stemmer):
#     stemmed = []
#     for item in tokens:
#         stemmed.append(stemmer.stem(item))
#     return stemmed

# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     stems = stem_tokens(tokens, stemmer)
#     return stems

# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         print(file)
#         file_path = subdir + os.path.sep + file
#         shakes = open(file_path, 'r')
#         text = shakes.read()
#         lowers = text.lower()
#         translating = str.maketrans('', '', string.punctuation)
#         no_punctuation = lowers.translate(translating)
#         token_dict[file] = no_punctuation
        
# #this can take some time
# tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
# tfs = tfidf.fit_transform(token_dict.values())

# print(tfidf)














# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

################################
## token = a word
## document = a sentence or paragraph
## corpus = a collection of 'documents' as a bag of words
################################


# file_docs creates an array
# tokens breaks each sentence into a token
# file_docs becomes an array of every token
# gen_docs creates a token out of each word, and becomes an array in which each sentence is an array inside that array, where the inner array contains words separated as each element
# dictionary creates a dictionary of every word, where the key becomes a unique id for each individual word
# corpus contains the word id and it's frequency. Only per sentence


# list1 = ['Coating', 'coating']


# path = "./test-txt-search/"

# file with coating
# test-txt-search/1-s2.0-S8756328205000050-main.txt

# first file
#  "test-txt-search/1-s2.0-S875632820800197X-main.txt"

# file_docs = []

# with open ('test-doc.txt', encoding='utf-8') as f:
#     tokens = sent_tokenize(f.read())
#     # for line in tokens:
#     #     file_docs.append(line)


# with open ('test-doc.txt', encoding='utf-8') as f:
#     words = word_tokenize(f.read())

# print("Tokens: ", type(tokens))
# print("Words: ", type(words))
# print(type(tokens))
# print()
# print(type(file_docs))
# print("Number of documents:",len(file_docs))


# I'm making a change so that the corpus contains a word id and frequency in relation to the entire doc instead of per sentence
# I'll keep the original code above the changes
# with open('test-doc.txt', 'r', encoding="utf-8") as f:
#     tokens = word_tokenize(f.read())
#     for line in tokens:
#         file_docs.append(line)
        

# gen_docs = [[w.lower() for w in word_tokenize(text)]
#             for text in tokens]

# gen_docss = [wl.lower() for wl in words]

# print()
# print("gendocs: ", gen_docs)
# print()
# print("gendocss: ", gen_docss)

# gen_docss = [gen_docss]
# print(gen_docs)
# print()


# This allows the change and it's kinda dumb. Dictionary needs it to be an array of arrays
# which would normnally happen when we do sent_tokenize and then word_tokenize
# but I only wanted word_tokenize
# gen_docss = [gen_docss]
# # print(gen_docss)


# dictionary = gensim.corpora.Dictionary(gen_docs)
# dictionary2 = gensim.corpora.Dictionary(gen_docss)
# # print(dictionary)
# print()
# print("Dictionary2")
# print(dictionary2)
# # dictionary = gensim.corpora.Dictionary(gen_docss)


# corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
# corpus2 = [dictionary2.doc2bow(gen_doc2) for gen_doc2 in gen_docss]
# # corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docss]
# print("corpus")
# print(corpus)
# print()
# print("Corpus2")
# print(corpus2)


# print()
# print("tf_idf")
# tf_idf = gensim.models.TfidfModel(corpus)
# for doc in tf_idf[corpus]:
#     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

# print()
# print("tf_idf2")
# tf_idf2 = gensim.models.TfidfModel(corpus2)
# for doc in tf_idf2[corpus2]:
#     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

# sims = gensim.similarities.Similarity('workdir/', tf_idf[corpus], num_features=len(dictionary))

# file2_docs = []

# with open ('test-txt-search/1-s2.0-S8756328205000050-main.txt', 'r', encoding="utf-8") as f:
#     tokens = sent_tokenize(f.read())
#     for line in tokens:
#         file2_docs.append(line)
        
        
# # print("Number of docs: ", len(file2_docs))

# for line in file2_docs:
#     query_doc = [w.lower() for w in word_tokenize(line)]
#     query_doc_bow = dictionary.doc2bow(query_doc)
    

# query_doc_tf_idf = tf_idf[query_doc_bow]
# print('Comparing result: ', sims[query_doc_tf_idf])








# for file in tqdm(os.listdir(path)):
#     with open(path + file, 'r', encoding="utf-8") as f:
#         textfile = f.read()
#         paper = Text(textfile)

        


# list2 = [text]
# list = []

# for word1 in tqdm(list1):
#     for word2 in list2:
#         wordFromList1 = wordnet.synsets(word1)[0]
#         wordFromList2 = wordnet.synsets(word2)[0]
#         s = wordFromList1.wup_similarity(wordFromList2)
#         list.append(s)

# print(max(list)) 


# sims = []