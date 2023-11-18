import time
begin = time.time()

import spacy

nlp = spacy.load("en_core_web_sm")

with open ("1-s2.0-S8756328205000050-main.txt", encoding='utf-8') as f:
    doc = nlp(f.read())
    for token in doc:
        print(token.text, token.pos_, token.dep_)









time.sleep(1)
end = time.time()
print(f"Time is {end - begin}")