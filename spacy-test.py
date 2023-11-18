import spacy

nlp = spacy.load("en_core_web_sm")

with open ("test-doc.txt", encoding='utf-8') as f:
    doc = nlp(f.read())
    for token in doc:
        print(token.text, token.pos_, token.dep_)