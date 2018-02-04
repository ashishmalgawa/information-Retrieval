from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
import os
from io import open
import whoosh.index as index
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.index import create_in
from nltk.stem import *
from nltk.stem.porter import *
#paths to other folders
DATA_PATH = '../data/'
INDEX_PATH= '../index/index_lemmatized'
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
schema = Schema(title=TEXT(stored=True), content=TEXT)
if not os.path.exists(INDEX_PATH):
    os.mkdir(INDEX_PATH)
ix = create_in(INDEX_PATH, schema)
writer = ix.writer()
i = 0
for filename in os.listdir(DATA_PATH):
    print i, "traversing ",filename
    i  += 1
    doc=open(DATA_PATH+filename,"r",encoding='utf-8').read() #reading docs one by one
    ufn=unicode(filename)
    if len(doc)>0:
        tokenised_doc=word_tokenize(doc)
        filtered_doc=[]
        for word in tokenised_doc:
            if word.lower() not in STOP_WORDS:
                filtered_doc.append(lemmatizer.lemmatize(word))
        stop_removed=' '.join(filtered_doc)
        writer.add_document(title=ufn,content=stop_removed)
writer.commit()
