from nltk.tokenize import word_tokenize
import sys
import os
from io import open
import whoosh.index as index
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.index import create_in
from whoosh import analysis
#paths to other folders
DATA_PATH = '../data/'
INDEX_PATH= '../index/index_raw'
schema = Schema(title=TEXT(stored=True, analyzer=analysis.StandardAnalyzer(stoplist=None)), content=TEXT( analyzer=analysis.StandardAnalyzer(stoplist=None)))
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
    writer.add_document(title=ufn,content=doc)
writer.commit()