from nltk.tokenize import word_tokenize
import sys
import os
from io import open
import whoosh.index as index
from whoosh.reading import IndexReader
from whoosh.fields import *
from whoosh.qparser import QueryParser
from nltk.stem import *
from nltk.stem.porter import *
import urllib2
import timeit
start = timeit.default_timer()
lemmatizer = WordNetLemmatizer()
QUERY_PATH = '../test/'
DATA_PATH='../data/'
INDEX_PATH="../index"
stri=input("Enter your choice : \n1.Raw Input \n2.Stop words \n3.Lemmatised\n");
k=int(stri)
if k == 1:
   INDEX_PATH +="/index_raw/"
if k == 2:
    INDEX_PATH+="/index_stopword/"
if k == 3:
   INDEX_PATH+="/index_lemmatized/"
schema = Schema(title=TEXT(stored=True), content=TEXT)
ix = index.open_dir(INDEX_PATH)
writer = ix.writer()
noOfTerms=0
ixRead=ix.reader()
s=ixRead.all_terms()
for i in s:
  noOfTerms=noOfTerms+1
if k==1:
  print "No of terms in Raw Index is: ",noOfTerms
if k==2:
  print "No of terms in Index after removal of Stopwords is: ",noOfTerms
if k==3:
  print "No of terms in Lemmatized Index after removal of Stopwords is: ",noOfTerms
f=open("../output/output.txt","w",encoding='utf-8')
QUERY_PATH += os.listdir(QUERY_PATH)[0]
queries=open(QUERY_PATH,"r").read().split("\n") #getting queries from query.txt
if k == 3:
    lemmatized=[]
    for query in queries:
	words=query.split(" ")
	lems=[]
	for word in words:        
		lems.append(lemmatizer.lemmatize(word.lower()))
	lemmatized.append(' '.join(lems))
    counter = 0
    for query in lemmatized:
        if query!="":
            output=queries[counter]+": "
        counter += 1
        qp = QueryParser("content", schema=ix.schema)
        q = qp.parse(unicode(query))
        with ix.searcher() as s:
            results = s.search(q,limit=None)
            freq=results.scored_length()
            if freq>0:
              output=output+str(freq)+" ["
              for i in range(freq):
               if i!=freq-1:
                 output=output+ urllib2.unquote(results[i]['title'])+", "
               else:
                 output=output+urllib2.unquote(results[i]['title'])+"]\n"
            elif freq==0:
              output=output+str(freq)+" [ ]\n"
        f.write(output)
    f.close()
else:
    for query in queries:
      if query!="":
        output=query+": "
        qp = QueryParser("content", schema=ix.schema)
        q = qp.parse(unicode(query))
        with ix.searcher() as s:
            results = s.search(q,limit=None)
            freq=results.scored_length()
            if freq>0:
              output=output+str(freq)+" ["
              for i in range(freq):
               if i!=freq-1:
                 output=output+ urllib2.unquote(results[i]['title'])+", "
               else:
                 output=output+urllib2.unquote(results[i]['title'])+"]\n"
            elif freq==0:
              output=output+str(freq)+" [ ]\n"
        f.write(output)
    f.close()
    stop = timeit.default_timer()
    #print stop - start
