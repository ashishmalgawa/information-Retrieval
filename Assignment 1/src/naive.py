from nltk.tokenize import word_tokenize
import string
import sys
import os
from io import open
query_path = '../test/'
data_path='../data/'
for filename in os.listdir(query_path):
    query_path += filename
queries=open(query_path,"r").read().split("\n") #getting queries from query.txt
results={}
i = 0
for query in queries:
    results[query]=""
for filename in os.listdir(data_path):
    print i, "traversing ",filename
    i  += 1
    doc=word_tokenize(open(data_path+filename,"r",encoding='utf-8').read().lower()) #reading docs one by one
    clean_words=[''.join(c for c in s if c not in string.punctuation) for s in doc]
    list=[]
    for word in doc:
        k = word.split("||")
        for w in k:
            list.append(w)
    clean_words=[''.join(c for c in s if c not in string.punctuation) for s in list]
    for query in queries:
        tokenized_query=word_tokenize(query.lower())
        count=0
        for word in tokenized_query:
            if word in clean_words:
                count=count+1;
        if count==len(tokenized_query):
            if results[query]=="":
                results[query] += filename
            else:
                s=", "+str(filename)
                results[query] += s
f=open("../output/output_naive.txt","w",encoding='utf-8')
for i in results:
    freq=len(results[i].split(","))
    op=i+": "+str(freq)+" ["+results[i]+"]\n"
    f.write(op)
f.close()