import string
import math
from io import open
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import operator
DATA_PATH = '../data/'
INPUT_DOC_FILENAME = 'cran.all.1400'
doc = open(DATA_PATH + INPUT_DOC_FILENAME, "r", encoding='utf-8').read()
wordnet_lemmatizer = WordNetLemmatizer()
doclist = doc.split(".I ")[1:]
dictdoclist = []
dataList = []
for doc in doclist:
    w = doc.find(".W\n")
    document = {}
    dataList.append(doc[w + 3:])
    dictdoclist.append(document)
dataList = [i.replace("\n", " ") for i in dataList]
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()  # Initializing Instance of Stemmer
# -----------------------------------Producing Filtered Lines-------------------------------------------#
filteredLines = []
for i in range(len(dataList)):
    filtered_sentence = []
    filtered_sentence[:] = []
    word_tokens = word_tokenize(dataList[i])
    # Converting to LowerCase
    word_lower_tokens = []
    word_lower_tokens = [x.lower() for x in word_tokens]
    # Removing Stop Words
    filtered_sentence = [w for w in word_lower_tokens if not w in stop_words]
    # Removing Punctuations
    filtered_sentence = [''.join(c for c in s if c not in string.punctuation) for s in filtered_sentence]
    # Removing Empty String
    filtered_sentence = [s for s in filtered_sentence if s]
    # Stemming
    filtered_stemmed_sentence = []
    filtered_stemmed_sentence[:] = []
    for j in filtered_sentence:
        filtered_stemmed_word = wordnet_lemmatizer.lemmatize(j)
        filtered_stemmed_sentence.append(filtered_stemmed_word)
    filteredLines.append(filtered_stemmed_sentence)  # Appending Filtered sentence to List.

N = len(filteredLines)

invertedIndex = {}
traveresedToken = []
for index in range(len(filteredLines)):
    for word in filteredLines[index]:
        if word not in traveresedToken:
            docIDTfdict = {}
            traveresedToken.append(word)
            docIDTfdict = defaultdict(lambda: 0, docIDTfdict)
            invertedIndex[word] = docIDTfdict
            invertedIndex[word][index + 1] = 0
        invertedIndex[word][index + 1] += 1
docDict = {}
#for language model
mc = 0
docDict = defaultdict(lambda: {}, docDict)
for index in range(len(filteredLines)):
    traveresedToken = []
    docNV = 0
    md = 0
    docDict[index + 1] = {}
    for word in filteredLines[index]:
        mc = mc + 1
        md = md + 1
        if word not in traveresedToken:
            traveresedToken.append(word)
            tf = invertedIndex[word][index + 1]
            idf = len(invertedIndex[word])
            docDict[index + 1][word] = tf * idf
            docNV = docNV + (tf * idf) ** 2
    docDict[index + 1][u"**MD**"] = md
    docDict[index + 1][u"**NV**"] = docNV ** 0.5
#                                                                              #
# ---------------------------------Query Analysis ----------------------------- #
#																			   #

DATA_PATH = '../data/'
INDEX_PATH = '../index/'
RESULT_PATH="../results/"
INPUT_QRY_DOC_FILENAME = 'cran.qry'
file = open(DATA_PATH + INPUT_QRY_DOC_FILENAME, "r", encoding='utf-8').read()
relevance_scores_strings= open(DATA_PATH+"cranqrel", "r", encoding="UTF-8").read().split("\n")
output=open(RESULT_PATH+"output.txt","w")
output.write(unicode("Query Number | TFIDF NDCG SCORE | Language Model NDCG SCORE\n"))

file = file.replace("\n", "")
unprocessed_queries = file.split(".I")[1:]
processed_queries = []
for query in unprocessed_queries:
    temp = query.split(".W")
    processed_queries.append(temp[1])
queryNo=1
relevance_scores_counter = 1
for inputQryPair in processed_queries:
    x = 0
    x = relevance_scores_strings[relevance_scores_counter].split(" ")[:3]
    relevance_scores_counter += 1
    scores = {}
    while int(x[0]) == queryNo:
        scores[int(x[1])] = 5 - abs(int(x[2]))
        if relevance_scores_counter == len(relevance_scores_strings)-1:
            break
        y=relevance_scores_strings[relevance_scores_counter].split(" ")[:2]
        if (int(y[0]) == queryNo):
            x = relevance_scores_strings[relevance_scores_counter].split(" ")[:3]
            relevance_scores_counter += 1
        else:
            break
    IdealDocuments = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:10]
    IDCG = 0
    i = 1
    for item in IdealDocuments:
        IDCG += (2 ** scores[item[0]]) - 1 / float(math.log(i + 1, 2))
        i += 1
    #--------------------------------------------
    query = inputQryPair
    filteredQuery = []
    queryTerms = query.split(" ")

    filtered_sentence = []
    filtered_sentence[:] = []
    word_tokens = word_tokenize(query)
    # Converting to LowerCase
    word_lower_tokens = []
    word_lower_tokens = [x.lower() for x in word_tokens]
    # Removing Stop Words
    filtered_sentence = [w for w in word_lower_tokens if not w in stop_words]
    # Removing Punctuations
    filtered_sentence = [''.join(c for c in s if c not in string.punctuation) for s in filtered_sentence]
    # Removing Empty String
    filtered_sentence = [s for s in filtered_sentence if s]
    # Stemming
    filtered_stemmed_sentence = []
    filtered_stemmed_sentence[:] = []
    for j in filtered_sentence:
            filtered_stemmed_word = wordnet_lemmatizer.lemmatize(j)
            # filtered_stemmed_sentence.append(filtered_stemmed_word)
            if invertedIndex.has_key(filtered_stemmed_word):
                filteredQuery.append(filtered_stemmed_word)
    # filteredQuery.append(filtered_stemmed_sentence)		#Appending Filtered sentence to List.

    traveresedQryToken = []
    queryDict = {}
    for q in filteredQuery:
        if q not in traveresedQryToken:
            traveresedQryToken.append(q)
            queryDict[q] = []
            queryDict[q].append(1)
            df = len(invertedIndex[q])
            idf = math.log10(N / float(df))
            queryDict[q].append(idf)
            continue
        queryDict[q][0] = queryDict[q][0] + 1
    queryDictList = queryDict.items()
    normalizedQryTfidf = 0
    queryDict = defaultdict(lambda: 0, queryDict)
    for item in queryDictList:
        tf = 1 + math.log10(item[1][0])
        tfidf = tf * item[1][1]
        normalizedQryTfidf += tfidf ** 2
        queryDict[item[0]] = tfidf
    normalizedQryTfidf = normalizedQryTfidf ** 0.5
    queryDictList = queryDict.items()
    queryDict = defaultdict(lambda: 0, queryDict)
    for item in queryDictList:
        queryDict[item[0]] = item[1] / normalizedQryTfidf
    ##-------------For Tf-idf Scoring and computing total Freq of terms -----------##
    docIDSet = set([])
    queryDictList = queryDict.items()
    scoreConvDict = {}
    scoreConvDict = defaultdict(lambda: 0, scoreConvDict)
    termFreqAllDocs = {}
    termFreqAllDocs = defaultdict(lambda: 0, termFreqAllDocs)
    for item in queryDictList:
        term = item[0]
        qrytfidf = item[1]
        totalfreq = 0
        postingList = invertedIndex[term].items()
        for idFreqPair in postingList:
            docId = idFreqPair[0]
            docIDSet.add(docId)
            # For Conventional Tf-IDF score
            doctfidf = docDict[docId][term] / docDict[docId][u"**NV**"]
            totaltfidf = doctfidf * qrytfidf
            scoreConvDict[docId] = scoreConvDict[docId] + totaltfidf
            # For Language Model Score
            totalfreq = totalfreq + idFreqPair[1]
        termFreqAllDocs[term] = totalfreq
          ##-------------------For Language Model Scoring ------------------------##
    scoreLMDict = {}
    scoreLMDict = defaultdict(lambda: 1, scoreLMDict)
    for docid in docIDSet:
        lmScore = 1
        for item in queryDictList:
            term = item[0]
            if docDict[docid].has_key(term):
                lmScore = 0.5 * invertedIndex[term][docid] / float(docDict[docid][u"**MD**"]) + 0.5 * termFreqAllDocs[
                    term] / float(mc)
            else:
                lmScore = 0.5 * termFreqAllDocs[term] / float(mc)
            scoreLMDict[docid] = scoreLMDict[docid] * lmScore
    TFIDF_scores =sorted(scoreConvDict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    LM_scores=sorted(scoreLMDict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    #incrementing queryNo
    TFIDF_DCG = 0
    i = 1
    TFr=0
    for item in TFIDF_scores:
        if scores.has_key(item[0]):
            TFr+=1
            TFIDF_DCG += (2 ** scores[item[0]]) - 1 / float(math.log(i + 1, 2))
        i += 1
    TFIDF_NDCG=TFIDF_DCG/float(IDCG)
    LM_DCG = 0
    i = 1
    LMr=0
    for item in LM_scores:
        if scores.has_key(item[0]):
            LMr+=1
            LM_DCG += (2 ** scores[item[0]]) - 1 / float(math.log(i + 1, 2))
        i += 1
    LM_NDCG=LM_DCG/float(IDCG)
    output.write(unicode(str(queryNo)+" |"+str(TFIDF_NDCG)+" |"+str(LM_NDCG)+str(LMr)+"\n"))
    queryNo += 1