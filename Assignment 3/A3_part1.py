from io import open
from collections import defaultdict
import os

#+++++++++++++++++++++++Function for Dot-product of two input vectors+++++++++++++++++++++++++++++++++#
def dotProduct(a,b):
    idf=0
    list1=a.items()
    for item in list1:
        idf+=a[item[0]]*b[item[0]]
    return idf;

#+++++++++++++++++++++++Defining constants+++++++++++++++++++++++++++++++++#
DATA_PATH = '../data/'
IDFS_FILENAME='idfs.txt'
SIGNAL_TRAIN_DOC_FILENAME = 'pa3.signal.train'
SIGNAL_DEV_DOC_FILENAME = 'pa3.signal.dev'
REL_TRAIN_DOC_FILENAME = 'pa3.rel.train'
REL_DEV_DOC_FILENAME = 'pa3.rel.dev'

#+++++++++++++++++++++++Initializing variables+++++++++++++++++++++++++++++++++#
idf_score={}
idf_score = defaultdict(lambda: 0, idf_score)
query_url_rel_dict={}
queryDocumentFeatureVector=[]
relevanceScoreVector=[]

#+++++++++++++++++++++++Parsing IDFS file and making IDFS dictionary+++++++++++++++++++++++++++++++++#
idfsDoc = open(DATA_PATH + IDFS_FILENAME, "r", encoding='utf-8').readlines()
for line in idfsDoc:
    temp = line.split(":")
    idf_score[temp[0]] = float(temp[1][:-2])


#+++++++++++++++++++++++Parsing Relevance Training file and making final relevance score vector+++++++++++++++++++++++++++++++++#
relTrainDoc = open(DATA_PATH + REL_TRAIN_DOC_FILENAME, "r", encoding='utf-8').read()
#Splitting pa3.rel.train for each query
relTrainDoc=relTrainDoc.split("query: ")
for i in range(len(relTrainDoc)):
    #Excluding Empty string @ 0index
    if i != 0:
        qryUrlList = relTrainDoc[i].split("\n  url: ")
        #Extracting query from 0th index
        query = qryUrlList[0]
        query_url_rel_dict[query]=[]

        #For each URLDATA corresponding to each query
        for j in range(len(qryUrlList)):
            #Excluding last url and first empty string
            if j != 0 and j != len(qryUrlList)-1:
                urlRelList = qryUrlList[j].split(" ")
                tempList=[]
                tempList.append(urlRelList[0])
                tempList.append(urlRelList[1])
                query_url_rel_dict[query].append(tempList)
                #Updating Final relevance score vector
                relevanceScoreVector.append(float(urlRelList[1]))
            #For last URL data for each query.
            elif j == len(qryUrlList)-1:
                urlRelList = qryUrlList[j].split(" ")
                tempList = []
                tempList.append(urlRelList[0])
                tlist=urlRelList[1].split("\n")
                tempList.append(tlist[0])
                query_url_rel_dict[query].append(tempList)
                # Updating Final relevance score vector
                relevanceScoreVector.append(float(tlist[0]))


#+++++++++++++++++++++++Parsing Signal Training File and making Query-Document Feature vector+++++++++++++++++++++++++++++++++#
sigTrainDoc = open(DATA_PATH + SIGNAL_TRAIN_DOC_FILENAME, "r", encoding='utf-8').read()
sigTrainDoc = sigTrainDoc.split("query: ")
for qryData in sigTrainDoc[1:]:
    urlDataList=qryData.split("\n  url: ")
    mainQuery=urlDataList[0]
    queryWordsList=mainQuery.split(" ")
    qryWordIDFdict={}

    #Updating query vector with their corresponding IDF score.
    for queryWord in queryWordsList:
        if not qryWordIDFdict.has_key(queryWord):
            qryWordIDFdict[queryWord]=idf_score[queryWord]

    for urlData in urlDataList[1:]:
        lineList=urlData.split("\n    ")
        mainURL = lineList[0]

        #Initializing all Vectors as dictionary and making default value as 0.
        URLWordTFdict = {}
        URLWordTFdict = defaultdict(lambda: 0, URLWordTFdict)
        titleWordTFdict = {}
        titleWordTFdict = defaultdict(lambda: 0, titleWordTFdict)
        headerWordTFdict = {}
        headerWordTFdict = defaultdict(lambda: 0, headerWordTFdict)
        bodyWordTFdict = {}
        bodyWordTFdict = defaultdict(lambda: 0, bodyWordTFdict)
        anchorWordTFdict = {}
        anchorWordTFdict = defaultdict(lambda: 0, anchorWordTFdict)

        #Updating URL feature vector with Term frequecy of words present in URL.
        for queryWord in queryWordsList:
            URLWordTFdict[queryWord] = mainURL.lower().count(queryWord) #yaha 0 par query bhi aa rhi hai

        for line in lineList[1:]:
            keyValue=line.split(": ")

            #Updating title vector with term frequency.
            if keyValue[0] == 'title':
                titleWordsList = keyValue[1].split(" ")
                for titleWord in titleWordsList:
                    if qryWordIDFdict.has_key(titleWord):
                        titleWordTFdict[titleWord] += 1

            # Updating header vector with term frequency.
            if keyValue[0] == 'header':
                headwordList=keyValue[1].split(" ")
                for headWord in headwordList:
                    if qryWordIDFdict.has_key(headWord):
                        headerWordTFdict[headWord] += 1

            # Updating anchor vector with term frequency.
            if keyValue[0] == 'anchor_text':
                anchorWordList = keyValue[1].split(" ")
                for anchorWord in anchorWordList:
                    if qryWordIDFdict.has_key(anchorWord):
                        anchorWordTFdict[anchorWord] += 1

            # Updating Body vector with term frequency.
            if keyValue[0] == 'body_hits':
                positionalList = keyValue[1].split(" ")
                bodyWordTFdict[positionalList[0]]=len(positionalList)-1

        #Updating final Query Document Feature Vector.
        tempList=[]
        tempList.append(dotProduct(qryWordIDFdict, URLWordTFdict))
        tempList.append(dotProduct(qryWordIDFdict, titleWordTFdict))
        tempList.append(dotProduct(qryWordIDFdict, bodyWordTFdict))
        tempList.append(dotProduct(qryWordIDFdict, headerWordTFdict))
        tempList.append(dotProduct(qryWordIDFdict, anchorWordTFdict))
        queryDocumentFeatureVector.append(tempList)


# -----------------------------------------------#
# Conversion to mean 0 and standard deviation 1--#
# -----------------------------------------------#
mean=[0,0,0,0,0] #to calculate E[X]
square_mean=[0,0,0,0,0]#to calculate E[X^2]
for feature in queryDocumentFeatureVector:
    for i in range(len(feature)):
        mean[i]+=feature[i]
        square_mean[i]+=feature[i]**2
standard_deviation=[0,0,0,0,0]
#Calculating Standard Deviation and Mean using summation of X and X**2 Values
for i in range(5):
    mean[i] /= len(queryDocumentFeatureVector)
    x=mean[i]**2
    standard_deviation[i] = (abs((square_mean[i] / len(queryDocumentFeatureVector) - (x)))) ** .5
    standard_deviation[i]=round(standard_deviation[i],9)
    mean[i]=round(mean[i],9)
#Using the old mean and standard Deviation to convert the data set into Mean 0 and 1 Standard Deviation
for j in range(len(queryDocumentFeatureVector)):
    for i in range(5):
        queryDocumentFeatureVector[j][i]=(queryDocumentFeatureVector[j][i]-mean[i])/standard_deviation[i]
#Creating Output File of Training Data
output=open("Training_data.txt","w",encoding="utf-8")
queryDocumentFeatureVectorOutput=[]
for element in queryDocumentFeatureVector:
    x = ""
    for value in element:
        x = x + str(value)
        x = x + " "
    x += "\n"
    queryDocumentFeatureVectorOutput.append(x)
outputRelevance=""
for value in relevanceScoreVector:
    outputRelevance = outputRelevance + str(value)
    outputRelevance += "\n"
print relevanceScoreVector
output.write(unicode(" ".join(queryDocumentFeatureVectorOutput)))
output.write(u"\n")
output.write(unicode(outputRelevance))