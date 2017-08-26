# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:26:20 2017

@author: xbblwd3
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:05:31 2017

@author: xbblwd3
"""

import nltk
import pandas as pd
from gensim import models
import sys
from gensim.models import Phrases
import traceback




pathToDirectory = "output/"
dictOfNgramValues = dict({1:"Unigram",2:"Bigram",3:"Trigram"})


def loadCSVFile(pathToDataFile):
    xl = pd.ExcelFile(pathToDataFile)
    df=  xl.parse("Sheet1")
    return df
#df= pd.read_csv(filePath, encoding="ISO-8859-1")



######## writing to text file ###############

def writeVocabToFile(pathToFile,vocab):
        
    thefile = open(pathToFile, 'w')
    
    for item in vocab:
      thefile.write(item+"\n")
    thefile.close()




################## function to generate model according to supplied corpus #####################

def generateModelFromCorpus(corpus):
    model = models.Word2Vec(corpus, min_count=1,size=300,workers=4) #300 NN Layers
    return model





#############################  Preparing the corpus to be trained for the model ###############################

#######  my csv had ErrorDetailedDescription data column, so generateCorpusOfTokensFromDataSet can be changed as need be ##########

def generateCorpusOfTokensFromDataSet(df):
    corpusForErrorDetailedDescription = []
    stopwords = nltk.corpus.stopwords.words('english')
    for row in range(1,len(df.ErrorDetailedDescription)):
        try:
            tokens = [word for sent in nltk.sent_tokenize(df.iloc[row,3]) for word in nltk.word_tokenize(sent) if word not in stopwords]
            corpusForErrorDetailedDescription.append(tokens)
            
        except Exception:
            print("Ignored empty text")
    return corpusForErrorDetailedDescription    











############################# for preparing commonly used phrases from the corpus ##################################



def GenerateNGramCorpus(corpusForErrorDetailedDescription, nGram):
    if(nGram>1):
        nGram= nGram-1
        corpus = GenerateNGramCorpus(corpusForErrorDetailedDescription,nGram)
        
        nGramPhrases = Phrases(corpus, min_count=1, threshold=2)
        return list(nGramPhrases[corpus])
    
    else:
        return corpusForErrorDetailedDescription




#finalCorpus = []
#
##   Model for singular tokens in ORE Dataset 
#modelForUnigramTokensofORE = generateModelFromCorpus(corpusForErrorDetailedDescription)
##bigram = Phrases(sentence_stream, min_count=1, threshold=4)
#
##   Model for bigram phrases tokens in ORE Dataset, eg: "payment_failure"
#bigram = Phrases(corpusForErrorDetailedDescription, min_count=1, threshold=2)
#modelForBigramPhrasesofORE = generateModelFromCorpus(bigram[corpusForErrorDetailedDescription])
#
##   Model for trigram tokens in ORE Dataset
#trigram = Phrases(bigram[corpusForErrorDetailedDescription], min_count=1, threshold=2)
#modelForTrigamPhrasesofORE = generateModelFromCorpus(trigram[bigram[corpusForErrorDetailedDescription]])
#
#writeVocabToFile(pathToDirectory+"TrigamPhrasesofORE.txt",trigram.vocab)
#finalCorpus.extend(corpusForErrorDetailedDescription)
#finalCorpus.extend(trigram[bigram[corpusForErrorDetailedDescription]])
#writeVocabToFile(pathToDirectory+"finalCorpus.txt",finalCorpus)





###############  bucketise similar phrases ###################

def generateSemanticRelatedPhrases(model):
    listOfPhrases = list(model.wv.vocab)
    from collections import defaultdict
    dictOfSimilarPhrases = defaultdict(list)
    for singlePhrase in listOfPhrases:
        try:
            listOfSimilarPhrases = [x for x,_ in list(model.most_similar(singlePhrase))]
            dictOfSimilarPhrases[singlePhrase].append(listOfSimilarPhrases)
        except Exception:
            print ("Ignored Key due to non match:\t" , singlePhrase)
    return dictOfSimilarPhrases
        
        


################################# writing to excel files ##############################################


def writeDictToXLSXFormat(dictForNGrams,nameOfDict):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(pathToDirectory+nameOfDict+".xlsx")
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    for key in dictForNGrams.keys():
        row += 1
        worksheet.write(row, col, key)
        for item in dictForNGrams[key]:
            worksheet.write(row, col + 1, str(item))
            row += 1
    
    workbook.close()











if __name__ == '__main__':

    try:
        
        pathToDataFile = sys.argv[1]
        nGram = int(sys.argv[2])
        df = loadCSVFile(pathToDataFile)
        corpusForErrorDetailedDescription = generateCorpusOfTokensFromDataSet(df)
  
            
        finalCorpus =GenerateNGramCorpus(corpusForErrorDetailedDescription, nGram)
        writeVocabToFile(pathToDirectory+(dictOfNgramValues.get(nGram) if nGram in dictOfNgramValues.keys() else str(nGram)+"Gram")+".txt")        
        modelForNgamPhrases = generateModelFromCorpus(finalCorpus)
        
        
        try:
    
            dictOfSimilarPhrasesForNgrams = generateSemanticRelatedPhrases(modelForNgamPhrases)
            writeDictToXLSXFormat(dictOfSimilarPhrasesForNgrams,dictOfNgramValues.get(nGram))
       
        except Exception:
            print("Error due to:\n",traceback.print_exc())
          
    except Exception:
        print("Usage : python NgramModel.py <full path to file> <integer value of n-gram>")