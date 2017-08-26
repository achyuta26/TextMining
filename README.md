# NGramModel

With the use of libraries from gensim, like Phrases and Word2Vec, similar meaning phrases are segregated. 
Phrases library outputs the commonly occurring words together as phrases. It is an iterative process, i.e. on the first run it gives bigram phrases like for tokens 'new' and 'york', the outcome after the first run would be 'new_york'. Hence after another run, it generates trigram phrases. All these ngram word phrases are emitted out as single tokens with words joined by a seperator "_" (can be modified after code tweaking). 

This code gives you the flexibility to plugin your value of N for your Ngram model generation using the recursive function as:


def GenerateNGramCorpus(corpusForErrorDetailedDescription, nGram):
    if(nGram>1):
        nGram= nGram-1
        corpus = GenerateNGramCorpus(corpusForErrorDetailedDescription,nGram)
        
        nGramPhrases = Phrases(corpus, min_count=1, threshold=2)
        return list(nGramPhrases[corpus])
    else:
        return corpusForErrorDetailedDescription

After the generation of the tokens, the Word2Vec model is trained using the phrasal tokens obtained above and all of the tokens are generated to vectors on the dimension specified in function's parameter. The dimension I took here is 300. Thus, all the tokens get converted to 300 dimension vector layers.
Using the most_similar(phrase) method, we are able to find out the closest vector or to say the phrase lying in the nearest proximity of the queried phrase, along with the probabilities of its proximity.

Code modifications to be done by user:

->My CSV had ErrorDetailedDescription data column, so the function generateCorpusOfTokensFromDataSet() can be changed as need be.

->df.ErrorDetailedDescription needs to be changed for whatever column needs to be used for NGramModel generation.

