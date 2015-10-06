def word_models_obj():

    import nltk
    import string
    import math

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_nonamb_8963.txt","r")
    file_stopwords=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/stopwords.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/labels_obj_nonamb_8963.txt","r")
    file_lexicon=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/dict_mpqa_stem.txt","r")

    len_obj=4543
##    len_subj=5261
    len_subj=4420
    count_min=1    ## the 2 following are thresholds for pruning of word polarity models
    score_min=0.55
    
    
    list_tweets=file_tweets.read().splitlines()
    list_stopwords=file_stopwords.read().splitlines()
    list_labels=file_labels.read().splitlines()

    porter=nltk.PorterStemmer()

    list_tweets_2=[]    ## tokenized (word_tokenize)
    list_tweets_3=[]    ## @'s and url's removed
    list_tweets_4=[]    ## punctuations and digits removed and de-capitalized and stemmed
    list_tweets_5=[]    ## removing the stop words

    list_stopwords_2=[]     ## stemmed stopwords
    

    list_combined_vocab=[]      ## contains 10 dictionaries, each corresponding to a different cross validation set
    list_combined_vocab_2=[]    ## contains a pruned set of above 10 dictionaries
    dict_augmented_vocab={}     ## contains average of normal and multinomial probabilities of words belonging to class
    dict_augmented_vocab_2={}   ## contains the average of the above 10 dictionaries for common words, plus raw count and standard devaition for pruned vocabulary
    dict_lexicon=eval(file_lexicon.read())  ## contains summable score for each word (the keys have to be stemmed)


    dict_final_sum={}           ## contains summable score for each word (joint prob. of word with objective class)
    dict_final_bayes={}         ## contains [P(word|obj),P(word|subj)]
    
    dict_final_lexicon={}       ## stemmed version of dict_lexicon


    list_score_sum=[0]*(len_obj+len_subj)       ## score of each tweet according to the sum of the word models
    list_score_bayes=[]                         ## p(obj|tweet) for each tweet. P(subj|tweet) = 1 - P(obj|tweet)

    list_score_lexicon=[0]*(len_obj+len_subj)
    list_score_lexicon_norm=[0]*(len_obj+len_subj)      ## normalized version of the above in terms of number of words in list_tweets_5


##########################
    

    for tweet in list_tweets:
        list_tweets_2.append(nltk.word_tokenize(tweet))             ## for populating list_tweets_2



    for index in range(0,len(list_tweets_2)):                       ## for populating list_tweets_3
        ind=0                                                          
        phrase=[]
        while ind < len(list_tweets_2[index]):
            word=list_tweets_2[index][ind]
            if word == "@":
                ind=ind+2
            elif word == "http":
                ind=ind+3
            else:
                phrase.append(word)
                ind=ind+1
        list_tweets_3.append(phrase)



    for tweet in list_tweets_3:
        tweet=string.join(tweet)

        tweet=[x for x in tweet]        ## for splitting the tweet sting by each character

        for digit in string.digits:
            tweet[:]=[z for z in tweet if z!= digit]        ## digits removed

        tweet=string.join(tweet,"")
        tweet=nltk.wordpunct_tokenize(tweet)

        for punct in string.punctuation:
            tweet[:]=[z for z in tweet if z!= punct]        ## punctuations removed

        list_tweets_4.append(tweet)



    for index in range(0,len(list_tweets_4)):
        for ind in range(0,len(list_tweets_4[index])):
            list_tweets_4[index][ind]=list_tweets_4[index][ind].lower()         ## removed word capitalizations
            list_tweets_4[index][ind]=porter.stem(list_tweets_4[index][ind])    ## stem the words



    for stopword in list_stopwords:         ## Stemming the stopwords
        stopword=porter.stem(stopword)
        list_stopwords_2.append(stopword)


    list_tweets_5=list_tweets_4[:]

    list_tweets_5=eval(str(list_tweets_5)) ## to remove any link of list_5 with list_4 (problem was earlier detected)


    for tweet in list_tweets_5:
        for stopword in list_stopwords_2:
            tweet[:]=[x for x in tweet if x!= stopword]



    words_obj=0
    words_subj=0
    
    for i in range(0,len(list_labels)):
        if list_labels[i]=="0":
            words_obj+=len(list_tweets_5[i])
        else:
            words_subj+=len(list_tweets_5[i])



    for word in dict_lexicon.keys():       ## stemming the lexicon dictionary keys
        word_stem=porter.stem(word)
        dict_final_lexicon[word_stem]=dict_lexicon[word]


############################

    portion_size=len(list_tweets_5)/10  ## the last portion might be a little larger
    
    ## the following are 10 equal portions of list_tweets_5 for cross validation while calculating the word polarity models

    list_tweets_5_1=list_tweets_5[0:portion_size]  
    list_tweets_5_2=list_tweets_5[portion_size:2*portion_size]
    list_tweets_5_3=list_tweets_5[2*portion_size:3*portion_size]
    list_tweets_5_4=list_tweets_5[3*portion_size:4*portion_size]
    list_tweets_5_5=list_tweets_5[4*portion_size:5*portion_size]
    list_tweets_5_6=list_tweets_5[5*portion_size:6*portion_size]
    list_tweets_5_7=list_tweets_5[6*portion_size:7*portion_size]
    list_tweets_5_8=list_tweets_5[7*portion_size:8*portion_size]
    list_tweets_5_9=list_tweets_5[8*portion_size:9*portion_size]
    list_tweets_5_10=list_tweets_5[9*portion_size:]

    ## the following are 10 different combinations of list_tweets_5_x

    crossval_tweets_1=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9    
    crossval_tweets_2=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_10
    crossval_tweets_3=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_4=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_5=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_6=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_7=list_tweets_5_1+list_tweets_5_2+list_tweets_5_3+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_8=list_tweets_5_1+list_tweets_5_2+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_9=list_tweets_5_1+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10
    crossval_tweets_10=list_tweets_5_2+list_tweets_5_3+list_tweets_5_4+list_tweets_5_5+list_tweets_5_6+list_tweets_5_7+list_tweets_5_8+list_tweets_5_9+list_tweets_5_10

    ## the following is the combined list of crossval_tweets_x for easy iterative calculation

    crossval_tweets_combined=[crossval_tweets_1,crossval_tweets_2,crossval_tweets_3,crossval_tweets_4,crossval_tweets_5,crossval_tweets_6,crossval_tweets_7,crossval_tweets_8,crossval_tweets_9,crossval_tweets_10,list_tweets_5]


    ## now doing the same for labels:

    labels_1=list_labels[0:portion_size]  
    labels_2=list_labels[portion_size:2*portion_size]
    labels_3=list_labels[2*portion_size:3*portion_size]
    labels_4=list_labels[3*portion_size:4*portion_size]
    labels_5=list_labels[4*portion_size:5*portion_size]
    labels_6=list_labels[5*portion_size:6*portion_size]
    labels_7=list_labels[6*portion_size:7*portion_size]
    labels_8=list_labels[7*portion_size:8*portion_size]
    labels_9=list_labels[8*portion_size:9*portion_size]
    labels_10=list_labels[9*portion_size:]

    crossval_labels_1=labels_1+labels_2+labels_3+labels_4+labels_5+labels_6+labels_7+labels_8+labels_9    
    crossval_labels_2=labels_1+labels_2+labels_3+labels_4+labels_5+labels_6+labels_7+labels_8+labels_10
    crossval_labels_3=labels_1+labels_2+labels_3+labels_4+labels_5+labels_6+labels_7+labels_9+labels_10
    crossval_labels_4=labels_1+labels_2+labels_3+labels_4+labels_5+labels_6+labels_8+labels_9+labels_10
    crossval_labels_5=labels_1+labels_2+labels_3+labels_4+labels_5+labels_7+labels_8+labels_9+labels_10 
    crossval_labels_6=labels_1+labels_2+labels_3+labels_4+labels_6+labels_7+labels_8+labels_9+labels_10
    crossval_labels_7=labels_1+labels_2+labels_3+labels_5+labels_6+labels_7+labels_8+labels_9+labels_10
    crossval_labels_8=labels_1+labels_2+labels_4+labels_5+labels_6+labels_7+labels_8+labels_9+labels_10
    crossval_labels_9=labels_1+labels_3+labels_4+labels_5+labels_6+labels_7+labels_8+labels_9+labels_10
    crossval_labels_10=labels_2+labels_3+labels_4+labels_5+labels_6+labels_7+labels_8+labels_9+labels_10

    crossval_labels_combined=[crossval_labels_1,crossval_labels_2,crossval_labels_3,crossval_labels_4,crossval_labels_5,crossval_labels_6,crossval_labels_7,crossval_labels_8,crossval_labels_9,crossval_labels_10,list_labels]


   ## note: the last element is for the orignial tweet list and labels list for reference
    

############################

    for index in range(0,len(crossval_tweets_combined)):

        crossval_set=crossval_tweets_combined[index]
        crossval_labels=crossval_labels_combined[index]

        dict_vocab={}       ## dictionary counting occurence of each vocabulary word (in terms of number of tweets)
        dict_vocab_2={}     ## subset of dict_vocab according to certain constraints thresholds (10 occurences and 0.65 value)
        
        for i in range(0,len(crossval_set)):
            tweet=crossval_set[i]
            for word in tweet:
                if word in dict_vocab:
                    dict_vocab[word][0]+=1
                else:
                    dict_vocab[word]=[1,0,0,0]  ## 1st raw word count, 2nd combined probability with objective class, 3rd prob. of word in objective class
                if crossval_labels[i]=="0":         ## 4th prob. in subjective class
                    dict_vocab[word][2]+=1
                else:
                    dict_vocab[word][3]+=1



## ADD LAPLACE SMOOTHING BELOW:
                    
        for word in dict_vocab.keys():
            dict_vocab[word][2]=float(dict_vocab[word][2])/(words_obj)    ## P(word|obj)
            dict_vocab[word][3]=float(dict_vocab[word][3])/(words_subj)   ## P(word|subj)
            dict_vocab[word][1]=float(dict_vocab[word][2])/float(dict_vocab[word][2]+dict_vocab[word][3])   ## P(word,objective_class). for subjective class simply subtract with 1
        

        for word in dict_vocab.keys():
            if dict_vocab[word][0]>=count_min:
                if dict_vocab[word][1]>=score_min or dict_vocab[word][1]<=1-score_min:
                    dict_vocab_2[word]=dict_vocab[word]


        list_combined_vocab.append(dict_vocab)      ## for naive-bayes based models
        list_combined_vocab_2.append(dict_vocab_2)        ## for summing based models



#############################
        


    l=list_combined_vocab
    for word in l[0]:
        if word in l[1] and word in l[2] and word in l[3] and word in l[4] and word in l[5] and word in l[6] and word in l[7] and word in l[8] and word in l[9]:
            dict_augmented_vocab[word]=[0,0,0]  ## 1st raw count, 2nd P(word|obj), 3rd P(word|subj)
            dict_augmented_vocab[word][0]=l[-1][word][0]
            dict_augmented_vocab[word][1]=float(l[0][word][2]+l[1][word][2]+l[2][word][2]+l[3][word][2]+l[4][word][2]+l[5][word][2]+l[6][word][2]+l[7][word][2]+l[8][word][2]+l[9][word][2])/10
            dict_augmented_vocab[word][2]=float(l[0][word][3]+l[1][word][3]+l[2][word][3]+l[3][word][3]+l[4][word][3]+l[5][word][3]+l[6][word][3]+l[7][word][3]+l[8][word][3]+l[9][word][3])/10



    l=list_combined_vocab_2
    for word in l[0]:
        if word in l[1] and word in l[2] and word in l[3] and word in l[4] and word in l[5] and word in l[6] and word in l[7] and word in l[8] and word in l[9]:
            dict_augmented_vocab_2[word]=[0,0,0]  ## 1st is raw count, 2nd is mean score of objectivity, 3rd is standard devaition of mean score
            dict_augmented_vocab_2[word][0]=l[-1][word][0]
            dict_augmented_vocab_2[word][1]=float(l[0][word][1]+l[1][word][1]+l[2][word][1]+l[3][word][1]+l[4][word][1]+l[5][word][1]+l[6][word][1]+l[7][word][1]+l[8][word][1]+l[9][word][1])/10
            u=dict_augmented_vocab_2[word][1]
            dict_augmented_vocab_2[word][2]=float(((l[0][word][1]-u)**2)+((l[1][word][1]-u)**2)+((l[2][word][1]-u)**2)+((l[3][word][1]-u)**2)+((l[4][word][1]-u)**2)+((l[5][word][1]-u)**2)+((l[6][word][1]-u)**2)+((l[7][word][1]-u)**2)+((l[8][word][1]-u)**2)+((l[9][word][1]-u)**2))/10
            dict_augmented_vocab_2[word][2]=float((dict_augmented_vocab_2[word][2])**0.5)



##########################

            

    for word in dict_augmented_vocab_2.keys():
        dict_final_sum[word]=2*(dict_augmented_vocab_2[word][1]-0.5)  ## normalizing score b/w +1 and -1



    for word in dict_augmented_vocab.keys():
        dict_final_bayes[word]=[dict_augmented_vocab[word][1],dict_augmented_vocab[word][2]]    ## [P(word|obj), P(word|subj)]
        


############################
    

    for i in range(0,len(list_tweets_5)):       ## list_score_sum and list_score_lexicon
        tweet=list_tweets_5[i]
        for word in tweet:
            if word in dict_final_sum.keys():
                list_score_sum[i]+=dict_final_sum[word]
            if word in dict_final_lexicon.keys():
                list_score_lexicon[i]+=math.fabs(dict_final_lexicon[word])


    for i in range(0,len(list_tweets_5)):       ## list_score_lexicon
        if list_score_lexicon[i]!=0:
            list_score_lexicon_norm[i]=float(list_score_lexicon[i])/len(list_tweets_5[i])
        else:
            list_score_lexicon_norm[i]=0



##    for i in range(0,len(list_tweets_5)):       ## for list_score_bayes 
##        tweet=list_tweets_5[i]
##        obj=1.0
##        subj=1.0
##        for word in tweet:
##            if word in dict_final_bayes.keys():
##                if dict_final_bayes[word][0]!=0 and dict_final_bayes[word][1]!=0:
##                    obj=float(obj)*dict_final_bayes[word][0]
##                    subj=float(subj)*dict_final_bayes[word][1]
##        obj=obj*float(len_obj)/len(list_tweets_5)
##        subj=subj*float(len_subj)/len(list_tweets_5)
##        if obj==0 and subj==0:
##            list_score_bayes.append(0.5)
##        else:
##            list_score_bayes.append(obj/float(subj+obj))

        
    
                
    
            

    ##return list_tweets,list_tweets_2,list_tweets_3,list_tweets_4,list_tweets_5
    return list_score_lexicon_norm,dict_final_lexicon,list_tweets_5,list_labels
    ##return crossval_tweets_combined,crossval_labels_combined
            
    
