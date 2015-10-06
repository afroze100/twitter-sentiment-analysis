def word_obj_test():

    import nltk
    import string
    import math

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_nonamb_8963.txt","r")
    file_stopwords=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/stopwords.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/labels_obj_nonamb_8963.txt","r")
##    file_lexicon=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/dict_mpqa_stem.txt","r")


    len_obj=4543
    len_subj=4420
##    len_subj=5261
    count_min=4    ## the 2 following are thresholds for pruning of word polarity models
    score_min=0.5
    smoothing_factor=1
    
    
    list_tweets=file_tweets.read().splitlines()
    list_stopwords=file_stopwords.read().splitlines()
    list_labels=file_labels.read().splitlines()

    porter=nltk.PorterStemmer()

    list_tweets_2=[]    ## tokenized (word_tokenize)
    list_tweets_3=[]    ## @'s and url's removed
    list_tweets_4=[]    ## punctuations and digits removed and de-capitalized and stemmed
    list_tweets_5=[]    ## removing the stop words

    list_stopwords_2=[]     ## stemmed stopwords

    test_tweets=[]
    training_tweets=[]
    test_labels=[]
    training_labels=[]

    dict_vocab={}       ## dictionary of each occuring word (generally for naive bayes model)
    dict_vocab_2={}     ## subset of dict_vocab acc. to some thrshold (generally for summing model)
##    dict_final_lexicon=eval(file_lexicon.read())
    

    dict_final_sum={}           ## contains summable score for each word (joint prob. of word with objective class)
    dict_final_bayes={}         ## contains [P(word|obj),P(word|subj)]


    list_score_sum=[]       ## score of each tweet according to the sum of the word models
    list_score_bayes=[]     ## p(obj|tweet) for each tweet. P(subj|tweet) = 1 - P(obj|tweet)
    list_score_lexicon=[0]*(len_obj+len_subj)

    final_list=[]


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



    for stopword in list_stopwords:
        stopword=porter.stem(stopword)
        list_stopwords_2.append(stopword)


    list_tweets_5=list_tweets_4[:]
    list_tweets_5=eval(str(list_tweets_5)) ## to remove any link of list_5 with list_4 (problem was earlier detected)


    for tweet in list_tweets_5:
        for stopword in list_stopwords_2:
            tweet[:]=[x for x in tweet if x!= stopword]
        


############################

    ## Separating test and training data (from tweets and their labels)

    portion_size=len(list_tweets_5)/10      ## dividing data into 10 equal portions
    select_size=int(0.1*portion_size)      ## 5% of tweets from each portion reserved for test data
    
    for i in range(0,10):
        test_tweets=test_tweets+list_tweets_5[i*portion_size:i*portion_size+select_size]
        test_labels=test_labels+list_labels[i*portion_size:i*portion_size+select_size]
        training_tweets=training_tweets+list_tweets_5[i*portion_size+select_size:(i+1)*portion_size]
        training_labels=training_labels+list_labels[i*portion_size+select_size:(i+1)*portion_size]




    words_obj_train=0
    words_subj_train=0    
    for i in range(0,len(list_labels)):
        if list_labels[i]=="0":
            words_obj_train+=len(list_tweets_5[i])
        else:
            words_subj_train+=len(list_tweets_5[i])

    len_obj_test=0
    len_subj_test=0
    for i in range(0,len(list_labels)):
        if list_labels[i]=="0":
            len_obj_test+=1
        else:
            len_subj_test+=1
    

############################
            
        
    for i in range(0,len(list_tweets_5)):
        tweet=list_tweets_5[i]
        for word in tweet:
            if word in dict_vocab:
                dict_vocab[word][0]+=1
            else:
                dict_vocab[word]=[1,0,0,0]  ## 1st raw word count, 2nd combined probability with objective class, 3rd P(word|obj), 4th P(word|subj)
            if list_labels[i]=="0":         
                dict_vocab[word][2]+=1
            else:
                dict_vocab[word][3]+=1



    for word in dict_vocab.keys():
        if dict_vocab[word][0]>=count_min:
            if dict_vocab[word][1]>=score_min or dict_vocab[word][1]<=1-score_min:
                dict_vocab_2[word]=dict_vocab[word]


## ADD LAPLACE SMOOTHING BELOW:
                    
    for word in dict_vocab_2.keys():
        dict_vocab_2[word][2]=float(dict_vocab_2[word][2]+smoothing_factor)/(words_obj_train+(smoothing_factor*len(dict_vocab_2)))    ## P(word|obj)
        dict_vocab_2[word][3]=float(dict_vocab_2[word][3]+smoothing_factor)/(words_subj_train+(smoothing_factor*len(dict_vocab_2)))   ## P(word|subj)
        dict_vocab_2[word][1]=float(dict_vocab_2[word][2])/float(dict_vocab_2[word][2]+dict_vocab_2[word][3])   ## P(word,obj). for subjective class simply subtract with 1
        

##########################

            

    for word in dict_vocab_2.keys():
        dict_final_sum[word]=2*(dict_vocab_2[word][1]-0.5)  ## normalizing score b/w +1 and -1



    for word in dict_vocab_2.keys():
        dict_final_bayes[word]=[dict_vocab_2[word][2], dict_vocab_2[word][3]]    ## [P(word|obj), P(word|subj)]
        


############################



    for i in range(0,len(test_tweets)):       ## for list_score_bayes 
        tweet=test_tweets[i]
        obj=1.0
        subj=1.0
        for word in tweet:
            if word in dict_final_bayes.keys():
                obj=float(obj)*dict_final_bayes[word][0]
                subj=float(subj)*dict_final_bayes[word][1]
##        obj=obj*float(len_obj)/len(list_tweets_5)
##        subj=subj*float(len_subj)/len(list_tweets_5)
        if obj==0 and subj==0:
            list_score_bayes.append(0.5)
        else:
            list_score_bayes.append(obj/float(subj+obj))


#############################


    return dict_final_bayes

