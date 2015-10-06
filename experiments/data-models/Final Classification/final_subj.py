def final_subj():

    import nltk
    import string
    import math
    import re

##    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification/tweets_subj_nonamb_4420.txt","r")
    file_stopwords=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Features/stopwords.txt","r")
##    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification/labels_subj_nonamb_4420.txt","r")
    file_dict_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Features/dict_mpqa_stem.txt","r")

    file_tweets = open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_nonamb_8963.txt","r")
    file_labels = open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Final Classification/labels_8963.txt","r")


    len_pos=2543
    len_neg=1877
    count_min=1    ## the 2 following are thresholds for pruning of word polarity models
    score_min=0.5
    smoothing_factor=2
    
    
    list_tweets=file_tweets.read().splitlines()
    list_stopwords=file_stopwords.read().splitlines()
    list_labels=file_labels.read().splitlines()
    dict_mpqa = eval(file_dict_mpqa.read())

    porter=nltk.PorterStemmer()


#############################

    test_tweets=[]
    training_tweets=[]
    test_labels=[]
    training_labels=[]

    list_tweets_training=[]
    list_tweets_test=[]

    list_tweets_2_training=[]    ## tokenized (word_tokenize)
    list_tweets_2_test=[]
    list_tweets_3_training=[]    ## @'s and url's removed
    list_tweets_3_test=[]
    list_tweets_4_training=[]    ## punctuations and digits removed and de-capitalized and stemmed
    list_tweets_4_test=[]
    list_tweets_5_training=[]    ## removing the stop words
    list_tweets_5_test=[]
    

    list_stopwords_2=[]     ## stemmed stopwords

    dict_vocab={}       ## dictionary of each occuring word (generally for naive bayes model)
    dict_vocab_2={}     ## subset of dict_vocab acc. to some thrshold (generally for summing model)
    dict_final_bayes={}         ## contains [P(word|obj),P(word|subj)]                                 USE THIS DICTIONARY

    list_score_bayes=[]     ## p(obj|tweet) for each tweet. P(subj|tweet) = 1 - P(obj|tweet)



########################################## TRAINING AND TEST DATA SEPARATION
            
            

    selector = 4               ## can be any itegerer from 0-9, each value returns with a different test/training sample for 10-fold cross validation
    



    portion_size=len(list_tweets)/10      ## dividing data into 10 equal portions
    select_size=int(0.1*portion_size)      ## 10% of tweets from each portion reserved for test data
    

    for i in range(0,10):
        test_tweets = test_tweets + list_tweets[(i*portion_size)+(selector*select_size):(i*portion_size)+(selector*select_size)+select_size]
        test_labels = test_labels + list_labels[(i*portion_size)+(selector*select_size):(i*portion_size)+(selector*select_size)+select_size]
        training_tweets = training_tweets + list_tweets[i*portion_size:(i*portion_size)+(selector*select_size)] + list_tweets[(i*portion_size)+(selector*select_size)+select_size:(i+1)*portion_size]
        training_labels = training_labels + list_labels[i*portion_size:(i*portion_size)+(selector*select_size)] + list_labels[(i*portion_size)+(selector*select_size)+select_size:(i+1)*portion_size]


    list_tweets_training = training_tweets[:]
    list_tweets_test = test_tweets[:]


##################################### TWEET FORMATTING FOR FEATURE EXTRACTION
    

    for tweet in list_tweets_training:
        list_tweets_2_training.append(nltk.word_tokenize(tweet))             ## for populating list_tweets_2

    for tweet in list_tweets_test:
        list_tweets_2_test.append(nltk.word_tokenize(tweet))


####
        

    for index in range(0,len(list_tweets_2_training)):                       ## for populating list_tweets_3
        ind=0                                                          
        phrase=[]
        while ind < len(list_tweets_2_training[index]):
            word=list_tweets_2_training[index][ind]
            if word == "@":
                ind=ind+2
            elif word == "http":
                ind=ind+3
            else:
                phrase.append(word)
                ind=ind+1
        list_tweets_3_training.append(phrase)


    for index in range(0,len(list_tweets_2_test)):                       
        ind=0                                                          
        phrase=[]
        while ind < len(list_tweets_2_test[index]):
            word=list_tweets_2_test[index][ind]
            if word == "@":
                ind=ind+2
            elif word == "http":
                ind=ind+3
            else:
                phrase.append(word)
                ind=ind+1
        list_tweets_3_test.append(phrase)


####
        

    for tweet in list_tweets_3_training:
        tweet=string.join(tweet)

        tweet=[x for x in tweet]        ## for splitting the tweet sting by each character

        for digit in string.digits:
            tweet[:]=[z for z in tweet if z!= digit]        ## digits removed

        tweet=string.join(tweet,"")
        tweet=nltk.wordpunct_tokenize(tweet)

        for punct in string.punctuation:
            tweet[:]=[z for z in tweet if z!= punct]        ## punctuations removed

        list_tweets_4_training.append(tweet)



    for tweet in list_tweets_3_test:
        tweet=string.join(tweet)

        tweet=[x for x in tweet]        

        for digit in string.digits:
            tweet[:]=[z for z in tweet if z!= digit]        

        tweet=string.join(tweet,"")
        tweet=nltk.wordpunct_tokenize(tweet)

        for punct in string.punctuation:
            tweet[:]=[z for z in tweet if z!= punct]        

        list_tweets_4_test.append(tweet)


####
        

    for index in range(0,len(list_tweets_4_training)):
        for ind in range(0,len(list_tweets_4_training[index])):
            list_tweets_4_training[index][ind]=list_tweets_4_training[index][ind].lower()         ## removed word capitalizations
            list_tweets_4_training[index][ind]=porter.stem(list_tweets_4_training[index][ind])    ## stem the words

    for stopword in list_stopwords:
        stopword=porter.stem(stopword)
        list_stopwords_2.append(stopword)

    list_tweets_5_training=list_tweets_4_training[:]
    list_tweets_5_training=eval(str(list_tweets_5_training)) ## to remove any link of list_5 with list_4 (problem was earlier detected)

    for tweet in list_tweets_5_training:
        for stopword in list_stopwords_2:
            tweet[:]=[x for x in tweet if x!= stopword]




    for index in range(0,len(list_tweets_4_test)):
        for ind in range(0,len(list_tweets_4_test[index])):
            list_tweets_4_test[index][ind]=list_tweets_4_test[index][ind].lower()         
            list_tweets_4_test[index][ind]=porter.stem(list_tweets_4_test[index][ind])    

    list_tweets_5_test=list_tweets_4_test[:]
    list_tweets_5_test=eval(str(list_tweets_5_test)) 

    for tweet in list_tweets_5_test:
        for stopword in list_stopwords_2:
            tweet[:]=[x for x in tweet if x!= stopword]
        




########################################## FOR TRAINING WORD MODELS



    words_pos_train=0           ## total number of words in training data
    words_neg_train=0    
    for i in range(0,len(training_labels)):
        if training_labels[i]=="1":
            words_pos_train+=len(list_tweets_5_training[i])
        elif training_labels[i] == "-1":
            words_neg_train+=len(list_tweets_5_training[i])

    len_pos_test=0              ## number of tweets in test data
    len_neg_test=0
    for i in range(0,len(test_labels)):
        if test_labels[i]=="1":
            len_pos_test+=1
        elif test_labels[i] == "-1":
            len_neg_test+=1



    for i in range(0,len(list_tweets_5_training)):
        tweet=list_tweets_5_training[i]
        for word in tweet:
            if word in dict_vocab:
                dict_vocab[word][0]+=1
            else:
                dict_vocab[word]=[1,0,0,0]  ## 1st raw word count, 2nd combined probability with pos class, 3rd P(word|pos), 4th P(word|neg)
            if training_labels[i]=="1":         
                dict_vocab[word][2]+=1
            elif training_labels[i] == "-1":
                dict_vocab[word][3]+=1



    for word in dict_vocab.keys():
        if dict_vocab[word][0]>=count_min:
            if dict_vocab[word][1]>=score_min or dict_vocab[word][1]<=1-score_min:
                dict_vocab_2[word]=dict_vocab[word]


## ADD LAPLACE SMOOTHING BELOW:
                    
    for word in dict_vocab_2.keys():
        dict_vocab_2[word][2]=float(dict_vocab_2[word][2]+smoothing_factor)/(words_pos_train+(smoothing_factor*len(dict_vocab_2)))    ## P(word|obj)
        dict_vocab_2[word][3]=float(dict_vocab_2[word][3]+smoothing_factor)/(words_neg_train+(smoothing_factor*len(dict_vocab_2)))   ## P(word|subj)
        dict_vocab_2[word][1]=float(dict_vocab_2[word][2])/float(dict_vocab_2[word][2]+dict_vocab_2[word][3])   ## P(word,obj). for subjective class simply subtract with 1


    for word in dict_vocab_2.keys():
        dict_final_bayes[word]=[dict_vocab_2[word][2],dict_vocab_2[word][3]]    ## [P(word|pos), P(word|neg)]
       
        


###########################################################################################################################################################################
                                    ## BELOW THIS POINT CODE HAS TO BE CALCULATED SEPARATELY FOR TEST OR TRAINING DATA ##




##    list_tweets = list_tweets_test   ## Overwritten variable from above
##    list_tweets_2 = list_tweets_2_test
##    list_tweets_3 = list_tweets_3_test
##    list_tweets_4 = list_tweets_4_test
##    list_tweets_5 = list_tweets_5_test
##
##    list_labels = test_labels   ## Overwritten variable from above




    list_tweets = list_tweets_training   ## Overwritten variable from above
    list_tweets_2 = list_tweets_2_training
    list_tweets_3 = list_tweets_3_training
    list_tweets_4 = list_tweets_4_training
    list_tweets_5 = list_tweets_5_training

    list_labels = training_labels   ## Overwritten variable from above




#####################
    
    list_pos=[] ## list of pos tags of each tweet

    pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)|[Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
    neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)|[\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."
    
    final_list=[]
    final_list_2=[]



#############################################################################  LIST OF FEATURES

    
    
    ## Features
 

    score_emoticon=[]
    score_mpqa_all=[]
    words_bayes=[]
    
    pos_vb_combined=[]
    pos_colon=[]
    pos_rb=[]
    pos_wrb=[]



    ## Extra features

    num_emoticon=[]
    num_emoticon_pos=[]
    num_emoticon_neg=[]
    score_mpqa_pos=[]
    score_mpqa_neg=[]

    pos_vb=[]
    pos_vbp=[]
    pos_vbn=[]
    pos_vbz=[]
    pos_vbg=[]
    pos_nnp=[]
    pos_nns=[]
    pos_cd=[]
    pos_in=[]
    pos_period=[]
    

############################################################################  FEATURE EXTRACTION



    for tweet in list_tweets:                                               ## for emoticon score
        num_emot_pos = len(re.findall(pos_emoticon_re,tweet))
        num_emot_neg = len(re.findall(neg_emoticon_re,tweet))
        score_emoticon.append(num_emot_pos - num_emot_neg)




########### Extra Features


    for tweet in list_tweets:                                               ## for number of emoticons in a tweet
        num_emoticon_pos.append(str(len(re.findall(pos_emoticon_re,tweet))))
        num_emoticon_neg.append(str(len(re.findall(neg_emoticon_re,tweet))))
        num_emoticon.append(str(int(num_emoticon_pos[-1]) + int(num_emoticon_neg[-1])))



################# POS TAGGING features

            
    pos_list = [] ## a list containing pos tagging info in form of dictionary for each tweet
    
    for tweet in list_tweets_3:                                                 ## POS tagging
        pos=nltk.pos_tag(tweet)
        pos_dict={}
        for tag in pos:
            if tag[1] in pos_dict:
                pos_dict[tag[1]]+=1
            else:
                pos_dict[tag[1]]=1
        pos_list.append(pos_dict)



    for pos in pos_list:
        pos_rb.append(pos.get("RB",0))
        pos_wrb.append(pos.get("WRB",0))
        pos_colon.append(pos.get(":",0))
        pos_vb_combined.append(pos.get("VB",0) + pos.get("VBG",0) + pos.get("VBN",0) + pos.get("VBZ",0) + pos.get("VBP",0))



####### Extra POS features

    for pos in pos_list:
        pos_vb.append(str(pos.get("VB",0)))
        pos_vbg.append(str(pos.get("VBG",0)))
        pos_vbn.append(str(pos.get("VBN",0)))
        pos_vbz.append(str(pos.get("VBZ",0)))
        pos_vbp.append(str(pos.get("VBP",0)))
        pos_nnp.append(str(pos.get("NNP",0)))
        pos_nns.append(str(pos.get("NNS",0)))
        pos_cd.append(str(pos.get("CD",0)))
        pos_in.append(str(pos.get("IN",0)))
        pos_period.append(str(pos.get(".",0)))
        

########### WORDS RELATED FEATURES
        


    for tweet in list_tweets_4:                 ## MPQA word lexicon score
        pos_mpqa=0
        neg_mpqa=0
        for word in tweet:
            if word in dict_mpqa.keys():
                if dict_mpqa[word]>0:
                    pos_mpqa+=dict_mpqa[word]
                elif dict_mpqa[word]<0:
                    neg_mpqa+=dict_mpqa[word]

        score_mpqa_pos.append(str(float(pos_mpqa)))
        score_mpqa_neg.append(str(float(neg_mpqa)))
        score_mpqa_all.append(float(pos_mpqa+neg_mpqa))




    for i in range(0,len(list_tweets_5)):       ## for list_score_bayes
        tweet=list_tweets_5[i]                   
        pos=1.0
        neg=1.0
        for word in tweet:
            if word in dict_final_bayes.keys():
                pos=float(pos)*dict_final_bayes[word][0]
                neg=float(neg)*dict_final_bayes[word][1]
##        pos=pos*float(len_pos)/8963
##        neg=neg*float(len_neg)/8963
        if pos==0 and neg==0:
            list_score_bayes.append(0.5)
        else:
            list_score_bayes.append(pos/float(pos+neg))

    words_bayes = list_score_bayes





####################################################################################  GENERATING WEKA FILE



    for index in range(0,len(list_tweets)):                     ## for ease in generating the csv file for WEKA
        final_list.append([str(score_emoticon[index]),str(score_mpqa_all[index]),str(words_bayes[index]),str(pos_vb_combined[index]),str(pos_colon[index]),str(pos_rb[index]),str(pos_wrb[index]),str(num_emoticon[index]),str(num_emoticon_pos[index]),str(num_emoticon_neg[index]),str(score_mpqa_pos[index]),str(score_mpqa_neg[index]),str(pos_vb[index]),str(pos_vbp[index]),str(pos_vbn[index]),str(pos_vbz[index]),str(pos_vbg[index]),str(pos_nnp[index]),str(pos_nns[index]),str(pos_cd[index]),str(pos_in[index]),str(pos_period[index]),str(list_labels[index])])

    for ind in range(0,len(final_list)):
        final_list[ind]=string.join(final_list[ind],",")

    final_list=string.join(final_list,"\n")




    return final_list, words_bayes, list_tweets, list_labels







    


