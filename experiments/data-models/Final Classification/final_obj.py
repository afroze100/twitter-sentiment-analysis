def final_obj():

    import nltk
    import string
    import math
    import re

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_nonamb_8963.txt","r")
    file_stopwords=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/stopwords.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/labels_obj_nonamb_8963.txt","r")
    file_dict_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/dict_mpqa_stem.txt","r")


    len_obj=4543
    len_subj=4420
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


    words_obj_train=0
    words_subj_train=0    
    for i in range(0,len(training_labels)):
        if training_labels[i]=="0":
            words_obj_train+=len(list_tweets_5_training[i])
        else:
            words_subj_train+=len(list_tweets_5_training[i])

    len_obj_test=0
    len_subj_test=0
    for i in range(0,len(test_labels)):
        if test_labels[i]=="0":
            len_obj_test+=1
        else:
            len_subj_test+=1
            


        
    for i in range(0,len(list_tweets_5_training)):
        tweet=list_tweets_5_training[i]
        for word in tweet:
            if word in dict_vocab:
                dict_vocab[word][0]+=1
            else:
                dict_vocab[word]=[1,0,0,0]  ## 1st raw word count, 2nd combined probability with objective class, 3rd P(word|obj), 4th P(word|subj)
            if training_labels[i]=="0":         
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



    for word in dict_vocab_2.keys():
        dict_final_bayes[word]=[dict_vocab_2[word][2],dict_vocab_2[word][3]]    ## [P(word|obj), P(word|subj)]
        


###########################################################################################################################################################################
                                    ## BELOW THIS POINT CODE HAS TO BE CALCULATED SEPARATELY FOR TEST OR TRAINING DATA ##




    list_tweets = list_tweets_test   ## Overwritten variable from above
    list_tweets_2 = list_tweets_2_test
    list_tweets_3 = list_tweets_3_test
    list_tweets_4 = list_tweets_4_test
    list_tweets_5 = list_tweets_5_test

    list_labels = test_labels   ## Overwritten variable from above




##    list_tweets = list_tweets_training   ## Overwritten variable from above
##    list_tweets_2 = list_tweets_2_training
##    list_tweets_3 = list_tweets_3_training
##    list_tweets_4 = list_tweets_4_training
##    list_tweets_5 = list_tweets_5_training
##
##    list_labels = training_labels   ## Overwritten variable from above


#####################
    
    list_pos=[] ## list of pos tags of each tweet

    pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)|[Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
    neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)|[\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."
    
    final_list=[]
    final_list_2=[]



#############################################################################  LIST OF FEATURES

    
    ## Features

    num_exclamations=[]
    presence_questions=[]
    presence_url=[]
    presence_emoticon=[]

    pos_prp_combined=[]           ## sum of pos_prp & pos_prp$
    pos_nnp_combined=[]           ## sum of pos_nnp & pos_nnps

    words_mpqa=[]
    words_bayes = []


    ## Extra features

    num_questions=[]
    presence_exclamations=[]
    num_digits=[] 
    num_caps_words=[]
    num_caps_chars=[]
    num_punct=[]
    num_emoticon=[]

    pos_jj=[]
    pos_jjs=[]
    pos_jjr=[]
    pos_vb=[]
    pos_vbg=[]
    pos_vbn=[]
    pos_vbz=[]
    pos_vbp=[] 
    pos_rb=[] 
    pos_prp=[]
    pos_prpS=[]
    pos_nnp=[]
    pos_nnps=[]
    pos_cd=[]
    pos_pos=[]
    pos_wp=[]
    pos_jj_combined=[]         ## sum of pos_jj, pos_jjs & pos_jjr
    pos_vb_combined=[]         ## sum of pos_vb, pos_vbg, pos_vbn & pos_vbz

    

############################################################################  FEATURE EXTRACTION



    for index in range(0,len(list_tweets)):

        num_exclamations.append(string.count(list_tweets[index],"!"))   ## exclamation marks

        if string.count(list_tweets[index],"?")==0:                         ## question marks
            presence_questions.append("no")
        else:
            presence_questions.append("yes")

        if string.find(list_tweets[index],"http://") == -1:                  ## url's
            presence_url.append("no") 
        else:
            presence_url.append("yes")



    for tweet in list_tweets:                                               ## for emoticons

        if re.search(pos_emoticon_re,tweet) or re.search(neg_emoticon_re,tweet):
            presence_emoticon.append("yes")
        else:
            presence_emoticon.append("no")




########### Extra Features


    for index in range(0,len(list_tweets)):
        tweet = list_tweets[index]
        num = len(re.findall(pos_emoticon_re, tweet)) + len(re.findall(neg_emoticon_re, tweet))
        num_emoticon.append(str(num))
        

    for index in range(0,len(list_tweets)):

        if string.count(list_tweets[index],"!")==0:                         ## presence of exclamation marks
            presence_exclamations.append("no")
        else:
            presence_exclamations.append("yes")

        num_questions.append(str(string.count(list_tweets[index],"?")))      ## number of question marks


    for tweet in list_tweets_3:                                 
        tweet_string=string.join(tweet)                                     ## for number of digits in a tweet
        num=0
        for char in tweet_string:
            if char in string.digits:
                num+=1
        num_digits.append(str(num))


    for tweet in list_tweets_3:                                             ## for number of punctuation marks in a tweet                         
        tweet=string.join(tweet)
        num=0
        for char in tweet:
            if char in string.punctuation:
                num+=1
        num_punct.append(str(num))


    for tweet in list_tweets_3:                                             ## for number of capitalized words in a tweet
        tweet=string.join(tweet)
        tweet=nltk.wordpunct_tokenize(tweet)
        num_words=0
        num_chars=0
        for word in tweet:
            length=len(word)
            count=0
            for char in word:
                if char in string.uppercase:
                    count+=1
                    num_chars+=1
            if count == length:
                    num_words+=1
        num_caps_words.append(str(num_words))
        num_caps_chars.append(str(num_chars))
        





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
        pos_prp_combined.append(pos.get("PRP",0) + pos.get("PRP$",0))
        pos_nnp_combined.append(pos.get("NNP",0) + pos.get("NNPS",0))


####### Extra POS features

    for pos in pos_list:
        pos_jj.append(str(pos.get("JJ",0)))
        pos_jjs.append(str(pos.get("JJS",0)))
        pos_jjr.append(str(pos.get("JJR",0)))
        pos_vb.append(str(pos.get("VB",0)))
        pos_vbg.append(str(pos.get("VBG",0)))
        pos_vbn.append(str(pos.get("VBN",0)))
        pos_vbz.append(str(pos.get("VBZ",0)))
        pos_vbp.append(str(pos.get("VBP",0)))
        pos_rb.append(str(pos.get("RB",0)))
        pos_prp.append(str(pos.get("PRP",0)))
        pos_prpS.append(str(pos.get("PRP$",0)))
        pos_nnp.append(str(pos.get("NNP",0)))
        pos_nnps.append(str(pos.get("NNPS",0)))
        pos_cd.append(str(pos.get("CD",0)))
        pos_pos.append(str(pos.get("POS",0)))
        pos_wp.append(str(pos.get("WP",0)))
        pos_jj_combined.append(str(pos.get("JJ",0) + pos.get("JJS",0) + pos.get("JJR",0)))
        pos_vb_combined.append(str(pos.get("VB",0) + pos.get("VBG",0) + pos.get("VBN",0) + pos.get("VBZ",0)))



########### WORDS RELATED FEATURES


    for tweet in list_tweets_4:         ## MPQA word lexicon
        score=0
        for word in tweet:
            if word in dict_mpqa.keys():
                score += math.fabs(dict_mpqa[word])
        words_mpqa.append(score)




    for i in range(0,len(list_tweets_5)):       ## Naive bayes WORD MODELLING
        tweet=list_tweets_5[i]
        obj=1.0
        subj=1.0
        for word in tweet:
            if word in dict_final_bayes.keys():
                obj=float(obj)*dict_final_bayes[word][0]
                subj=float(subj)*dict_final_bayes[word][1]
##        obj=obj*float(len_obj)/len(list_tweets)
##        subj=subj*float(len_subj)/len(list_tweets)
        if obj==0 and subj==0:
            list_score_bayes.append(0.5)
        else:
            list_score_bayes.append(obj/float(subj+obj))

    words_bayes = list_score_bayes




####################################################################################  GENERATING WEKA FILE



    for index in range(0,len(list_tweets)):                     ## for ease in generating the csv file for WEKA
        final_list.append([str(presence_url[index]),str(presence_emoticon[index]),str(words_mpqa[index]),str(words_bayes[index]),str(pos_prp_combined[index]),str(num_exclamations[index]),str(pos_nnp_combined[index]),str(presence_questions[index]),str(num_questions[index]),str(presence_exclamations[index]),str(num_digits[index]),str(num_caps_words[index]),str(num_caps_chars[index]),str(num_punct[index]),str(num_emoticon[index]),str(pos_jj[index]),str(pos_jjr[index]),str(pos_jjs[index]),str(pos_vb[index]),str(pos_vbn[index]),str(pos_vbg[index]),str(pos_vbp[index]),str(pos_vbz[index]),str(pos_rb[index]),str(pos_prp[index]),str(pos_prpS[index]),str(pos_nnp[index]),str(pos_nnps[index]),str(pos_cd[index]),str(pos_pos[index]),str(pos_wp[index]),str(pos_jj_combined[index]),str(pos_vb_combined[index]),str(list_labels[index])])
        final_list_2.append([str(presence_url[index]),str(presence_emoticon[index]),str(words_mpqa[index]),str(words_bayes[index]),str(pos_prp_combined[index]),str(num_exclamations[index])])

    for ind in range(0,len(final_list)):
        final_list[ind]=string.join(final_list[ind],",")
        final_list_2[ind]=string.join(final_list_2[ind],",")

    final_list=string.join(final_list,"\n")
    final_list_2=string.join(final_list_2,"\n")




##    return final_list_2, words_bayes, list_tweets, list_labels

    return dict_final_bayes







    


