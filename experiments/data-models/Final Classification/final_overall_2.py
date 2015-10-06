## try varying:
## smoothing factor
## features
## prior probability for subjective class

def final_class():

    import nltk
    import string
    import math
    import re

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_nonamb_8963.txt","r")
    file_stopwords=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/stopwords.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Final Classification/labels_8963.txt","r")
    file_dict_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/dict_mpqa_stem.txt","r")


    len_obj=4543
    len_subj=4420
    len_pos=2543
    len_neg=1877
    
    count_min=1    ## the 2 following are thresholds for pruning of word polarity models
    score_min=0.5
    smoothing_factor=1
    
    
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

    dict_vocab_obj={}       ## dictionary of each occuring word (generally for naive bayes model)
    dict_vocab_obj_2={}     ## subset of dict_vocab acc. to some thrshold (generally for summing model)
    dict_final_bayes_obj={}         ## contains [P(word|obj),P(word|subj)]          USE THIS DICTIONARY

    dict_vocab_subj={}       
    dict_vocab_subj_2={} 
    dict_final_bayes_subj={}         ## contains [P(word|pos),P(word|neg)]          USE THIS DICTIONARY


    list_score_bayes_obj_train=[]     ## p(obj|tweet) for each tweet. P(subj|tweet) = 1 - P(obj|tweet)
    list_score_bayes_obj_test=[]
    list_score_bayes_subj_train=[]    ## p(pos|tweet) for each tweet. P(neg|tweet) = 1 - P(pos|tweet)
    list_score_bayes_subj_test=[]



########################################## TRAINING AND TEST DATA SEPARATION
            
            

    selector = 0               ## can be any itegerer from 0-9, each value returns with a different test/training sample for 10-fold cross validation
    



    portion_size=len(list_tweets)/10      ## dividing data into 10 equal portions
    select_size=int(0.1*portion_size)      ## 10% of tweets from each portion reserved for test data
    

    for i in range(0,10):
        test_tweets = test_tweets + list_tweets[(i*portion_size)+(selector*select_size):(i*portion_size)+(selector*select_size)+select_size]
        test_labels = test_labels + list_labels[(i*portion_size)+(selector*select_size):(i*portion_size)+(selector*select_size)+select_size]
        training_tweets = training_tweets + list_tweets[i*portion_size:(i*portion_size)+(selector*select_size)] + list_tweets[(i*portion_size)+(selector*select_size)+select_size:(i+1)*portion_size]
        training_labels = training_labels + list_labels[i*portion_size:(i*portion_size)+(selector*select_size)] + list_labels[(i*portion_size)+(selector*select_size)+select_size:(i+1)*portion_size]


    list_tweets_training = training_tweets[:]
    list_tweets_test = test_tweets[:]


    print "check_1: Test/Train data separation"


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
        


    print "check_2: Tweet formatting"
    


################################################### FOR TRAINING WORD MODELS




######################## For Objectivity Classification
            


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
            if word in dict_vocab_obj:
                dict_vocab_obj[word][0]+=1
            else:
                dict_vocab_obj[word]=[1,0,0,0]  ## 1st raw word count, 2nd combined probability with objective class, 3rd P(word|obj), 4th P(word|subj)
            if training_labels[i]=="0":         
                dict_vocab_obj[word][2]+=1
            else:
                dict_vocab_obj[word][3]+=1



    for word in dict_vocab_obj.keys():
        if dict_vocab_obj[word][0]>=count_min:
            if dict_vocab_obj[word][1]>=score_min or dict_vocab_obj[word][1]<=1-score_min:
                dict_vocab_obj_2[word]=dict_vocab_obj[word]


## ADD LAPLACE SMOOTHING BELOW:
                    
    for word in dict_vocab_obj_2.keys():
        dict_vocab_obj_2[word][2]=float(dict_vocab_obj_2[word][2]+smoothing_factor)/(words_obj_train+(smoothing_factor*len(dict_vocab_obj_2)))    ## P(word|obj)
        dict_vocab_obj_2[word][3]=float(dict_vocab_obj_2[word][3]+smoothing_factor)/(words_subj_train+(smoothing_factor*len(dict_vocab_obj_2)))   ## P(word|subj)
        dict_vocab_obj_2[word][1]=float(dict_vocab_obj_2[word][2])/float(dict_vocab_obj_2[word][2]+dict_vocab_obj_2[word][3])   ## P(word,obj). for subjective class simply subtract with 1



    for word in dict_vocab_obj_2.keys():
        dict_final_bayes_obj[word]=[dict_vocab_obj_2[word][2],dict_vocab_obj_2[word][3]]    ## [P(word|obj), P(word|subj)]




###################### For Subjectivity classification



    words_pos_train=0           ## total number of words in training data
    words_neg_train=0    
    for i in range(0,len(training_labels)):
        if training_labels[i]=="1":
            words_pos_train+=len(list_tweets_5_training[i])
        elif training_labels[i]=="-1":
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
            if word in dict_vocab_subj:
                dict_vocab_subj[word][0]+=1
            else:
                dict_vocab_subj[word]=[1,0,0,0]  ## 1st raw word count, 2nd combined probability with pos class, 3rd P(word|pos), 4th P(word|neg)
            if training_labels[i]=="1":         
                dict_vocab_subj[word][2]+=1
            elif training_labels[i]=="-1":
        
                dict_vocab_subj[word][3]+=1



    for word in dict_vocab_subj.keys():
        if dict_vocab_subj[word][0]>=count_min:
            if dict_vocab_subj[word][1]>=score_min or dict_vocab_subj[word][1]<=1-score_min:
                dict_vocab_subj_2[word]=dict_vocab_subj[word]


## ADD LAPLACE SMOOTHING BELOW:
                    
    for word in dict_vocab_subj_2.keys():
        dict_vocab_subj_2[word][2]=float(dict_vocab_subj_2[word][2]+smoothing_factor)/(words_pos_train+(smoothing_factor*len(dict_vocab_subj_2)))    ## P(word|pos)
        dict_vocab_subj_2[word][3]=float(dict_vocab_subj_2[word][3]+smoothing_factor)/(words_neg_train+(smoothing_factor*len(dict_vocab_subj_2)))   ## P(word|neg)
        dict_vocab_subj_2[word][1]=float(dict_vocab_subj_2[word][2])/float(dict_vocab_subj_2[word][2]+dict_vocab_subj_2[word][3])   ## P(word,obj). for subjective class simply subtract with 1


    for word in dict_vocab_subj_2.keys():
        dict_final_bayes_subj[word]=[dict_vocab_subj_2[word][2],dict_vocab_subj_2[word][3]]    ## [P(word|pos), P(word|neg)]
    
        

    print "check_3: Training word_models"

    

###########################################################################################################################################################################
                                    ## BELOW THIS POINT CODE HAS TO BE CALCULATED SEPARATELY FOR TEST OR TRAINING DATA ##




#####################



    pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)|[Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
    neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)|[\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."
    
    final_list_test=[]
    final_list_train=[]



#############################################################################  LIST OF FEATURES

    
    ## Objectivity Features

    num_exclamations_train=[]
    presence_url_train=[]
    presence_emoticon_train=[]

    pos_prp_combined_train=[]           ## sum of pos_prp & pos_prp$

    words_mpqa_train=[]
    words_bayes_obj_train = []


#######


    num_exclamations_test=[]
    presence_url_test=[]
    presence_emoticon_test=[]

    pos_prp_combined_test=[]           ## sum of pos_prp & pos_prp$

    words_mpqa_test=[]
    words_bayes_obj_test = []




    ## Subjectivity Features

    score_emoticon_train=[]
    score_mpqa_all_train=[]
    words_bayes_subj_train=[]

    num_emoticon_pos_train=[]
    num_emoticon_neg_train=[]
    score_mpqa_neg_train=[]
    


#######

    score_emoticon_test=[]
    score_mpqa_all_test=[]
    words_bayes_subj_test=[]
    
    num_emoticon_pos_test=[]
    num_emoticon_neg_test=[]
    score_mpqa_neg_test=[]
    

    
    

############################################################################  FEATURE EXTRACTION



#################### Objectivity Feature Extraction => TRAINING DATA



    for index in range(0,len(list_tweets_training)):

        num_exclamations_train.append(string.count(list_tweets_training[index],"!"))   ## exclamation marks

        if string.find(list_tweets_training[index],"http://") == -1:                  ## url's
            presence_url_train.append("no") 
        else:
            presence_url_train.append("yes")



    for tweet in list_tweets_training:                                               ## for emoticons

        if re.search(pos_emoticon_re,tweet) or re.search(neg_emoticon_re,tweet):
            presence_emoticon_train.append("yes")
        else:
            presence_emoticon_train.append("no")


            

    pos_list = [] ## a list containing pos tagging info in form of dictionary for each tweet
    
    for tweet in list_tweets_3_training:                                                 ## POS tagging
        pos=nltk.pos_tag(tweet)
        pos_dict={}
        for tag in pos:
            if tag[1] in pos_dict:
                pos_dict[tag[1]]+=1
            else:
                pos_dict[tag[1]]=1
        pos_list.append(pos_dict)


    for pos in pos_list:
        pos_prp_combined_train.append(pos.get("PRP",0) + pos.get("PRP$",0))




    for tweet in list_tweets_4_training:         ## MPQA word lexicon
        score=0
        for word in tweet:
            if word in dict_mpqa.keys():
                score += math.fabs(dict_mpqa[word])
        words_mpqa_train.append(score)


    

    for i in range(0,len(list_tweets_5_training)):       ## Naive bayes WORD MODELLING
        tweet=list_tweets_5_training[i]
        obj=1.0
        subj=1.0
        for word in tweet:
            if word in dict_final_bayes_obj.keys():
                obj=float(obj)*dict_final_bayes_obj[word][0]
                subj=float(subj)*dict_final_bayes_obj[word][1]
##        obj=obj*float(len_obj)/len(list_tweets)
##        subj=subj*float(len_subj)/len(list_tweets)
        if obj==0 and subj==0:
            list_score_bayes_obj_train.append(0.5)
        else:
            list_score_bayes_obj_train.append(obj/float(subj+obj))

    words_bayes_obj_train = list_score_bayes_obj_train[:]



    print "check_4A: Feature Extraction 1/4"

#################### Objectivity Feature Extraction => TEST DATA



    for index in range(0,len(list_tweets_test)):

        num_exclamations_test.append(string.count(list_tweets_test[index],"!"))   ## exclamation marks

        if string.find(list_tweets_test[index],"http://") == -1:                  ## url's
            presence_url_test.append("no") 
        else:
            presence_url_test.append("yes")



    for tweet in list_tweets_test:                                               ## for emoticons

        if re.search(pos_emoticon_re,tweet) or re.search(neg_emoticon_re,tweet):
            presence_emoticon_test.append("yes")
        else:
            presence_emoticon_test.append("no")


            

    pos_list = [] ## a list containing pos tagging info in form of dictionary for each tweet
    
    for tweet in list_tweets_3_test:                                                 ## POS tagging
        pos=nltk.pos_tag(tweet)
        pos_dict={}
        for tag in pos:
            if tag[1] in pos_dict:
                pos_dict[tag[1]]+=1
            else:
                pos_dict[tag[1]]=1
        pos_list.append(pos_dict)


    for pos in pos_list:
        pos_prp_combined_test.append(pos.get("PRP",0) + pos.get("PRP$",0))




    for tweet in list_tweets_4_test:         ## MPQA word lexicon
        score=0
        for word in tweet:
            if word in dict_mpqa.keys():
                score += math.fabs(dict_mpqa[word])
        words_mpqa_test.append(score)




    for i in range(0,len(list_tweets_5_test)):       ## Naive bayes WORD MODELLING
        tweet=list_tweets_5_test[i]
        obj=1.0
        subj=1.0
        for word in tweet:
            if word in dict_final_bayes_obj.keys():
                obj=float(obj)*dict_final_bayes_obj[word][0]
                subj=float(subj)*dict_final_bayes_obj[word][1]
##        obj=obj*float(len_obj)/len(list_tweets)
##        subj=subj*float(len_subj)/len(list_tweets)
        if obj==0 and subj==0:
            list_score_bayes_obj_test.append(0.5)
        else:
            list_score_bayes_obj_test.append(obj/float(subj+obj))

    words_bayes_obj_test = list_score_bayes_obj_test[:]




    print "check_4B: Feature Extraction 2/4"

#####################  Subjectivity Feature Extraction => TRAINING DATA



    for tweet in list_tweets_training:                                               ## for emoticon score
        num_emot_pos = len(re.findall(pos_emoticon_re,tweet))
        num_emot_neg = len(re.findall(neg_emoticon_re,tweet))
        score_emoticon_train.append(num_emot_pos - num_emot_neg)
        num_emoticon_pos_train.append(num_emot_pos)
        num_emoticon_neg_train.append(num_emot_neg)





    for tweet in list_tweets_4_training:                 ## MPQA word lexicon score
        pos_mpqa=0
        neg_mpqa=0
        for word in tweet:
            if word in dict_mpqa.keys():
                if dict_mpqa[word]>0:
                    pos_mpqa+=dict_mpqa[word]
                elif dict_mpqa[word]<0:
                    neg_mpqa+=dict_mpqa[word]
        score_mpqa_all_train.append(float(pos_mpqa+neg_mpqa))
        score_mpqa_neg_train.append(float(neg_mpqa))





    for i in range(0,len(list_tweets_5_training)):       ## for list_score_bayes
        tweet=list_tweets_5_training[i]                   
        pos=1.0
        neg=1.0
        for word in tweet:
            if word in dict_final_bayes_subj.keys():
                pos=float(pos)*dict_final_bayes_subj[word][0]
                neg=float(neg)*dict_final_bayes_subj[word][1]
##        pos=pos*float(len_pos)/8963
##        neg=neg*float(len_neg)/8963
        if pos==0 and neg==0:
            list_score_bayes_subj_train.append(0.5)
        else:
            list_score_bayes_subj_train.append(pos/float(pos+neg))

    words_bayes_subj_train = list_score_bayes_subj_train[:]



    print "check_4C: Feature Extraction 3/4"

#####################  Subjectivity Feature Extraction => TEST DATA



    for tweet in list_tweets_test:                                               ## for emoticon score
        num_emot_pos = len(re.findall(pos_emoticon_re,tweet))
        num_emot_neg = len(re.findall(neg_emoticon_re,tweet))
        score_emoticon_test.append(num_emot_pos - num_emot_neg)
        num_emoticon_pos_test.append(num_emot_pos)
        num_emoticon_neg_test.append(num_emot_neg)





    for tweet in list_tweets_4_test:                 ## MPQA word lexicon score
        pos_mpqa=0
        neg_mpqa=0
        for word in tweet:
            if word in dict_mpqa.keys():
                if dict_mpqa[word]>0:
                    pos_mpqa+=dict_mpqa[word]
                elif dict_mpqa[word]<0:
                    neg_mpqa+=dict_mpqa[word]
        score_mpqa_all_test.append(float(pos_mpqa+neg_mpqa))
        score_mpqa_neg_test.append(float(neg_mpqa))





    for i in range(0,len(list_tweets_5_test)):       ## for list_score_bayes
        tweet=list_tweets_5_test[i]                   
        pos=1.0
        neg=1.0
        for word in tweet:
            if word in dict_final_bayes_subj.keys():
                pos=float(pos)*dict_final_bayes_subj[word][0]
                neg=float(neg)*dict_final_bayes_subj[word][1]
##        pos=pos*float(len_pos)/8963
##        neg=neg*float(len_neg)/8963
        if pos==0 and neg==0:
            list_score_bayes_subj_test.append(0.5)
        else:
            list_score_bayes_subj_test.append(pos/float(pos+neg))

    words_bayes_subj_test = list_score_bayes_subj_test[:]


    
    
    print "check_4: Feature Extraction"


####################################################################################  LIKELIHOOD CALCULATOR (NAIVE BAYES TRAINING)

    len_obj_train = 0
    len_subj_train = 0
    len_pos_train = 0
    len_neg_train = 0
    
    for label in training_labels:
        if label=="0":
            len_obj_train+=1
        elif label=="1":
            len_pos_train+=1
        elif label=="-1":
            len_neg_train+=1
    len_subj_train = len_pos_train + len_neg_train


    likelihood_obj=[]
    likelihood_subj=[]
    
 

#######################  Objectivity Features Likelihood


    num_exclamations_likelihood = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    presence_url_likelihood = [[0,0],[0,0]]
    presence_emoticon_likelihood = [[0,0],[0,0]]

    pos_prp_combined_likelihood = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

    words_mpqa_likelihood = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]


#########


    for i in range(0,len(presence_url_train)):
        feature = presence_url_train[i]
        if feature == "yes" and training_labels[i] == "0":
            presence_url_likelihood[0][0]+=1
        elif feature == "yes" and training_labels[i] != "0":
            presence_url_likelihood[0][1]+=1
        elif feature == "no" and training_labels[i] == "0":
            presence_url_likelihood[1][0]+=1
        elif feature == "no" and training_labels[i] != "0":
            presence_url_likelihood[1][1]+=1
    presence_url_likelihood[0][0] = float(presence_url_likelihood[0][0])/len_obj_train
    presence_url_likelihood[0][1] = float(presence_url_likelihood[0][1])/len_subj_train
    presence_url_likelihood[1][0] = float(presence_url_likelihood[1][0])/len_obj_train
    presence_url_likelihood[1][1] = float(presence_url_likelihood[1][1])/len_subj_train


    for i in range(0,len(presence_emoticon_train)):
        feature = presence_emoticon_train[i]
        if feature == "yes" and training_labels[i] == "0":
            presence_emoticon_likelihood[0][0]+=1
        elif feature == "yes" and training_labels[i] != "0":
            presence_emoticon_likelihood[0][1]+=1
        elif feature == "no" and training_labels[i] == "0":
            presence_emoticon_likelihood[1][0]+=1
        elif feature == "no" and training_labels[i] != "0":
            presence_emoticon_likelihood[1][1]+=1
    presence_emoticon_likelihood[0][0] = float(presence_emoticon_likelihood[0][0])/len_obj_train
    presence_emoticon_likelihood[0][1] = float(presence_emoticon_likelihood[0][1])/len_subj_train
    presence_emoticon_likelihood[1][0] = float(presence_emoticon_likelihood[1][0])/len_obj_train
    presence_emoticon_likelihood[1][1] = float(presence_emoticon_likelihood[1][1])/len_subj_train


    for i in range(0,len(pos_prp_combined_train)):
        feature = pos_prp_combined_train[i]
        if feature == 0 and training_labels[i] == "0":
            pos_prp_combined_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] != "0":
            pos_prp_combined_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "0":
            pos_prp_combined_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] != "0":
            pos_prp_combined_likelihood[1][1]+=1
        elif feature == 2 and training_labels[i] == "0":
            pos_prp_combined_likelihood[2][0]+=1
        elif feature == 2 and training_labels[i] != "0":
            pos_prp_combined_likelihood[2][1]+=1
        elif feature == 3 and training_labels[i] == "0":
            pos_prp_combined_likelihood[3][0]+=1
        elif feature == 3 and training_labels[i] != "0":
            pos_prp_combined_likelihood[3][1]+=1
        elif feature == 4 and training_labels[i] == "0":
            pos_prp_combined_likelihood[4][0]+=1
        elif feature == 4 and training_labels[i] != "0":
            pos_prp_combined_likelihood[4][1]+=1
        elif feature == 5 and training_labels[i] == "0":
            pos_prp_combined_likelihood[5][0]+=1
        elif feature == 5 and training_labels[i] != "0":
            pos_prp_combined_likelihood[5][1]+=1
        elif feature == 6 and training_labels[i] == "0":
            pos_prp_combined_likelihood[6][0]+=1
        elif feature == 6 and training_labels[i] != "0":
            pos_prp_combined_likelihood[6][1]+=1
        elif feature == 7 and training_labels[i]== "0":
            pos_prp_combined_likelihood[7][0]+=1
        elif feature == 7 and training_labels[i] != "0":
            pos_prp_combined_likelihood[7][1]+=1
        elif feature >= 8 and training_labels[i] == "0":
            pos_prp_combined_likelihood[8][0]+=1
        elif feature >= 8 and training_labels[i] != "0":
            pos_prp_combined_likelihood[8][1]+=1
    pos_prp_combined_likelihood[0][0] = float(pos_prp_combined_likelihood[0][0]+1)/(len_obj_train+9)   ## Laplace smoothing applied
    pos_prp_combined_likelihood[0][1] = float(pos_prp_combined_likelihood[0][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[1][0] = float(pos_prp_combined_likelihood[1][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[1][1] = float(pos_prp_combined_likelihood[1][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[2][0] = float(pos_prp_combined_likelihood[2][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[2][1] = float(pos_prp_combined_likelihood[2][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[3][0] = float(pos_prp_combined_likelihood[3][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[3][1] = float(pos_prp_combined_likelihood[3][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[4][0] = float(pos_prp_combined_likelihood[4][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[4][1] = float(pos_prp_combined_likelihood[4][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[5][0] = float(pos_prp_combined_likelihood[5][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[5][1] = float(pos_prp_combined_likelihood[5][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[6][0] = float(pos_prp_combined_likelihood[6][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[6][1] = float(pos_prp_combined_likelihood[6][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[7][0] = float(pos_prp_combined_likelihood[7][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[7][1] = float(pos_prp_combined_likelihood[7][1]+1)/(len_subj_train+9)
    pos_prp_combined_likelihood[8][0] = float(pos_prp_combined_likelihood[8][0]+1)/(len_obj_train+9)
    pos_prp_combined_likelihood[8][1] = float(pos_prp_combined_likelihood[8][1]+1)/(len_subj_train+9)


    for i in range(0,len(words_mpqa_train)):
        feature = words_mpqa_train[i]
        if feature == 0 and training_labels[i] == "0":
            words_mpqa_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] != "0":
            words_mpqa_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "0":
            words_mpqa_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] != "0":
            words_mpqa_likelihood[1][1]+=1
        elif feature == 2 and training_labels[i] == "0":
            words_mpqa_likelihood[2][0]+=1
        elif feature == 2 and training_labels[i] != "0":
            words_mpqa_likelihood[2][1]+=1
        elif (feature == 3 or feature == 4) and (training_labels[i] == "0"):
            words_mpqa_likelihood[3][0]+=1
        elif (feature == 3 or feature == 4) and (training_labels[i] != "0"):
            words_mpqa_likelihood[3][1]+=1
        elif (feature == 5 or feature == 6) and (training_labels[i] == "0"):
            words_mpqa_likelihood[4][0]+=1
        elif (feature == 5 or feature == 6) and (training_labels[i] != "0"):
            words_mpqa_likelihood[4][1]+=1
        elif feature == 7 and training_labels[i] == "0":
            words_mpqa_likelihood[5][0]+=1
        elif feature == 7 and training_labels[i] != "0":
            words_mpqa_likelihood[5][1]+=1
        elif (feature == 8 or feature == 9) and (training_labels[i] == "0"):
            words_mpqa_likelihood[6][0]+=1
        elif (feature == 8 or feature == 9) and (training_labels[i] != "0"):
            words_mpqa_likelihood[6][1]+=1
        elif (feature >= 10 and feature <= 12) and (training_labels[i] == "0"):
            words_mpqa_likelihood[7][0]+=1
        elif (feature >= 10 and feature <= 12) and (training_labels[i] != "0"):
            words_mpqa_likelihood[7][1]+=1
        elif (feature >= 13 and feature <= 16) and (training_labels[i] == "0"):
            words_mpqa_likelihood[8][0]+=1
        elif (feature >= 13 and feature <= 16) and (training_labels[i] != "0"):
            words_mpqa_likelihood[8][1]+=1
        elif feature >= 17 and training_labels[i] == "0":
            words_mpqa_likelihood[9][0]+=1
        elif feature >= 17 and training_labels[i] != "0":
            words_mpqa_likelihood[9][1]+=1
    words_mpqa_likelihood[0][0] = float(words_mpqa_likelihood[0][0]+1)/(len_obj_train+10)      ## Laplace smoothing applied
    words_mpqa_likelihood[0][1] = float(words_mpqa_likelihood[0][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[1][0] = float(words_mpqa_likelihood[1][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[1][1] = float(words_mpqa_likelihood[1][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[2][0] = float(words_mpqa_likelihood[2][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[2][1] = float(words_mpqa_likelihood[2][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[3][0] = float(words_mpqa_likelihood[3][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[3][1] = float(words_mpqa_likelihood[3][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[4][0] = float(words_mpqa_likelihood[4][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[4][1] = float(words_mpqa_likelihood[4][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[5][0] = float(words_mpqa_likelihood[5][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[5][1] = float(words_mpqa_likelihood[5][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[6][0] = float(words_mpqa_likelihood[6][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[6][1] = float(words_mpqa_likelihood[6][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[7][0] = float(words_mpqa_likelihood[7][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[7][1] = float(words_mpqa_likelihood[7][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[8][0] = float(words_mpqa_likelihood[8][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[8][1] = float(words_mpqa_likelihood[8][1]+1)/(len_subj_train+10)
    words_mpqa_likelihood[9][0] = float(words_mpqa_likelihood[9][0]+1)/(len_obj_train+10)
    words_mpqa_likelihood[9][1] = float(words_mpqa_likelihood[9][1]+1)/(len_subj_train+10)


    for i in range(0,len(num_exclamations_train)):
        feature = num_exclamations_train[i]
        if feature == 0 and training_labels[i] == "0":
            num_exclamations_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] != "0":
            num_exclamations_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "0":
            num_exclamations_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] != "0":
            num_exclamations_likelihood[1][1]+=1
        elif feature == 2 and training_labels[i] == "0":
            num_exclamations_likelihood[2][0]+=1
        elif feature == 2 and training_labels[i] != "0":
            num_exclamations_likelihood[2][1]+=1
        elif feature == 3 and training_labels[i] == "0":
            num_exclamations_likelihood[3][0]+=1
        elif feature == 3 and training_labels[i] != "0":
            num_exclamations_likelihood[3][1]+=1
        elif (feature == 4 or feature == 5) and (training_labels[i] == "0"):
            num_exclamations_likelihood[4][0]+=1
        elif (feature == 4 or feature == 5) and (training_labels[i] != "0"):
            num_exclamations_likelihood[4][1]+=1
        elif (feature >= 6 and feature <= 9) and (training_labels[i] == "0"):
            num_exclamations_likelihood[5][0]+=1
        elif (feature >= 6 and feature <= 9) and (training_labels[i] != "0"):
            num_exclamations_likelihood[5][1]+=1
        elif feature >= 10 and (training_labels[i] == "0"):
            num_exclamations_likelihood[6][0]+=1
        elif feature >= 10 and (training_labels[i] != "0"):
            num_exclamations_likelihood[6][1]+=1
    num_exclamations_likelihood[0][0] = float(num_exclamations_likelihood[0][0])/(len_obj_train)   
    num_exclamations_likelihood[0][1] = float(num_exclamations_likelihood[0][1])/(len_subj_train)
    num_exclamations_likelihood[1][0] = float(num_exclamations_likelihood[1][0])/(len_obj_train)
    num_exclamations_likelihood[1][1] = float(num_exclamations_likelihood[1][1])/(len_subj_train)
    num_exclamations_likelihood[2][0] = float(num_exclamations_likelihood[2][0])/(len_obj_train)
    num_exclamations_likelihood[2][1] = float(num_exclamations_likelihood[2][1])/(len_subj_train)
    num_exclamations_likelihood[3][0] = float(num_exclamations_likelihood[3][0])/(len_obj_train)
    num_exclamations_likelihood[3][1] = float(num_exclamations_likelihood[3][1])/(len_subj_train)
    num_exclamations_likelihood[4][0] = float(num_exclamations_likelihood[4][0])/(len_obj_train)
    num_exclamations_likelihood[4][1] = float(num_exclamations_likelihood[4][1])/(len_subj_train)
    num_exclamations_likelihood[5][0] = float(num_exclamations_likelihood[5][0])/(len_obj_train)
    num_exclamations_likelihood[5][1] = float(num_exclamations_likelihood[5][1])/(len_subj_train)
    num_exclamations_likelihood[6][0] = float(num_exclamations_likelihood[6][0])/(len_obj_train)
    num_exclamations_likelihood[6][1] = float(num_exclamations_likelihood[6][1])/(len_subj_train)
    


    likelihood_obj = [presence_url_likelihood, presence_emoticon_likelihood, words_mpqa_likelihood, num_exclamations_likelihood, pos_prp_combined_likelihood]
    


#######################  Subjectivity Features Likelihood


    score_emoticon_likelihood = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    score_mpqa_all_likelihood = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    
    num_emoticon_pos_likelihood = [[0,0],[0,0],[0,0]]
    num_emoticon_neg_likelihood = [[0,0],[0,0],[0,0]]
    score_mpqa_neg_likelihood = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
 


############



    for i in range(0,len(score_emoticon_train)):
        feature = score_emoticon_train[i]
        if feature == 0 and training_labels[i] == "1":
            score_emoticon_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] == "-1":
            score_emoticon_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "1":
            score_emoticon_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] == "-1":
            score_emoticon_likelihood[1][1]+=1
        elif feature >= 2 and training_labels[i] == "1":
            score_emoticon_likelihood[2][0]+=1
        elif feature >= 2 and training_labels[i] == "-1":
            score_emoticon_likelihood[2][1]+=1
        elif feature == -1 and training_labels[i] == "1":
            score_emoticon_likelihood[3][0]+=1
        elif feature == -1 and training_labels[i] == "-1":
            score_emoticon_likelihood[3][1]+=1
        elif feature <= -2 and (training_labels[i] == "1"):
            score_emoticon_likelihood[4][0]+=1
        elif feature <= -2 and (training_labels[i] == "-1"):
            score_emoticon_likelihood[4][1]+=1
    score_emoticon_likelihood[0][0] = float(score_emoticon_likelihood[0][0]+1)/(len_pos_train+5)   
    score_emoticon_likelihood[0][1] = float(score_emoticon_likelihood[0][1]+1)/(len_neg_train+5)
    score_emoticon_likelihood[1][0] = float(score_emoticon_likelihood[1][0]+1)/(len_pos_train+5)
    score_emoticon_likelihood[1][1] = float(score_emoticon_likelihood[1][1]+1)/(len_neg_train+5)
    score_emoticon_likelihood[2][0] = float(score_emoticon_likelihood[2][0]+1)/(len_pos_train+5)
    score_emoticon_likelihood[2][1] = float(score_emoticon_likelihood[2][1]+1)/(len_neg_train+5)
    score_emoticon_likelihood[3][0] = float(score_emoticon_likelihood[3][0]+1)/(len_pos_train+5)
    score_emoticon_likelihood[3][1] = float(score_emoticon_likelihood[3][1]+1)/(len_neg_train+5)
    score_emoticon_likelihood[4][0] = float(score_emoticon_likelihood[4][0]+1)/(len_pos_train+5)
    score_emoticon_likelihood[4][1] = float(score_emoticon_likelihood[4][1]+1)/(len_neg_train+5)


    for i in range(0,len(score_mpqa_all_train)):
        feature = score_mpqa_all_train[i]
        if feature == 0 and training_labels[i] == "1":
            score_mpqa_all_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] == "-1":
            score_mpqa_all_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "1":
            score_mpqa_all_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] == "-1":
            score_mpqa_all_likelihood[1][1]+=1
        elif (feature == 2 or feature == 3) and training_labels[i] == "1":
            score_mpqa_all_likelihood[2][0]+=1
        elif (feature == 2 or feature == 3) and training_labels[i] == "-1":
            score_mpqa_all_likelihood[2][1]+=1
        elif (feature == 4 or feature == 5) and training_labels[i] == "1":
            score_mpqa_all_likelihood[3][0]+=1
        elif (feature == 4 or feature == 5) and training_labels[i] == "-1":
            score_mpqa_all_likelihood[3][1]+=1
        elif (feature >= 6 and feature <= 9) and (training_labels[i] == "1"):
            score_mpqa_all_likelihood[4][0]+=1
        elif (feature >= 6 and feature <= 9) and (training_labels[i] == "-1"):
            score_mpqa_all_likelihood[4][1]+=1
        elif feature >= 10 and training_labels[i] == "1":
            score_mpqa_all_likelihood[5][0]+=1
        elif feature >= 10 and training_labels[i] == "-1":
            score_mpqa_all_likelihood[5][1]+=1
        elif feature == -1 and training_labels[i] == "1":
            score_mpqa_all_likelihood[6][0]+=1
        elif feature == -1 and training_labels[i] == "-1":
            score_mpqa_all_likelihood[6][1]+=1
        elif feature == -2 and training_labels[i] == "1":
            score_mpqa_all_likelihood[7][0]+=1
        elif feature == -2 and training_labels[i] == "-1":
            score_mpqa_all_likelihood[7][1]+=1
        elif (feature == -3 or feature == -4) and (training_labels[i] == "1"):
            score_mpqa_all_likelihood[8][0]+=1
        elif (feature == -3 or feature == -4) and (training_labels[i] == "-1"):
            score_mpqa_all_likelihood[8][1]+=1
        elif feature <= -5 and (training_labels[i] == "1"):
            score_mpqa_all_likelihood[9][0]+=1
        elif feature <= -5 and (training_labels[i] == "-1"):
            score_mpqa_all_likelihood[9][1]+=1
    score_mpqa_all_likelihood[0][0] = float(score_mpqa_all_likelihood[0][0]+1)/(len_pos_train+10)   
    score_mpqa_all_likelihood[0][1] = float(score_mpqa_all_likelihood[0][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[1][0] = float(score_mpqa_all_likelihood[1][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[1][1] = float(score_mpqa_all_likelihood[1][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[2][0] = float(score_mpqa_all_likelihood[2][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[2][1] = float(score_mpqa_all_likelihood[2][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[3][0] = float(score_mpqa_all_likelihood[3][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[3][1] = float(score_mpqa_all_likelihood[3][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[4][0] = float(score_mpqa_all_likelihood[4][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[4][1] = float(score_mpqa_all_likelihood[4][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[5][0] = float(score_mpqa_all_likelihood[5][0]+1)/(len_pos_train+10)   
    score_mpqa_all_likelihood[5][1] = float(score_mpqa_all_likelihood[5][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[6][0] = float(score_mpqa_all_likelihood[6][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[6][1] = float(score_mpqa_all_likelihood[6][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[7][0] = float(score_mpqa_all_likelihood[7][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[7][1] = float(score_mpqa_all_likelihood[7][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[8][0] = float(score_mpqa_all_likelihood[8][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[8][1] = float(score_mpqa_all_likelihood[8][1]+1)/(len_neg_train+10)
    score_mpqa_all_likelihood[9][0] = float(score_mpqa_all_likelihood[9][0]+1)/(len_pos_train+10)
    score_mpqa_all_likelihood[9][1] = float(score_mpqa_all_likelihood[9][1]+1)/(len_neg_train+10)


    for i in range(0,len(num_emoticon_pos_train)):
        feature = num_emoticon_pos_train[i]
        if feature == 0 and training_labels[i] == "1":
            num_emoticon_pos_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] == "-1":
            num_emoticon_pos_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "1":
            num_emoticon_pos_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] == "-1":
            num_emoticon_pos_likelihood[1][1]+=1
        elif feature >= 2 and training_labels[i] == "1":
            num_emoticon_pos_likelihood[2][0]+=1
        elif feature >= 2 and training_labels[i] == "-1":
            num_emoticon_pos_likelihood[2][1]+=1
    num_emoticon_pos_likelihood[0][0] = float(num_emoticon_pos_likelihood[0][0]+1)/(len_pos_train+3)   
    num_emoticon_pos_likelihood[0][1] = float(num_emoticon_pos_likelihood[0][1]+1)/(len_neg_train+3)
    num_emoticon_pos_likelihood[1][0] = float(num_emoticon_pos_likelihood[1][0]+1)/(len_pos_train+3)
    num_emoticon_pos_likelihood[1][1] = float(num_emoticon_pos_likelihood[1][1]+1)/(len_neg_train+3)
    num_emoticon_pos_likelihood[2][0] = float(num_emoticon_pos_likelihood[2][0]+1)/(len_pos_train+3)
    num_emoticon_pos_likelihood[2][1] = float(num_emoticon_pos_likelihood[2][1]+1)/(len_neg_train+3)


    for i in range(0,len(num_emoticon_neg_train)):
        feature = num_emoticon_neg_train[i]
        if feature == 0 and training_labels[i] == "1":
            num_emoticon_neg_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] == "-1":
            num_emoticon_neg_likelihood[0][1]+=1
        elif feature == 1 and training_labels[i] == "1":
            num_emoticon_neg_likelihood[1][0]+=1
        elif feature == 1 and training_labels[i] == "-1":
            num_emoticon_neg_likelihood[1][1]+=1
        elif feature >= 2 and training_labels[i] == "1":
            num_emoticon_neg_likelihood[2][0]+=1
        elif feature >= 2 and training_labels[i] == "-1":
            num_emoticon_neg_likelihood[2][1]+=1
    num_emoticon_neg_likelihood[0][0] = float(num_emoticon_neg_likelihood[0][0])/(len_pos_train)   
    num_emoticon_neg_likelihood[0][1] = float(num_emoticon_neg_likelihood[0][1])/(len_neg_train)
    num_emoticon_neg_likelihood[1][0] = float(num_emoticon_neg_likelihood[1][0])/(len_pos_train)
    num_emoticon_neg_likelihood[1][1] = float(num_emoticon_neg_likelihood[1][1])/(len_neg_train)
    num_emoticon_neg_likelihood[2][0] = float(num_emoticon_neg_likelihood[2][0])/(len_pos_train)
    num_emoticon_neg_likelihood[2][1] = float(num_emoticon_neg_likelihood[2][1])/(len_neg_train)


    for i in range(0,len(score_mpqa_neg_train)):
        feature = score_mpqa_neg_train[i]
        if feature == 0 and training_labels[i] == "1":
            score_mpqa_neg_likelihood[0][0]+=1
        elif feature == 0 and training_labels[i] == "-1":
            score_mpqa_neg_likelihood[0][1]+=1
        elif feature == -1 and training_labels[i] == "1":
            score_mpqa_neg_likelihood[1][0]+=1
        elif feature == -1 and training_labels[i] == "-1":
            score_mpqa_neg_likelihood[1][1]+=1
        elif feature == -2 and training_labels[i] == "1":
            score_mpqa_neg_likelihood[2][0]+=1
        elif feature == -2 and training_labels[i] == "-1":
            score_mpqa_neg_likelihood[2][1]+=1
        elif (feature == -3 or feature == -4) and training_labels[i] == "1":
            score_mpqa_neg_likelihood[3][0]+=1
        elif (feature == -3 or feature == -4) and training_labels[i] == "-1":
            score_mpqa_neg_likelihood[3][1]+=1
        elif (feature == -5 or feature == -6) and training_labels[i] == "1":
            score_mpqa_neg_likelihood[4][0]+=1
        elif (feature == -5 or feature == -6) and training_labels[i] == "-1":
            score_mpqa_neg_likelihood[4][1]+=1
        elif feature <= -7 and training_labels[i] == "1":
            score_mpqa_neg_likelihood[5][0]+=1
        elif feature <= -7 and training_labels[i] == "-1":
            score_mpqa_neg_likelihood[5][1]+=1
    score_mpqa_neg_likelihood[0][0] = float(score_mpqa_neg_likelihood[0][0]+1)/(len_pos_train+6)   
    score_mpqa_neg_likelihood[0][1] = float(score_mpqa_neg_likelihood[0][1]+1)/(len_neg_train+6)
    score_mpqa_neg_likelihood[1][0] = float(score_mpqa_neg_likelihood[1][0]+1)/(len_pos_train+6)
    score_mpqa_neg_likelihood[1][1] = float(score_mpqa_neg_likelihood[1][1]+1)/(len_neg_train+6)
    score_mpqa_neg_likelihood[2][0] = float(score_mpqa_neg_likelihood[2][0]+1)/(len_pos_train+6)
    score_mpqa_neg_likelihood[2][1] = float(score_mpqa_neg_likelihood[2][1]+1)/(len_neg_train+6)
    score_mpqa_neg_likelihood[3][0] = float(score_mpqa_neg_likelihood[3][0]+1)/(len_pos_train+6)
    score_mpqa_neg_likelihood[3][1] = float(score_mpqa_neg_likelihood[3][1]+1)/(len_neg_train+6)
    score_mpqa_neg_likelihood[4][0] = float(score_mpqa_neg_likelihood[4][0]+1)/(len_pos_train+6)
    score_mpqa_neg_likelihood[4][1] = float(score_mpqa_neg_likelihood[4][1]+1)/(len_neg_train+6)
    score_mpqa_neg_likelihood[5][0] = float(score_mpqa_neg_likelihood[5][0]+1)/(len_pos_train+6)
    score_mpqa_neg_likelihood[5][1] = float(score_mpqa_neg_likelihood[5][1]+1)/(len_neg_train+6)



    likelihood_subj =  [score_emoticon_likelihood,score_mpqa_all_likelihood,num_emoticon_pos_likelihood,num_emoticon_neg_likelihood,score_mpqa_neg_likelihood]


    print "check_5: Likelihood calculation"

    
####################################################################################  CLASSIFICATION NAIVE BAYES




################### Objectivity Classification of Training Data



    feature_list_obj_train=[]
    obj_score_list_train = []

    for index in range(0,len(list_tweets_training)):
        feat_obj = [0]*6
        feat_subj = [0]*6

        if presence_url_train[index] == "yes":
            feat_obj[0] = presence_url_likelihood[0][0]
            feat_subj[0] = presence_url_likelihood[0][1]
        else:
            feat_obj[0] = presence_url_likelihood[1][0]
            feat_subj[0] = presence_url_likelihood[1][1]


        if presence_emoticon_train[index] == "yes":
            feat_obj[1] = presence_emoticon_likelihood[0][0]
            feat_subj[1] = presence_emoticon_likelihood[0][1]
        else:
            feat_obj[1] = presence_emoticon_likelihood[1][0]
            feat_subj[1] = presence_emoticon_likelihood[1][1]


        if words_mpqa_train[index] == 0:
            feat_obj[2] = words_mpqa_likelihood[0][0]
            feat_subj[2] = words_mpqa_likelihood[0][1]
        elif words_mpqa_train[index] == 1:
            feat_obj[2] = words_mpqa_likelihood[1][0]
            feat_subj[2] = words_mpqa_likelihood[1][1]
        elif words_mpqa_train[index] == 2:
            feat_obj[2] = words_mpqa_likelihood[2][0]
            feat_subj[2] = words_mpqa_likelihood[2][1]
        elif words_mpqa_train[index] == 3 or words_mpqa_train[index] == 4:
            feat_obj[2] = words_mpqa_likelihood[3][0]
            feat_subj[2] = words_mpqa_likelihood[3][1]
        elif words_mpqa_train[index] == 5 or words_mpqa_train[index] == 6:
            feat_obj[2] = words_mpqa_likelihood[4][0]
            feat_subj[2] = words_mpqa_likelihood[4][1]
        elif words_mpqa_train[index] == 7:
            feat_obj[2] = words_mpqa_likelihood[5][0]
            feat_subj[2] = words_mpqa_likelihood[5][1]
        elif words_mpqa_train[index] == 8 or words_mpqa_train[index] == 9:
            feat_obj[2] = words_mpqa_likelihood[6][0]
            feat_subj[2] = words_mpqa_likelihood[6][1]
        elif words_mpqa_train[index]>=10 and words_mpqa_train[index]<=12:
            feat_obj[2] = words_mpqa_likelihood[7][0]
            feat_subj[2] = words_mpqa_likelihood[7][1]
        elif words_mpqa_train[index]>=13 and words_mpqa_train[index]<=16:
            feat_obj[2] = words_mpqa_likelihood[8][0]
            feat_subj[2] = words_mpqa_likelihood[8][1]
        else:
            feat_obj[2] = words_mpqa_likelihood[9][0]
            feat_subj[2] = words_mpqa_likelihood[9][1]
        

        feat_obj[3] = words_bayes_obj_train[index]
        feat_subj[3] = 1-(words_bayes_obj_train[index])


        if num_exclamations_train[index] == 0:
            feat_obj[4] = num_exclamations_likelihood[0][0]
            feat_subj[4] = num_exclamations_likelihood[0][1]
        elif num_exclamations_train[index] == 1:
            feat_obj[4] = num_exclamations_likelihood[1][0]
            feat_subj[4] = num_exclamations_likelihood[1][1]
        elif num_exclamations_train[index] == 2:
            feat_obj[4] = num_exclamations_likelihood[2][0]
            feat_subj[4] = num_exclamations_likelihood[2][1]
        elif num_exclamations_train[index] == 3:
            feat_obj[4] = num_exclamations_likelihood[3][0]
            feat_subj[4] = num_exclamations_likelihood[3][1]
        elif num_exclamations_train[index]==4 or num_exclamations_train[index]==5:
            feat_obj[4] = num_exclamations_likelihood[4][0]
            feat_subj[4] = num_exclamations_likelihood[4][1]
        elif num_exclamations_train[index]>=6 and num_exclamations_train[index]<=9 :
            feat_obj[4] = num_exclamations_likelihood[5][0]
            feat_subj[4] = num_exclamations_likelihood[5][1]
        else:
            feat_obj[4] = num_exclamations_likelihood[6][0]
            feat_subj[4] = num_exclamations_likelihood[6][1]


        if pos_prp_combined_train[index] == 0:
            feat_obj[5] = pos_prp_combined_likelihood[0][0]
            feat_subj[5] = pos_prp_combined_likelihood[0][1]
        elif pos_prp_combined_train[index] == 1:
            feat_obj[5] = pos_prp_combined_likelihood[1][0]
            feat_subj[5] = pos_prp_combined_likelihood[1][1]
        elif pos_prp_combined_train[index] == 2:
            feat_obj[5] = pos_prp_combined_likelihood[2][0]
            feat_subj[5] = pos_prp_combined_likelihood[2][1]
        elif pos_prp_combined_train[index] == 3:
            feat_obj[5] = pos_prp_combined_likelihood[3][0]
            feat_subj[5] = pos_prp_combined_likelihood[3][1]
        elif pos_prp_combined_train[index] == 4:
            feat_obj[5] = pos_prp_combined_likelihood[4][0]
            feat_subj[5] = pos_prp_combined_likelihood[4][1]
        elif pos_prp_combined_train[index] == 5:
            feat_obj[5] = pos_prp_combined_likelihood[5][0]
            feat_subj[5] = pos_prp_combined_likelihood[5][1]
        elif pos_prp_combined_train[index] == 6:
            feat_obj[5] = pos_prp_combined_likelihood[6][0]
            feat_subj[5] = pos_prp_combined_likelihood[6][1]
        elif pos_prp_combined_train[index] == 7:
            feat_obj[5] = pos_prp_combined_likelihood[7][0]
            feat_subj[5] = pos_prp_combined_likelihood[7][1]
        else:
            feat_obj[5] = pos_prp_combined_likelihood[8][0]
            feat_subj[5] = pos_prp_combined_likelihood[8][1]
        

        score_obj = feat_obj[0]*feat_obj[1]*feat_obj[3]*feat_obj[5]
        score_subj = feat_subj[0]*feat_subj[1]*feat_subj[3]*feat_subj[5]
        final_score_obj_train = float(score_obj)/(score_obj + score_subj)

        obj_score_list_train.append("%.4f" % final_score_obj_train)

        feature_list_obj_train.append([[feat_obj[0],feat_subj[0]],[feat_obj[1],feat_subj[1]],[feat_obj[2],feat_subj[2]],[feat_obj[3],feat_subj[3]],[feat_obj[4],feat_subj[4]],[feat_obj[5],feat_subj[5]]])




################### Objectivity Classification of Test Data



    feature_list_obj_test=[]
    obj_score_list_test = []

    for index in range(0,len(list_tweets_test)):
        feat_obj = [0]*6
        feat_subj = [0]*6

        if presence_url_test[index] == "yes":
            feat_obj[0] = presence_url_likelihood[0][0]
            feat_subj[0] = presence_url_likelihood[0][1]
        else:
            feat_obj[0] = presence_url_likelihood[1][0]
            feat_subj[0] = presence_url_likelihood[1][1]


        if presence_emoticon_test[index] == "yes":
            feat_obj[1] = presence_emoticon_likelihood[0][0]
            feat_subj[1] = presence_emoticon_likelihood[0][1]
        else:
            feat_obj[1] = presence_emoticon_likelihood[1][0]
            feat_subj[1] = presence_emoticon_likelihood[1][1]


        if words_mpqa_test[index] == 0:
            feat_obj[2] = words_mpqa_likelihood[0][0]
            feat_subj[2] = words_mpqa_likelihood[0][1]
        elif words_mpqa_test[index] == 1:
            feat_obj[2] = words_mpqa_likelihood[1][0]
            feat_subj[2] = words_mpqa_likelihood[1][1]
        elif words_mpqa_test[index] == 2:
            feat_obj[2] = words_mpqa_likelihood[2][0]
            feat_subj[2] = words_mpqa_likelihood[2][1]
        elif words_mpqa_test[index] == 3 or words_mpqa_test[index] == 4:
            feat_obj[2] = words_mpqa_likelihood[3][0]
            feat_subj[2] = words_mpqa_likelihood[3][1]
        elif words_mpqa_test[index] == 5 or words_mpqa_test[index] == 6:
            feat_obj[2] = words_mpqa_likelihood[4][0]
            feat_subj[2] = words_mpqa_likelihood[4][1]
        elif words_mpqa_test[index] == 7:
            feat_obj[2] = words_mpqa_likelihood[5][0]
            feat_subj[2] = words_mpqa_likelihood[5][1]
        elif words_mpqa_test[index] == 8 or words_mpqa_test[index] == 9:
            feat_obj[2] = words_mpqa_likelihood[6][0]
            feat_subj[2] = words_mpqa_likelihood[6][1]
        elif words_mpqa_test[index]>=10 and words_mpqa_test[index]<=12:
            feat_obj[2] = words_mpqa_likelihood[7][0]
            feat_subj[2] = words_mpqa_likelihood[7][1]
        elif words_mpqa_test[index]>=13 and words_mpqa_test[index]<=16:
            feat_obj[2] = words_mpqa_likelihood[8][0]
            feat_subj[2] = words_mpqa_likelihood[8][1]
        else:
            feat_obj[2] = words_mpqa_likelihood[9][0]
            feat_subj[2] = words_mpqa_likelihood[9][1]
        

        feat_obj[3] = words_bayes_obj_test[index]
        feat_subj[3] = 1-(words_bayes_obj_test[index])


        if num_exclamations_test[index] == 0:
            feat_obj[4] = num_exclamations_likelihood[0][0]
            feat_subj[4] = num_exclamations_likelihood[0][1]
        elif num_exclamations_test[index] == 1:
            feat_obj[4] = num_exclamations_likelihood[1][0]
            feat_subj[4] = num_exclamations_likelihood[1][1]
        elif num_exclamations_test[index] == 2:
            feat_obj[4] = num_exclamations_likelihood[2][0]
            feat_subj[4] = num_exclamations_likelihood[2][1]
        elif num_exclamations_test[index] == 3:
            feat_obj[4] = num_exclamations_likelihood[3][0]
            feat_subj[4] = num_exclamations_likelihood[3][1]
        elif num_exclamations_test[index]==4 or num_exclamations_test[index]==5:
            feat_obj[4] = num_exclamations_likelihood[4][0]
            feat_subj[4] = num_exclamations_likelihood[4][1]
        elif num_exclamations_test[index]>=6 and num_exclamations_test[index]<=9 :
            feat_obj[4] = num_exclamations_likelihood[5][0]
            feat_subj[4] = num_exclamations_likelihood[5][1]
        else:
            feat_obj[4] = num_exclamations_likelihood[6][0]
            feat_subj[4] = num_exclamations_likelihood[6][1]


        if pos_prp_combined_test[index] == 0:
            feat_obj[5] = pos_prp_combined_likelihood[0][0]
            feat_subj[5] = pos_prp_combined_likelihood[0][1]
        elif pos_prp_combined_test[index] == 1:
            feat_obj[5] = pos_prp_combined_likelihood[1][0]
            feat_subj[5] = pos_prp_combined_likelihood[1][1]
        elif pos_prp_combined_test[index] == 2:
            feat_obj[5] = pos_prp_combined_likelihood[2][0]
            feat_subj[5] = pos_prp_combined_likelihood[2][1]
        elif pos_prp_combined_test[index] == 3:
            feat_obj[5] = pos_prp_combined_likelihood[3][0]
            feat_subj[5] = pos_prp_combined_likelihood[3][1]
        elif pos_prp_combined_test[index] == 4:
            feat_obj[5] = pos_prp_combined_likelihood[4][0]
            feat_subj[5] = pos_prp_combined_likelihood[4][1]
        elif pos_prp_combined_test[index] == 5:
            feat_obj[5] = pos_prp_combined_likelihood[5][0]
            feat_subj[5] = pos_prp_combined_likelihood[5][1]
        elif pos_prp_combined_test[index] == 6:
            feat_obj[5] = pos_prp_combined_likelihood[6][0]
            feat_subj[5] = pos_prp_combined_likelihood[6][1]
        elif pos_prp_combined_test[index] == 7:
            feat_obj[5] = pos_prp_combined_likelihood[7][0]
            feat_subj[5] = pos_prp_combined_likelihood[7][1]
        else:
            feat_obj[5] = pos_prp_combined_likelihood[8][0]
            feat_subj[5] = pos_prp_combined_likelihood[8][1]
        

        score_obj = feat_obj[0]*feat_obj[1]*feat_obj[3]*feat_obj[5]
        score_subj = feat_subj[0]*feat_subj[1]*feat_subj[3]*feat_subj[5]
        final_score_obj_test = float(score_obj)/(score_obj + score_subj)

        obj_score_list_test.append("%.4f" % final_score_obj_test)



################### Subjectivity Classification of Training Data


    feature_list_subj_train=[]
    pos_score_list_train = []

    for index in range(0,len(list_tweets_training)):
        feat_pos = [0]*6
        feat_neg = [0]*6



        if score_emoticon_train[index] == 0:
            feat_pos[0]=score_emoticon_likelihood[0][0]
            feat_neg[0]=score_emoticon_likelihood[0][1]
        elif score_emoticon_train[index] == 1:
            feat_pos[0]=score_emoticon_likelihood[1][0]
            feat_neg[0]=score_emoticon_likelihood[1][1]
        elif score_emoticon_train[index] >= 2:
            feat_pos[0]=score_emoticon_likelihood[2][0]
            feat_neg[0]=score_emoticon_likelihood[2][1]
        elif score_emoticon_train[index] == -1:
            feat_pos[0]=score_emoticon_likelihood[3][0]
            feat_neg[0]=score_emoticon_likelihood[3][1]
        else:
            feat_pos[0]=score_emoticon_likelihood[4][0]
            feat_neg[0]=score_emoticon_likelihood[4][1]



        if score_mpqa_all_train[index] == 0:
            feat_pos[1]=score_mpqa_all_likelihood[0][0]
            feat_neg[1]=score_mpqa_all_likelihood[0][1]
        elif score_mpqa_all_train[index] == 1:
            feat_pos[1]=score_mpqa_all_likelihood[1][0]
            feat_neg[1]=score_mpqa_all_likelihood[1][1]
        elif score_mpqa_all_train[index]==2 or score_mpqa_all_train[index]==3:
            feat_pos[1]=score_mpqa_all_likelihood[2][0]
            feat_neg[1]=score_mpqa_all_likelihood[2][1]
        elif score_mpqa_all_train[index]==4 or score_mpqa_all_train[index]==5:
            feat_pos[1]=score_mpqa_all_likelihood[3][0]
            feat_neg[1]=score_mpqa_all_likelihood[3][1]
        elif score_mpqa_all_train[index]>=6 or score_mpqa_all_train[index]<=9:
            feat_pos[1]=score_mpqa_all_likelihood[4][0]
            feat_neg[1]=score_mpqa_all_likelihood[4][1]
        elif score_mpqa_all_train[index]>=10:
            feat_pos[1]=score_mpqa_all_likelihood[5][0]
            feat_neg[1]=score_mpqa_all_likelihood[5][1]
        elif score_mpqa_all_train[index] == -1:
            feat_pos[1]=score_mpqa_all_likelihood[6][0]
            feat_neg[1]=score_mpqa_all_likelihood[6][1]
        elif score_mpqa_all_train[index] == -2:
            feat_pos[1]=score_mpqa_all_likelihood[7][0]
            feat_neg[1]=score_mpqa_all_likelihood[7][1]
        elif score_mpqa_all_train[index]==-3 or score_mpqa_all_train[index]==-4:
            feat_pos[1]=score_mpqa_all_likelihood[8][0]
            feat_neg[1]=score_mpqa_all_likelihood[8][1]
        else:
            feat_pos[1]=score_mpqa_all_likelihood[9][0]
            feat_neg[1]=score_mpqa_all_likelihood[9][1]



        feat_pos[2] = words_bayes_subj_train[index]
        feat_neg[2] = 1 - (words_bayes_subj_train[index])



        if num_emoticon_pos_train[index]==0:
            feat_pos[3]=num_emoticon_pos_likelihood[0][0]
            feat_neg[3]=num_emoticon_pos_likelihood[0][1]
        elif num_emoticon_pos_train[index]==1:
            feat_pos[3]=num_emoticon_pos_likelihood[1][0]
            feat_neg[3]=num_emoticon_pos_likelihood[1][1]
        else:
            feat_pos[3]=num_emoticon_pos_likelihood[2][0]
            feat_neg[3]=num_emoticon_pos_likelihood[2][1]



        if num_emoticon_neg_train[index]==0:
            feat_pos[4]=num_emoticon_neg_likelihood[0][0]
            feat_neg[4]=num_emoticon_neg_likelihood[0][1]
        elif num_emoticon_neg_train[index]==1:
            feat_pos[4]=num_emoticon_neg_likelihood[1][0]
            feat_neg[4]=num_emoticon_neg_likelihood[1][1]
        else:
            feat_pos[4]=num_emoticon_neg_likelihood[2][0]
            feat_neg[4]=num_emoticon_neg_likelihood[2][1]



        if score_mpqa_neg_train[index] == 0:
            feat_pos[5]=score_mpqa_neg_likelihood[0][0]
            feat_neg[5]=score_mpqa_neg_likelihood[0][1]
        elif score_mpqa_neg_train[index] == -1:
            feat_pos[5]=score_mpqa_neg_likelihood[1][0]
            feat_neg[5]=score_mpqa_neg_likelihood[1][1]
        elif score_mpqa_neg_train[index] == -2:
            feat_pos[5]=score_mpqa_neg_likelihood[2][0]
            feat_neg[5]=score_mpqa_neg_likelihood[2][1]
        elif score_mpqa_neg_train[index]==-3 or score_mpqa_neg_train[index]==-4:
            feat_pos[5]=score_mpqa_neg_likelihood[3][0]
            feat_neg[5]=score_mpqa_neg_likelihood[3][1]
        elif score_mpqa_neg_train[index]==-5 or score_mpqa_neg_train[index]==-6:
            feat_pos[5]=score_mpqa_neg_likelihood[4][0]
            feat_neg[5]=score_mpqa_neg_likelihood[4][1]
        else:
            feat_pos[5]=score_mpqa_neg_likelihood[5][0]
            feat_neg[5]=score_mpqa_neg_likelihood[5][1]



        score_pos = feat_pos[0]*feat_pos[2]*feat_pos[3]*feat_pos[4]
        score_neg = feat_neg[0]*feat_neg[2]*feat_neg[3]*feat_neg[4]
        final_score_pos = float(score_pos)/(score_pos + score_neg)

        pos_score_list_train.append("%.4f" % final_score_pos)

        feature_list_subj_train.append([[feat_pos[0],feat_neg[0]],[feat_pos[1],feat_neg[1]],[feat_pos[2],feat_neg[2]],[feat_pos[3],feat_neg[3]],[feat_pos[4],feat_neg[4]],[feat_pos[5],feat_neg[5]]])



################### Subjectivity Classification of Test Data



    feature_list_subj_test=[]
    pos_score_list_test = []

    for index in range(0,len(list_tweets_test)):
        feat_pos = [0]*6
        feat_neg = [0]*6



        if score_emoticon_test[index] == 0:
            feat_pos[0]=score_emoticon_likelihood[0][0]
            feat_neg[0]=score_emoticon_likelihood[0][1]
        elif score_emoticon_test[index] == 1:
            feat_pos[0]=score_emoticon_likelihood[1][0]
            feat_neg[0]=score_emoticon_likelihood[1][1]
        elif score_emoticon_test[index] >= 2:
            feat_pos[0]=score_emoticon_likelihood[2][0]
            feat_neg[0]=score_emoticon_likelihood[2][1]
        elif score_emoticon_test[index] == -1:
            feat_pos[0]=score_emoticon_likelihood[3][0]
            feat_neg[0]=score_emoticon_likelihood[3][1]
        else:
            feat_pos[0]=score_emoticon_likelihood[4][0]
            feat_neg[0]=score_emoticon_likelihood[4][1]



        if score_mpqa_all_test[index] == 0:
            feat_pos[1]=score_mpqa_all_likelihood[0][0]
            feat_neg[1]=score_mpqa_all_likelihood[0][1]
        elif score_mpqa_all_test[index] == 1:
            feat_pos[1]=score_mpqa_all_likelihood[1][0]
            feat_neg[1]=score_mpqa_all_likelihood[1][1]
        elif score_mpqa_all_test[index]==2 or score_mpqa_all_test[index]==3:
            feat_pos[1]=score_mpqa_all_likelihood[2][0]
            feat_neg[1]=score_mpqa_all_likelihood[2][1]
        elif score_mpqa_all_test[index]==4 or score_mpqa_all_test[index]==5:
            feat_pos[1]=score_mpqa_all_likelihood[3][0]
            feat_neg[1]=score_mpqa_all_likelihood[3][1]
        elif score_mpqa_all_test[index]>=6 or score_mpqa_all_test[index]<=9:
            feat_pos[1]=score_mpqa_all_likelihood[4][0]
            feat_neg[1]=score_mpqa_all_likelihood[4][1]
        elif score_mpqa_all_test[index]>=10:
            feat_pos[1]=score_mpqa_all_likelihood[5][0]
            feat_neg[1]=score_mpqa_all_likelihood[5][1]
        elif score_mpqa_all_test[index] == -1:
            feat_pos[1]=score_mpqa_all_likelihood[6][0]
            feat_neg[1]=score_mpqa_all_likelihood[6][1]
        elif score_mpqa_all_test[index] == -2:
            feat_pos[1]=score_mpqa_all_likelihood[7][0]
            feat_neg[1]=score_mpqa_all_likelihood[7][1]
        elif score_mpqa_all_test[index]==-3 or score_mpqa_all_test[index]==-4:
            feat_pos[1]=score_mpqa_all_likelihood[8][0]
            feat_neg[1]=score_mpqa_all_likelihood[8][1]
        else:
            feat_pos[1]=score_mpqa_all_likelihood[9][0]
            feat_neg[1]=score_mpqa_all_likelihood[9][1]



        feat_pos[2] = words_bayes_subj_test[index]
        feat_neg[2] = 1 - (words_bayes_subj_test[index])



        if num_emoticon_pos_test[index]==0:
            feat_pos[3]=num_emoticon_pos_likelihood[0][0]
            feat_neg[3]=num_emoticon_pos_likelihood[0][1]
        elif num_emoticon_pos_test[index]==1:
            feat_pos[3]=num_emoticon_pos_likelihood[1][0]
            feat_neg[3]=num_emoticon_pos_likelihood[1][1]
        else:
            feat_pos[3]=num_emoticon_pos_likelihood[2][0]
            feat_neg[3]=num_emoticon_pos_likelihood[2][1]



        if num_emoticon_neg_test[index]==0:
            feat_pos[4]=num_emoticon_neg_likelihood[0][0]
            feat_neg[4]=num_emoticon_neg_likelihood[0][1]
        elif num_emoticon_neg_test[index]==1:
            feat_pos[4]=num_emoticon_neg_likelihood[1][0]
            feat_neg[4]=num_emoticon_neg_likelihood[1][1]
        else:
            feat_pos[4]=num_emoticon_neg_likelihood[2][0]
            feat_neg[4]=num_emoticon_neg_likelihood[2][1]



        if score_mpqa_neg_test[index] == 0:
            feat_pos[5]=score_mpqa_neg_likelihood[0][0]
            feat_neg[5]=score_mpqa_neg_likelihood[0][1]
        elif score_mpqa_neg_test[index] == -1:
            feat_pos[5]=score_mpqa_neg_likelihood[1][0]
            feat_neg[5]=score_mpqa_neg_likelihood[1][1]
        elif score_mpqa_neg_test[index] == -2:
            feat_pos[5]=score_mpqa_neg_likelihood[2][0]
            feat_neg[5]=score_mpqa_neg_likelihood[2][1]
        elif score_mpqa_neg_test[index]==-3 or score_mpqa_neg_test[index]==-4:
            feat_pos[5]=score_mpqa_neg_likelihood[3][0]
            feat_neg[5]=score_mpqa_neg_likelihood[3][1]
        elif score_mpqa_neg_test[index]==-5 or score_mpqa_neg_test[index]==-6:
            feat_pos[5]=score_mpqa_neg_likelihood[4][0]
            feat_neg[5]=score_mpqa_neg_likelihood[4][1]
        else:
            feat_pos[5]=score_mpqa_neg_likelihood[5][0]
            feat_neg[5]=score_mpqa_neg_likelihood[5][1]



        score_pos = feat_pos[0]*feat_pos[2]*feat_pos[3]*feat_pos[4]
        score_neg = feat_neg[0]*feat_neg[2]*feat_neg[3]*feat_neg[4]
        final_score_pos = float(score_pos)/(score_pos + score_neg)

        pos_score_list_test.append("%.4f" % final_score_pos)

        


        
    print "check_6: Naive Bayes Classification"


####################################################################################  GENERATING WEKA FILE


##    list_obj_train=[]
##    list_subj_train=[]
##
##    for index in range(0,len(list_tweets_training)):
##        list_obj_train.append([str(presence_url_train[index]),str(presence_emoticon_train[index]),str(words_mpqa_train[index]),str(words_bayes_obj_train[index]),str(num_exclamations_train[index]),str(pos_prp_combined_train[index]),obj_score_list_train[index],training_labels[index]])
##        list_subj_train.append([str(score_emoticon_train[index]),str(score_mpqa_all_train[index]),str(words_bayes_subj_train[index]),str(num_emoticon_pos_train[index]),str(num_emoticon_neg_train[index]),str(score_mpqa_neg_train[index]),pos_score_list_train[index],training_labels[index]])





    for index in range(0,len(list_tweets_training)):
        final_list_train.append([obj_score_list_train[index],pos_score_list_train[index],str(training_labels[index])])
    for index in range(0,len(list_tweets_test)):
        final_list_test.append([obj_score_list_test[index],pos_score_list_test[index],str(test_labels[index])])



##    for index in range(0,len(list_tweets_training)):
##        final_list_train.append([str(num_exclamations_train[index]),str(presence_url_train[index]),str(presence_emoticon_train[index]),str(pos_prp_combined_train[index]),str(words_mpqa_train[index]),str(words_bayes_obj_train[index]),str(score_emoticon_train[index]),str(num_emoticon_pos_train[index]),str(num_emoticon_neg_train[index]),str(score_mpqa_all_train[index]),str(score_mpqa_neg_train[index]),str(words_bayes_subj_train[index]),str(training_labels[index])])
##    for index in range(0,len(list_tweets_test)):
##        final_list_test.append([str(num_exclamations_test[index]),str(presence_url_test[index]),str(presence_emoticon_test[index]),str(pos_prp_combined_test[index]),str(words_mpqa_test[index]),str(words_bayes_obj_test[index]),str(score_emoticon_test[index]),str(num_emoticon_pos_test[index]),str(num_emoticon_neg_test[index]),str(score_mpqa_all_test[index]),str(score_mpqa_neg_test[index]),str(words_bayes_subj_test[index]),str(test_labels[index])])




    for ind in range(0,len(final_list_train)):
        final_list_train[ind]=string.join(final_list_train[ind],",")
    for ind in range(0,len(final_list_test)):
        final_list_test[ind]=string.join(final_list_test[ind],",")


    final_list_train = string.join(final_list_train,"\n")
    final_list_test = string.join(final_list_test,"\n")



##    return list_obj_train, list_subj_train, feature_list_obj_train, feature_list_subj_train, likelihood_obj, likelihood_subj

    return final_list_train, final_list_test




