def score_obj():

    import nltk
    import re
    import string
    import math

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/test-codes/test_tweets_890.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/test-codes/test_labels_890.txt","r")
    file_dict_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/dict_mpqa_stem.txt","r")
    file_words_bayes=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/test-codes/words_bayes_obj_890.txt")
    

    list_tweets=eval(file_tweets.read())
    list_labels=eval(file_labels.read())
    dict_mpqa = eval(file_dict_mpqa.read())
    words_bayes = eval(file_words_bayes.read())


    list_tweets_2=[] ## list_tweets_2 is a tokenized list (word_tokenize)
    list_tweets_3=[] ## list_tweets_3 is a tokenized list which excludes @'s and url's
    list_tweets_4=[]    ## punctuations and digits removed and de-capitalized and stemmed


    porter=nltk.PorterStemmer()

    pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)|[Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
    neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)|[\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."
    

    final_list=[]
    final_list_2=[]

#####################################

    
    ## Features

    num_exclamations=[]
    presence_questions=[]
    presence_url=[]
    presence_emoticon=[]

    pos_prp_combined=[]           ## sum of pos_prp & pos_prp$
    pos_nnp_combined=[]           ## sum of pos_nnp & pos_nnps

    words_mpqa=[]


    

####################################


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


        list_tweets_2.append(nltk.word_tokenize((list_tweets[index])))


    

    for tweet in list_tweets:                                               ## for emoticons

        if re.search(pos_emoticon_re,tweet) or re.search(neg_emoticon_re,tweet):
            presence_emoticon.append("yes")
        else:
            presence_emoticon.append("no")




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



    for tweet in list_tweets_3:         ## for populating list_tweets_4
        tweet=string.join(tweet)

        tweet=[x for x in tweet]        ## for splitting the tweet string by each character

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
            



    print "yahoo"


################################# following features are pertinent to Parts-of-Speech Tagging:
        

    pos_list = [] ## a list containing pos tagging info in form of dictionary for each tweet
    
    for tweet in list_tweets_3:
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



#######################################     The following features are relevant to words


    for tweet in list_tweets_4:
        score=0
        for word in tweet:
            if word in dict_mpqa.keys():
                score += math.fabs(dict_mpqa[word])

        words_mpqa.append(score)

    print "yahoo"


############################################################# The following code is relevant to classification algorithm



    obj_score_list = []
    feature_list = []

    for index in range(0,len(list_tweets)):
        feat_obj = [0]*8
        feat_subj = [0]*8

        if presence_url[index] == "yes":
            feat_obj[0] = 0.2404
            feat_subj[0] = 0.0355
        else:
            feat_obj[0] = 0.7596
            feat_subj[0] = 0.9645


        if presence_emoticon[index] == "yes":
            feat_obj[1] = 0.0308
            feat_subj[1] = 0.2346
        else:
            feat_obj[1] = 0.9692
            feat_subj[1] = 0.7654


        if words_mpqa[index] == 0:
            feat_obj[2] = 0.19855
            feat_subj[2] = 0.1086
        elif words_mpqa[index] == 1:
            feat_obj[2] = 0.1545
            feat_subj[2] = 0.09685
        elif words_mpqa[index] == 2:
            feat_obj[2] = 0.1841
            feat_subj[2] = 0.1836
        elif words_mpqa[index] == 3 or words_mpqa[index] == 4:
            feat_obj[2] = 0.2452
            feat_subj[2] = 0.26644
        elif words_mpqa[index] == 5 or words_mpqa[index] == 6:
            feat_obj[2] = 0.1282
            feat_subj[2] = 0.1813
        elif words_mpqa[index] == 7:
            feat_obj[2] = 0.03178
            feat_subj[2] = 0.0547
        elif words_mpqa[index] == 8 or words_mpqa[index] == 9:
            feat_obj[2] = 0.03156
            feat_subj[2] = 0.0624
        elif words_mpqa[index]>=10 and words_mpqa[index]<=12:
            feat_obj[2] = 0.0186
            feat_subj[2] = 0.03536
        elif words_mpqa[index]>=13 and words_mpqa[index]<=16:
            feat_obj[2] = 0.00504
            feat_subj[2] = 0.0065
        else:
            feat_obj[2] = 0.00022
            feat_subj[2] = 0.002027
        

        feat_obj[3] = words_bayes[index]
        feat_subj[3] = 1-(words_bayes[index])


        if num_exclamations[index] == 0:
            feat_obj[4] = 0.1992
            feat_subj[4] = 0.1088
        elif num_exclamations[index] == 1:
            feat_obj[4] = 0.15496
            feat_subj[4] = 0.09706
        elif num_exclamations[index] == 2:
            feat_obj[4] = 0.1847
            feat_subj[4] = 0.18416
        elif num_exclamations[index] == 3:
            feat_obj[4] = 0.1345
            feat_subj[4] = 0.13145
        elif num_exclamations[index]==4 or num_exclamations[index]==5:
            feat_obj[4] = 0.18446
            feat_subj[4] = 0.23756
        elif num_exclamations[index]>=6 and num_exclamations[index]<=9 :
            feat_obj[4] = 0.11886
            feat_subj[4] = 0.1975
        else:
            feat_obj[4] = 0.0233
            feat_subj[4] = 0.04344


        if presence_questions[index] == "yes":
            feat_obj[5] = 0.1503
            feat_subj[5] = 0.0749
        else:
            feat_obj[5] = 0.8497
            feat_subj[5] = 0.9251


        if pos_prp_combined[index] == 0:
            feat_obj[6] = 0.3674
            feat_subj[6] = 0.22195
        elif pos_prp_combined[index] == 1:
            feat_obj[6] = 0.3271
            feat_subj[6] = 0.3215
        elif pos_prp_combined[index] == 2:
            feat_obj[6] = 0.1794
            feat_subj[6] = 0.2244
        elif pos_prp_combined[index] == 3:
            feat_obj[6] = 0.0768
            feat_subj[6] = 0.1249
        elif pos_prp_combined[index] == 4:
            feat_obj[6] = 0.03016
            feat_subj[6] = 0.0622
        elif pos_prp_combined[index] == 5:
            feat_obj[6] = 0.01365
            feat_subj[6] = 0.02896
        elif pos_prp_combined[index] == 6:
            feat_obj[6] = 0.00374
            feat_subj[6] = 0.0093
        elif pos_prp_combined[index] == 7:
            feat_obj[6] = 0.00132
            feat_subj[6] = 0.0052
        else:
            feat_obj[6] = 0.0004
            feat_subj[6] = 0.001584


        if pos_nnp_combined[index] == 0:
            feat_obj[7] = 0.2804
            feat_subj[7] = 0.3165
        elif pos_nnp_combined[index] == 1:
            feat_obj[7] = 0.20515
            feat_subj[7] = 0.2498
        elif pos_nnp_combined[index] == 2:
            feat_obj[7] = 0.14066
            feat_subj[7] = 0.16606
        elif pos_nnp_combined[index] == 3:
            feat_obj[7] = 0.0984
            feat_subj[7] = 0.10407
        elif pos_nnp_combined[index]==4 or pos_nnp_combined[index]==5:
            feat_obj[7] = 0.1184
            feat_subj[7] = 0.0923
        elif pos_nnp_combined[index]==6 or pos_nnp_combined[index]==7:
            feat_obj[7] = 0.0702
            feat_subj[7] = 0.03665
        elif pos_nnp_combined[index]>=8 and pos_nnp_combined[index]<=10:
            feat_obj[7] = 0.04975
            feat_subj[7] = 0.01833
        elif pos_nnp_combined[index]>=11 and pos_nnp_combined[index]<=13:
            feat_obj[7] = 0.022
            feat_subj[7] = 0.0095
        elif pos_nnp_combined[index]>=14 and pos_nnp_combined[index]<=18:
            feat_obj[7] = 0.01272
            feat_subj[7] = 0.004977
        else:
            feat_obj[7] = 0.0022
            feat_subj[7] = 0.00181
        

        score_obj = feat_obj[0]*feat_obj[1]*feat_obj[2]*feat_obj[3]*feat_obj[4]*feat_obj[5]*feat_obj[6]*feat_obj[7]*(float(4543)/8963)
        score_subj = feat_subj[0]*feat_subj[1]*feat_subj[2]*feat_subj[3]*feat_subj[4]*feat_subj[5]*feat_subj[6]*feat_subj[7]*(float(4420)/8963)
        final_score_obj = float(score_obj)/(score_obj + score_subj)

        obj_score_list.append("%.4f" % final_score_obj)

        feature_list.append([[feat_obj[0],feat_subj[0]],[feat_obj[1],feat_subj[1]],[feat_obj[2],feat_subj[2]],[feat_obj[3],feat_subj[3]],[feat_obj[6],feat_subj[6]],[feat_obj[4],feat_subj[4]],[feat_obj[7],feat_subj[7]],[feat_obj[5],feat_subj[5]]])


#############################################################


    for ind in range(0,len(words_bayes)):
        words_bayes[ind] = "%.4f" % words_bayes[ind]
        words_mpqa[ind] = str(words_mpqa[ind])
        num_exclamations[ind] = str(num_exclamations[ind])
        pos_prp_combined[ind]=str(pos_prp_combined[ind])
        pos_nnp_combined[ind]=str(pos_nnp_combined[ind])
        
        if list_labels[ind]!="0":
            list_labels[ind]="s"

#############################################################
            

    for index in range(0,len(list_tweets)):                     ## for ease in generating the csv file for WEKA
        final_list.append([presence_url[index],presence_emoticon[index],words_mpqa[index],words_bayes[index],pos_prp_combined[index],num_exclamations[index],pos_nnp_combined[index],presence_questions[index],obj_score_list[index],list_labels[index]])
        final_list_2.append([obj_score_list[index],list_labels[index]])
    
    
    for ind in range(0,len(final_list)):
        final_list[ind]=string.join(final_list[ind],",")
        final_list_2[ind]=string.join(final_list_2[ind],",")

    final_list=string.join(final_list,"\n")
    final_list_2=string.join(final_list_2,"\n")


    


##    result = final_list,list_tweets
##    return result

    return obj_score_list,final_list,final_list_2
            
        
