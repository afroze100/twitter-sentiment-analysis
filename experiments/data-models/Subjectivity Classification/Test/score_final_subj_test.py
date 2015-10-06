def score_subj():

    import nltk
    import string
    import re

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Test/test_tweets_890.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Test/test_labels_890.txt","r")
    file_feature_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Features/dict_mpqa_stem.txt","r")
    file_words_bayes=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Test/words_likelihood_subj_890.txt")


    dict_mpqa=eval(file_feature_mpqa.read())   ## dictionary of the word lexicons
    words_bayes = eval(file_words_bayes.read())
    list_tweets=eval(file_tweets.read())
    list_labels=eval(file_labels.read())    


    list_tweets_2=[] ## list_tweets_2 is a tokenized list (word_tokenize)
    list_tweets_3=[] ## list_tweets_3 is a tokenized list which excludes @'s and url's
    list_tweets_4=[]    ## punctuations and digits removed and de-capitalized and stemmed

    porter=nltk.PorterStemmer()

    pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)|[Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
    neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)|[\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."
    

#############################
    
    
    ## Features
 
    score_emoticon=[]
    
    score_mpqa_all=[]

    pos_rb=[]
    pos_wrb=[]
    pos_colon=[]
    pos_vb_combined=[]

    final_list=[]
    final_list_2=[]


#############################


    for index in range(0,len(list_tweets)):
        list_tweets_2.append(nltk.word_tokenize((list_tweets[index])))  




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



    for tweet in list_tweets_3:         ## for opulating list_tweets_4
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




    for tweet in list_tweets:                                   ## for number of emoticons in a tweet

        num_emoticon_pos = len(re.findall(pos_emoticon_re,tweet))
        num_emoticon_neg = len(re.findall(neg_emoticon_re,tweet))

        score_emoticon.append(num_emoticon_pos - num_emoticon_neg)



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

        pos_rb.append(pos.get("RB",0))
        pos_wrb.append(pos.get("WRB",0))
        pos_colon.append(pos.get(":",0))
        pos_vb_combined.append(pos.get("VB",0) + pos.get("VBG",0) + pos.get("VBN",0) + pos.get("VBZ",0) + pos.get("VBP",0))



################################# the following tokens are pertinent to word models



    for tweet in list_tweets_4:
        pos_mpqa=0
        neg_mpqa=0
        for word in tweet:
            if word in dict_mpqa.keys():
                if dict_mpqa[word]>0:
                    pos_mpqa+=dict_mpqa[word]
                elif dict_mpqa[word]<0:
                    neg_mpqa+=dict_mpqa[word]

        score_mpqa_all.append(float(pos_mpqa+neg_mpqa))
            



################################################################# Classification



    pos_score_list = []
    feature_list = []

    for index in range(0,len(list_tweets)):
        feat_pos = [0]*7
        feat_neg = [0]*7


        if score_emoticon[index] == 0:
            feat_pos[0]=0.7007
            feat_neg[0]=0.8548
        elif score_emoticon[index] == 1:
            feat_pos[0]=0.2585
            feat_neg[0]=0.0074
        elif score_emoticon[index] >= 2:
            feat_pos[0]=0.0306
            feat_neg[0]=0.00107
        elif score_emoticon[index] == -1:
            feat_pos[0]=0.0094
            feat_neg[0]=0.1314
        else:
            feat_pos[0]=0.00078
            feat_neg[0]=0.0053


        if score_mpqa_all[index] == 0:
            feat_pos[1]=0.1649
            feat_neg[1]=0.2061
        elif score_mpqa_all[index] == 1:
            feat_pos[1]=0.1775
            feat_neg[1]=0.1436
        elif score_mpqa_all[index]==2 or score_mpqa_all[index]==3:
            feat_pos[1]=0.3118
            feat_neg[1]=0.1929
        elif score_mpqa_all[index]==4 or score_mpqa_all[index]==5:
            feat_pos[1]=0.1708
            feat_neg[1]=0.0896
        elif score_mpqa_all[index]>=6 or score_mpqa_all[index]<=9:
            feat_pos[1]=0.1238
            feat_neg[1]=0.0525
        elif score_mpqa_all[index]>=10:
            feat_pos[1]=0.01998
            feat_neg[1]=0.0037
        elif score_mpqa_all[index] == -1:
            feat_pos[1]=0.0458
            feat_neg[1]=0.1219
        elif score_mpqa_all[index] == -2:
            feat_pos[1]=0.0302
            feat_neg[1]=0.1108
        elif score_mpqa_all[index]==-3 or score_mpqa_all[index]==-4:
            feat_pos[1]=0.0149
            feat_neg[1]=0.0694
        else:
            feat_pos[1]=0.0043
            feat_neg[1]=0.0148


        feat_pos[2] = words_bayes[index]
        feat_neg[2] = 1 - (words_bayes[index])


        if pos_vb_combined[index] == 0:
            feat_pos[3]=0.2033
            feat_neg[3]=0.1273
        elif pos_vb_combined[index] == 1:
            feat_pos[3]=0.2863
            feat_neg[3]=0.2541
        elif pos_vb_combined[index] == 2:
            feat_pos[3]=0.2312
            feat_neg[3]=0.2568
        elif pos_vb_combined[index] == 3:
            feat_pos[3]=0.1463
            feat_neg[3]=0.1668
        elif pos_vb_combined[index] == 4:
            feat_pos[3]=0.0696
            feat_neg[3]=0.1060
        elif pos_vb_combined[index] == 5:
            feat_pos[3]=0.0433
            feat_neg[3]=0.0437
        elif pos_vb_combined[index] == 6:
            feat_pos[3]=0.0122
            feat_neg[3]=0.0229
        elif pos_vb_combined[index] == 7:
            feat_pos[3]=0.0059
            feat_neg[3]=0.0165
        else:
            feat_pos[3]=0.00197
            feat_neg[3]=0.00586


        if pos_colon[index] == 0:
            feat_pos[4]=0.54345
            feat_neg[4]=0.7304
        elif pos_colon[index] == 1:
            feat_pos[4]=0.1746
            feat_neg[4]=0.14012
        elif pos_colon[index] == 2:
            feat_pos[4]=0.1581
            feat_neg[4]=0.0719
        elif pos_colon[index] == 3:
            feat_pos[4]=0.0653
            feat_neg[4]=0.0256
        elif pos_colon[index] == 4:
            feat_pos[4]=0.0256
            feat_neg[4]=0.01385
        elif pos_colon[index]==5 or pos_colon[index]==6:
            feat_pos[4]=0.02477
            feat_neg[4]=0.0117
        elif pos_colon[index]==7 or pos_colon[index]==8:
            feat_pos[4]=0.0059
            feat_neg[4]=0.00426
        else:
            feat_pos[4]=0.00236
            feat_neg[4]=0.00213


        if pos_rb[index] == 0:
            feat_pos[5]=0.5369
            feat_neg[5]=0.4315
        elif pos_rb[index] == 1:
            feat_pos[5]=0.3004
            feat_neg[5]=0.3307
        elif pos_rb[index] == 2:
            feat_pos[5]=0.1188
            feat_neg[5]=0.1592
        elif pos_rb[index] == 3:
            feat_pos[5]=0.03255
            feat_neg[5]=0.0584
        elif pos_rb[index] == 4:
            feat_pos[5]=0.00745
            feat_neg[5]=0.0143
        elif pos_rb[index] == 5:
            feat_pos[5]=0.00353
            feat_neg[5]=0.00425
        else:
            feat_pos[5]=0.00159
            feat_neg[5]=0.00039


        if pos_wrb[index] == 0:
            feat_pos[6]=0.9339
            feat_neg[6]=0.8535
        elif pos_wrb[index] == 0:
            feat_pos[6]=0.0641
            feat_neg[6]=0.13745
        else:
            feat_pos[6]=0.001966
            feat_neg[6]=0.00906


        score_pos = feat_pos[0]*feat_pos[1]*feat_pos[2]*feat_pos[3]*feat_pos[4]*feat_pos[5]*feat_pos[6]
        score_neg = feat_neg[0]*feat_neg[1]*feat_neg[2]*feat_neg[3]*feat_neg[4]*feat_neg[5]*feat_neg[6]
        final_score_pos = float(score_pos)/(score_pos + score_neg)

        pos_score_list.append("%.4f" % final_score_pos)

        feature_list.append([[feat_pos[0],feat_neg[0]],[feat_pos[1],feat_neg[1]],[feat_pos[2],feat_neg[2]],[feat_pos[3],feat_neg[3]],[feat_pos[4],feat_neg[4]],[feat_pos[5],feat_neg[5]],[feat_pos[6],feat_neg[6]]])



################################################################
        
    for ind in range(0,len(words_bayes)):
        words_bayes[ind] = "%.4f" % words_bayes[ind]                                                                



    for index in range(0,len(list_tweets)):                     ## for ease in generating the csv file for WEKA
        final_list.append([str(score_emoticon[index]),str(score_mpqa_all[index]),words_bayes[index],str(pos_vb_combined[index]),str(pos_colon[index]),str(pos_rb[index]),str(pos_wrb[index]),pos_score_list[index],list_labels[index]])
        final_list_2.append([pos_score_list[index],list_labels[index]])
    

    for ind in range(0,len(final_list)):
        final_list[ind]=string.join(final_list[ind],",")
        final_list_2[ind]=string.join(final_list_2[ind],",")

    final_list=string.join(final_list,"\n")
    final_list_2=string.join(final_list_2,"\n")


    return final_list, final_list_2, pos_score_list, list_labels
