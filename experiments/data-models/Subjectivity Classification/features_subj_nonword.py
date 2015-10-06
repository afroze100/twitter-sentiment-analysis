def feature_subj():

    import nltk
    import string
    import re

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification/tweets_subj_nonamb_4420.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification/labels_subj_nonamb_4420.txt","r")
    file_feature_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Features/dict_mpqa_stem.txt","r")
    file_feature_combined=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification\Features/dict_combined_stem.txt","r")
    

    list_tweets=file_tweets.read().splitlines()
    list_labels=file_labels.readlines()


    list_tweets_2=[] ## list_tweets_2 is a tokenized list (word_tokenize)
    list_tweets_3=[] ## list_tweets_3 is a tokenized list which excludes @'s and url's
    list_tweets_4=[]    ## punctuations and digits removed and de-capitalized and stemmed


    dict_mpqa=eval(file_feature_mpqa.read())   ## dictionary of the word lexicons
    dict_combined=eval(file_feature_combined.read())

    porter=nltk.PorterStemmer()

    pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)|[Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
    neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)|[\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."
    

#############################
    
    
    ## Features
 
    num_emoticon=[]
    num_emoticon_pos=[]
    num_emoticon_neg=[]
    score_emoticon=[]

    score_mpqa_pos=[]
    score_mpqa_neg=[]
    score_mpqa_all=[]
    score_combined_pos=[]
    score_combined_neg=[]
    score_combined_all=[]

    pos_vb=[]
    pos_vbp=[]
    pos_vbn=[]
    pos_vbz=[]
    pos_vbg=[]
    pos_rb=[]
    pos_nnp=[]
    pos_nns=[]
    pos_cd=[]
    pos_in=[]
    pos_wrb=[]
    pos_period=[]
    pos_colon=[]
    pos_vb_combined=[]

    final_list=[]


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




    for  index in range(0,len(list_tweets_3)):                  ## for ratio calculations (removing @'s, url's and punctuation)
        tweet_string=string.join(list_tweets_3[index])          
        tweet_token=nltk.wordpunct_tokenize(tweet_string)

        for ind_p in range(0,len(string.punctuation)):
            num = tweet_token.count(string.punctuation[ind_p])
            iteration = 0
            while iteration < num:
                tweet_token.remove(string.punctuation[ind_p])
                iteration = iteration + 1




    for tweet in list_tweets:                                   ## for number of emoticons in a tweet

        num_emoticon_pos.append(str(len(re.findall(pos_emoticon_re,tweet))))
        num_emoticon_neg.append(str(len(re.findall(neg_emoticon_re,tweet))))

        num_emoticon.append(str(int(num_emoticon_pos[-1]) + int(num_emoticon_neg[-1])))
        score_emoticon.append(str(int(num_emoticon_pos[-1]) - int(num_emoticon_neg[-1])))



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
        pos_vb.append(str(pos.get("VB",0)))
        pos_vbg.append(str(pos.get("VBG",0)))
        pos_vbn.append(str(pos.get("VBN",0)))
        pos_vbz.append(str(pos.get("VBZ",0)))
        pos_vbp.append(str(pos.get("VBP",0)))
        pos_rb.append(str(pos.get("RB",0)))
        pos_nnp.append(str(pos.get("NNP",0)))
        pos_nns.append(str(pos.get("NNS",0)))
        pos_cd.append(str(pos.get("CD",0)))
        pos_in.append(str(pos.get("IN",0)))
        pos_wrb.append(str(pos.get("WRB",0)))
        pos_period.append(str(pos.get(".",0)))
        pos_colon.append(str(pos.get(":",0)))
        pos_vb_combined.append(str(pos.get("VB",0) + pos.get("VBG",0) + pos.get("VBN",0) + pos.get("VBZ",0) + pos.get("VBP",0)))



################################# the following tokens are pertinent to word models



    for tweet in list_tweets_4:
        pos_mpqa=0
        neg_mpqa=0
        pos_comb=0
        neg_comb=0
        for word in tweet:
            if word in dict_mpqa.keys():
                if dict_mpqa[word]>0:
                    pos_mpqa+=dict_mpqa[word]
                elif dict_mpqa[word]<0:
                    neg_mpqa+=dict_mpqa[word]
            if word in dict_combined.keys():
                if dict_combined[word]>0:
                    pos_comb+=dict_combined[word]
                elif dict_combined[word]<0:
                    neg_comb+=dict_combined[word]

        score_mpqa_pos.append(str(float(pos_mpqa)))
        score_mpqa_neg.append(str(float(neg_mpqa)))
        score_combined_pos.append(str(float(pos_comb)))
        score_combined_neg.append(str(float(neg_comb)))
        score_mpqa_all.append(str(float(pos_mpqa+neg_mpqa)))
        score_combined_all.append(str(float(pos_comb+neg_comb)))
            



#########################################################################
        
                                                                


    for index in range(0,len(list_tweets)):                     ## for ease in generating the csv file for WEKA
        final_list.append([num_emoticon[index],num_emoticon_pos[index],num_emoticon_neg[index],score_emoticon[index],score_mpqa_pos[index],score_mpqa_neg[index],score_mpqa_all[index],score_combined_pos[index],score_combined_neg[index],score_combined_all[index],pos_vb[index],pos_vbp[index],pos_vbg[index],pos_vbn[index],pos_vbz[index],pos_vb_combined[index],pos_period[index],pos_colon[index],pos_cd[index],pos_in[index],pos_nnp[index],pos_nns[index],pos_rb[index],pos_wrb[index],list_labels[index]])

    

    for ind in range(0,len(final_list)):
        final_list[ind]=string.join(final_list[ind],",")

    final_list=string.join(final_list,"")


    return final_list,list_tweets

            
        
