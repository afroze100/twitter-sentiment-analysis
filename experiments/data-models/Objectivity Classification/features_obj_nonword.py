def feature_obj():

    import nltk
    import string
    import enchant

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_3_9804.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/labels_obj_9804.txt","r")
    file_emoticons=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/emoticons_list_short.txt","r")
    file_feature_mpqa=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification\Features/feature_unigram_mpqa_norm_9804.txt","r")
    file_feature_combined=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification\Features/feature_unigram_combined_norm_9804.txt","r")
##    file_feature_sum=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification\Features/feature_unigram_0.65_10_sum_9804.txt","r")
    

    list_tweets=file_tweets.readlines()
    list_labels=file_labels.readlines()
    list_emoticons=file_emoticons.readlines()


    for index in range(0,len(list_emoticons)):      ## the last element is /n and has to be removed
        list_emoticons[index]=list_emoticons[index][0:-1]

    dict_US = enchant.Dict("en_US")                 ## enchant english dictionaries
    dict_UK = enchant.Dict("en_UK")

    list_tweets_2=[] ## list_tweets_2 is a tokenized list (word_tokenie)
    list_tweets_3=[] ## list_tweets_3 is a tokenized list which excludes @'s and url's

    list_pos=[] ## list of pos tags of each tweet
    

    
    ## Features

    num_exclamations=[] ##
    num_questions=[] 
    presence_exclamations=[]
    presence_questions=[] ##
    presence_url=[] ## ##
    presence_digits=[]
    ratio_caps=[]
    ratio_nondict=[]
    presence_emoticon=[] ## ## 

    num_digits=[] ##
    num_numerals=[]
    num_caps_eng=[]
    num_punct=[] ## ? (c)
    num_emoticon=[]

    pos_jj=[]
    pos_jjs=[]
    pos_jjr=[]
    pos_vb=[]
    pos_vbg=[]
    pos_vbn=[]
    pos_vbz=[]
    pos_vbp=[] ## ?
    pos_rb=[] ## ?
    pos_prp=[]
    pos_prpS=[]
    pos_nnp=[]
    pos_nnps=[]
    pos_cd=[]
    pos_pos=[]
    pos_wp=[]
    pos_jj_combined=[] ## ?         ## sum of pos_jj, pos_jjs & pos_jjr
    pos_prp_combined=[] ## ##       ## sum of pos_prp & pos_prp$
    pos_vb_combined=[] ## ? (c)        ## sum of pos_vb, pos_vbg, pos_vbn & pos_vbz
    pos_nnp_combined=[] ##          ## sum of pos_nnp & pos_nnps

    words_mpqa=eval(file_feature_mpqa.read()) ## ##
    words_combined=eval(file_feature_combined.read())

    final_list=[]
    



    for index in range(0,len(list_tweets)):

        num_exclamations.append(str(string.count(list_tweets[index],"!")))   ## exclamation marks

        if string.count(list_tweets[index],"!")==0:
            presence_exclamations.append("no")
        else:
            presence_exclamations.append("yes")

        num_questions.append(str(string.count(list_tweets[index],"?")))      ## question marks

        if string.count(list_tweets[index],"?")==0:
            presence_questions.append("no")
        else:
            presence_questions.append("yes")

        if string.find(list_tweets[index],"http://") == -1:                  ## url's
            presence_url.append("no") 
        else:
            presence_url.append("yes")


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

        check=0
        phrase2=string.join(list_tweets_3[index])
        for ind in range(0,len(string.digits)):
            if string.find(phrase2,string.digits[ind])!=-1 and check==0:     ## digits/numerals
                presence_digits.append("yes")
                check=1
        if check==0:
            presence_digits.append("no")
            check=1




    for  index in range(0,len(list_tweets_3)):                  ## for ratio calculations (removing @'s, url's and punctuation)
        tweet_string=string.join(list_tweets_3[index])          
        tweet_token=nltk.wordpunct_tokenize(tweet_string)

        for ind_p in range(0,len(string.punctuation)):
            num = tweet_token.count(string.punctuation[ind_p])
            iteration = 0
            while iteration < num:
                tweet_token.remove(string.punctuation[ind_p])
                iteration = iteration + 1

        capital_words=0
        for ind in range(0,len(tweet_token)):                   ## ratio of words in caps to total words 
            if len(tweet_token[ind]) >= 2:                      ## also see the effect of removing stop words from total words in tweet
                x=0                                             ## and also of using the numeric count of caps words as feature
                for ind2 in range(0,len(tweet_token[ind])):
                    if tweet_token[ind][ind2] in string.uppercase:  ## can be replaced by ascii_uppecase for more efficiency
                        x=x+1
                    else:
                        break
                if len(tweet_token[ind])==x:
                    capital_words=capital_words+1

        ratio1=float(capital_words)/len(tweet_token)
        ratio_caps.append("%.2f" % ratio1)




        nondict_words=0                                         ## ratio of non-dictionaty words to total words in tweet
        for word in tweet_token:                                ## (since using wordpunc_tokenize so YET to take care of excluding "apostrophe s" as a word) 
            word=word.lower()                                   ## effect of using numeric count of non-dictionary words as a feature
            if dict_US.check(word) == False:
                if dict_UK.check(word) == False:
                    nondict_words=nondict_words+1

        ratio2=float(nondict_words)/len(tweet_token)
        ratio_nondict.append("%.2f" % ratio2)
                
            



    for index in range(0,len(list_tweets)):                     ## for emoticons (also try for number or emoticons per tweet as a feature)
        check=0                                                 ## also check if "~" makes a good emoticon
        for emoticon in list_emoticons:
            if string.find(list_tweets[index],emoticon) != -1:
                presence_emoticon.append("yes")
                check=1
                break
        if check==0:
            presence_emoticon.append("no")




    for tweet in list_tweets_3:                                 ## for number of digits in a tweet
        tweet_string=string.join(tweet)
        num=0
        for char in tweet_string:
            if char in string.digits:
                num+=1
        num_digits.append(str(num))




    for tweet in list_tweets_3:                                 ## fot number of complete numeral words
        num=0
        for word in tweet:
            length=len(word)
            count=0
            for char in word:
                if char in string.digits:
                    count+=1
                elif (char == ",") or (char == "."):
                    if count>=1:
                        count+=1
            if count == length:
                num+=1
        num_numerals.append(str(num))




    for tweet in list_tweets_3:                                 ## for number of capitalized words in a tweet which appear in dictionary
        tweet=string.join(tweet)
        tweet=nltk.wordpunct_tokenize(tweet)
        num=0
        for word in tweet:
            length=len(word)
            if length>=3:       ## only capitalizations of words of length 3 characters or greater
                count=0
                for char in word:
                    if char in string.uppercase:
                        count+=1
                if count == length:
                    if dict_US.check(word) == True or dict_UK.check(word) == True:
                        num+=1
        num_caps_eng.append(str(num))
                



    for tweet in list_tweets_3:                                 ## for number of punctuation marks in a tweet                         
        tweet=string.join(tweet)

        num=0
        for char in tweet:
            if char in string.punctuation:
                num+=1
        num_punct.append(str(num))




    for tweet in list_tweets:                                   ## for number of emoticons in a tweet
        start=0
        num=0
        for emoticon in list_emoticons:
            while start<len(list_tweets) and start>=0:
                start=string.find(tweet,emoticon,start)
                if start!=-1:
                    start=start+1
                    num+=1
        num_emoticon.append(str(num))



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
        pos_prp_combined.append(str(pos.get("PRP",0) + pos.get("PRP$",0)))
        pos_nnp_combined.append(str(pos.get("NNP",0) + pos.get("NNPS",0)))



################################# the following tokens are pertinent to word models


##    for i in range(0,len(words_sum)):
##        words_sum[i]=("%.2f" % words_sum[i])
                
    for i in range(0,len(words_mpqa)):
        words_mpqa[i]=("%.2f" % words_mpqa[i])

    for i in range(0,len(words_combined)):
        words_combined[i]=("%.2f" % words_combined[i])




#########################################################################
        
                                                                


    for index in range(0,len(list_tweets)):                     ## for ease in generating the csv file for WEKA
        final_list.append([num_exclamations[index],num_questions[index],presence_exclamations[index],presence_questions[index],presence_url[index],presence_digits[index],ratio_caps[index],ratio_nondict[index],presence_emoticon[index],num_digits[index],num_numerals[index],num_caps_eng[index],num_punct[index],num_emoticon[index],pos_jj[index],pos_jjs[index],pos_jjr[index],pos_vb[index],pos_vbg[index],pos_vbn[index],pos_vbz[index],pos_vbp[index],pos_rb[index],pos_prp[index],pos_prpS[index],pos_nnp[index],pos_nnps[index],pos_cd[index],pos_pos[index],pos_wp[index],pos_jj_combined[index],pos_vb_combined[index],pos_prp_combined[index],pos_nnp_combined[index],words_mpqa[index],words_combined[index],list_labels[index]])

    

    for ind in range(0,len(final_list)):
        final_list[ind]=string.join(final_list[ind],",")

    final_list=string.join(final_list,"")


    result = [final_list,list_tweets]

    ##return result
    return result
            
        
