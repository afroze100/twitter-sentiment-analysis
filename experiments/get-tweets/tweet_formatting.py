def file2tweet(directory_tweet): ##directory is in the form of a string

##could also give more effect to polarity of more conneced tweets/user

    import string
    
    directory_tweet="sample_eval_10apr.txt"
    fp=open(directory_tweet,"r")
    dictlist=fp.readlines() ##returns a list of individual tweets as string

    directory_dict="D:\SEECS\Research & Projects\FYP\Resources\Raw Resources\Word Lists\word lists/common english words_short.txt"
    fd=open(directory_dict,"r")
    eng_commonwords=fd.readlines() ##list of common english words
    for word_ind in range(0,len(eng_commonwords)):
        eng_commonwords[word_ind]=string.replace(eng_commonwords[word_ind],"\n","")

    ##all 6 below are in output format and are arranged in level of increasing filtering
    
    tweetlist_raw=[] ##contains RT's and is a form of continuous/block form (non-paragraphed data)
    tweetlist_RTremoved=[] ##removes RT's and is continuous/block form (non-paragraphed data)
    tweetlist_shortremoved=[] ##removes very short tweet with little or no info
    tweetlist_dissimilar=[] ##removed dissimilar tweets, similarity is a percentage of common words
    tweetlist_english=[] ##removes non-english tweets
    tweetlist_final=[] ##the most final form, most filtered

    tweetlist_parsed=[] ##for processing tweetlist_shortremoved
    threshold_short = 20 ##threshold for removing tweet sizes smaller than this
    threshold_similar = 0.8 ##tweets more than or equal to 90% similar will be discarded
    threshold_english = 0.15 ##tweets below this will be removed, meaning x% of tweet words have to be from some english dictionary
    
    

    for index in range(0,len(dictlist)):
        dictlist[index]=eval(dictlist[index])
        if (dictlist[index]['user']['lang']=='en'): ##only takes english tweets
            tweetlist_raw.append(dictlist[index]['text'])

            for ind in range(0,len(tweetlist_raw[-1])): 
                if (ord(tweetlist_raw[-1][ind])>128): ##remove special characters in unicode
                    tweetlist_raw[-1]=tweetlist_raw[-1].replace(tweetlist_raw[-1][ind],' ')       
            tweetlist_raw[-1]=str(tweetlist_raw[-1]) ##converts unicode to ascii atring

            tweetlist_RTremoved.append(tweetlist_raw[-1])
            label=0 ##label checks the presence of Retweets

            if(tweetlist_RTremoved[-1].find('\n') != -1): ##for removing \n in the tweet text
                tweetlist_RTremoved[-1]=string.split(tweetlist_RTremoved[-1])
                tweetlist_RTremoved[-1]=string.join(tweetlist_RTremoved[-1])            

            if(tweetlist_RTremoved[-1].find('RT @') != -1): ##Code to remove RT's
                del tweetlist_RTremoved[-1]
                label=1 ##if label is 1 then there was RT and so the last tweet was deleted


            #### following code is to removed very short tweets
                
            ##print tweetlist_RTremoved
            if(tweetlist_RTremoved != []):
                if((tweetlist_RTremoved[-1] != []) and (label != 1)): ##if it is not an empty list and if the last item in RTremoved was not deleted.
                    tweetlist_parsed.append(tweetlist_RTremoved[-1])
                    tweetlist_shortremoved.append(tweetlist_RTremoved[-1])
                
                    tweetlist_parsed[-1]=tweetlist_parsed[-1].split() ##split the tweet into words for easy comparison

                    ind_1=0
                    while ind_1 < len(tweetlist_parsed[-1]): ##code for removing unnecessary words starting with @
                        if tweetlist_parsed[-1][ind_1][0]=="@":
                            del tweetlist_parsed[-1][ind_1]
                            ind_1=ind_1-1
                        ind_1=ind_1+1

                    ind_2=0
                    while ind_2 < len(tweetlist_parsed[-1]): ##code for removing unnecessary words starting with #
                        if tweetlist_parsed[-1][ind_2][0]=="#":
                            del tweetlist_parsed[-1][ind_2]
                            ind_2=ind_2-1
                        ind_2=ind_2+1       

                    ind_3=0
                    while ind_3 < len(tweetlist_parsed[-1]): ##code for removing unncessary urls
                        if tweetlist_parsed[-1][ind_3][0:4]=="http":
                            del tweetlist_parsed[-1][ind_3]
                            ind_3=ind_3-1
                        ind_3=ind_3+1

                    tweetlist_parsed[-1]=string.join(tweetlist_parsed[-1]) ##code for removing unnecessary punctuation
                    for items in string.punctuation:
                        tweetlist_parsed[-1]=string.replace(tweetlist_parsed[-1],items,"")

                    tweetlist_parsed[-1]=string.split(tweetlist_parsed[-1]) ##splitting and joining to remove unncessary spaces from the tweet
                    tweetlist_parsed[-1]=string.join(tweetlist_parsed[-1])

                    if (len(tweetlist_parsed[-1]) < threshold_short):
                        del tweetlist_shortremoved[-1]
                        del tweetlist_parsed[-1]


    ##code for removing similar tweets
    print "check_1"
    print len(tweetlist_parsed)

    similarity_list_augmented=[]
    for tweet_ind2 in range(0,len(tweetlist_parsed)):
        print tweet_ind2
        tweet_first=string.split(tweetlist_parsed[tweet_ind2]) ##the tweet with respect to which we will calculate similarity values of all tweets ahead of it
        similarity_list=[]
        for tweet_ind2_2 in range (tweet_ind2+1,len(tweetlist_parsed)):
            tweet_other=string.split(tweetlist_parsed[tweet_ind2_2]) ##the tweets ahead whose similarity value is calculated wrt to the tweet_first
            common_1=0
            for words_ind2 in range(0,len(tweet_first)):
                if words_ind2 < len(tweet_other):  ##so that index is not exceeded
                    for words_ind2_2 in range(0,len(tweet_other)):
                        if tweet_first[words_ind2]==tweet_other[words_ind2_2]:
                            common_1+=1 ##count of common words b/w the two (a simple bag of words model with no structure element)
                            break ##so that one word is only counted as 1 even if it occurs multiple times in the other tweet
            similarity=float(common_1)/len(tweet_other) ##algo for calculating similarity
            similarity_list.append(similarity) ##contains the similarity values of a particular tweet with all tweets ahead of it
        similarity_list_augmented.append(similarity_list) ##a super set containing the above info for all tweets

        similarity_indexes=[]
        for dim1 in range(0,len(similarity_list_augmented)): ##going through the list of tweet_first (wrt which similarity is measured)
            for dim2 in range(0,len(similarity_list_augmented[dim1])): ##going through the list of tweet_other (whose similarity is measured)
                if similarity_list_augmented[dim1][dim2]>=threshold_similar: ##if the similarity value goes above a certain threshold
                    similarity_indexes.append(dim1+dim2+1) ##the indexes of tweet_others whose similarity value is high so need to be removed

    print "check_2"
    for ind_sim in range(0,len(tweetlist_parsed)):
        if similarity_indexes.count(ind_sim)==0: ##index not occuring in the similarity list
            tweetlist_dissimilar.append(tweetlist_shortremoved[ind_sim]) ##so only those indexes are appended which are not listed in similarity_indexes


    ##code for removing non-english tweets
    print "check_3"
    
    ##tweetlist_english=tweetlist_shortremoved[:] ##if we are not doing the dissimilar part of filering (above) which takes extensive time!
    tweetlist_english=tweetlist_dissimilar[:]
    tweet_ind=0
    while tweet_ind < len(tweetlist_english): ##for removing non-english tweets by comparing them to a short english dictionary
        tweetlist=string.split(tweetlist_english[tweet_ind])
        common=0
        for word_dict in eng_commonwords: ##list of ~150 most common english words as a comparison dictionary/reference
            for word_tweet in tweetlist:
                word_tweet=string.lower(word_tweet)
                if word_tweet==word_dict: 
                    common+=1 ##common words wrt to the short english dictionary
        size=len(string.split(tweetlist_english[tweet_ind]))
        value=float(common)/size ##value acts as a probability of the tweet belonging to english
        if value < threshold_english:
            del tweetlist_english[tweet_ind]
            tweet_ind-=1
        tweet_ind+=1



    tweetlist_final=tweetlist_english ##this is the most filtered list of the order ~250/1500
                
                                                      
##    tweetlist_final=string.join(tweetlist_final,'\n') ##for formatting the list in a form which can be directly printed in a text file
    
    return tweetlist_final 
