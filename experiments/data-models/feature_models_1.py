def feature_models(directory):
    
    directory = "D:\SEECS\Research & Projects\FYP\Codes\Data Models/sample_afroze.txt"
    file = open(directory,"r")

    import string
    import nltk

    feature = string.digits
    ##feature2 = "?!"
    feature_count = 25
    

    list_tweets_1=[] ## list contains tweets as from the text file
    list_tweets_2=[] ## list contains tweets in tokenized form (each tweet in list form)
    list_tweets_3=[] ## list of each tweet in a string form (with extra items like /n /t removed at end)

    list_tweets_1=file.readlines()

    list_tweets_4=[] ## list containing tweets in split form excluding @'s and hyperlinks
    list_tweets_5=[] ## list containing tweets in tokenized form excluding @'s and hyperlinks


########
    

    for ind in range(0,len(list_tweets_1)):
        list_tweets_2.append(nltk.word_tokenize((list_tweets_1[ind])))
        list_tweets_3.append(string.join(string.split(list_tweets_1[ind]))) ## tokenizing separated emoticons like :) :P etc

        ind2=0
        phrase=[]
        while ind2 < len(string.split(list_tweets_3[ind])):  ##removing @'s and hyperlinks from the list_tweets_4
            word=string.split(list_tweets_3[ind])[ind2] ## 2
            if string.find(word,"@") >= 0:
                ind2=ind2+1
            elif string.find(word,"http") >= 0:
                ind2=ind2+1
            else:
                phrase.append(word)
                ind2=ind2+1
        list_tweets_4.append(phrase)

        ind3=0
        phrase2=[]
        while ind3 < len(list_tweets_2[ind]):
            word2=list_tweets_2[ind][ind3]
            if word2 == "@":
                ind3=ind3+2
            elif word2 == "http":
                ind3=ind3+3
            else:
                phrase2.append(word2)
                ind3=ind3+1
        list_tweets_5.append(phrase2)


##########


    index_pos=[] ## index of positive tweets
    index_neg=[] ## index of negative tweets
    index_neu=[] ## index of objective tweets
    index_ambig=[] ## index of ambiguous tweets


    for ind in range(0, len(list_tweets_2)):
        tweet=list_tweets_2[ind]
        if tweet[-1] == "1" or tweet[-1] == "2":
            index_pos.append(ind)
        elif tweet[-1] == "-1" or tweet[-1] == "-2":
            index_neg.append(ind)
        elif tweet[-1] == "0":
            index_neu.append(ind)
        elif tweet[-1] == "?" and tweet[-2] == "?":
            index_ambig.append(ind)


##########


    feature_pos=0 ## calculates the total tweet in each class in which this feature occurs
    feature_neg=0
    feature_neu=0
    feature_ambig=0

    occur_pos=0 ## calculates the total occurence of the feature in the class
    occur_neg=0
    occur_neu=0
    occur_ambig=0


#################################################################################### Presence of a particular exact element in a list (eg. ! ? @ $ % #)


##    for ind in index_ambig: 
##        tweet=list_tweets_2[ind][0:-2] ## finding features in ambiguous tweets
##        num = tweet.count(feature)
##        if num >= feature_count:
##            feature_ambig=feature_ambig+1
##        occur_ambig=occur_ambig+num
##
##
##    for ind in index_neu:
##        tweet=list_tweets_2[ind][0:-1] ## finding features in objective tweets
##        num = tweet.count(feature)
##        if num >= feature_count:
##            feature_neu=feature_neu+1
##        occur_neu=occur_neu+num
##
##    for ind in index_pos:
##        tweet=list_tweets_2[ind][0:-1] ## finding features in positive tweets
##        num = tweet.count(feature)
##        if num >= feature_count:
##            feature_pos=feature_pos+1
##        occur_pos=occur_pos+num
##
##    for ind in index_neg:
##        tweet=list_tweets_2[ind][0:-1] ## finding features in negative tweets
##        num = tweet.count(feature)
##        if num >= feature_count:
##            feature_neg=feature_neg+1
##        occur_neg=occur_neg+num
    

##################################################################################### Presence of digits/numerals in a tweet (use list_tweets_5)

##    for ind in index_ambig:
##        tweet=string.join(list_tweets_5[ind][0:-2]) ## finding features in objective tweets 
##        for ind2 in range(0,10):
##            num=string.count(tweet,string.digits[ind2])
##            occur_ambig=occur_ambig+num
##            position = string.find(tweet,string.digits[ind2])
##            if position >= 0:
##                position2 = string.find(tweet,string.digits[ind2],position+1)
##                if position2 >= 0:
##                    feature_ambig=feature_ambig+1
##                    break
##            
##
##    for ind in index_neu:
##        tweet=string.join(list_tweets_5[ind][0:-1]) ## finding features in objective tweets
##        for ind2 in range(0,10):
##            num=string.count(tweet,string.digits[ind2])
##            occur_neu=occur_neu+num
##            position = string.find(tweet,string.digits[ind2])
##            if position >= 0:
##                position2 = string.find(tweet,string.digits[ind2],position+1)
##                if position2 >= 0:
##                    feature_neu=feature_neu+1
##                    break
##        
##
##    for ind in index_pos:
##        tweet=string.join(list_tweets_5[ind][0:-1]) ## finding features in positive tweets
##        for ind2 in range(0,10):
##            num=string.count(tweet,string.digits[ind2])
##            occur_pos=occur_pos+num
##            position = string.find(tweet,string.digits[ind2])
##            if position >= 0:
##                position2 = string.find(tweet,string.digits[ind2],position+1)
##                if position2 >= 0:
##                    feature_pos=feature_pos+1
##                    break
##        
##
##    for ind in index_neg:
##        tweet=string.join(list_tweets_5[ind][0:-1]) ## finding features in negative tweets
##        for ind2 in range(0,10):
##            num=string.count(tweet,string.digits[ind2])
##            occur_neg=occur_neg+num
##            position = string.find(tweet,string.digits[ind2])
##            if position >= 0:
##                position2 = string.find(tweet,string.digits[ind2],position+1)
##                if position2 >= 0:
##                    feature_neg=feature_neg+1
##                    break
        

###################################################################################### for features related to length, words and alphabets.

    for ind in index_ambig:
        tweet=list_tweets_4[ind][0:-1]
        count=0
        for ind2 in range(0,len(tweet)):
            upper=0
            for ind3 in range(0,26):
                num=string.count(tweet[ind2],string.ascii_uppercase[ind3])
                upper=upper+num
            if upper == len(tweet[ind2]):
                count=count+1
        if count >= 2:
            feature_ambig=feature_ambig+1
                

    for ind in index_neu:
        tweet=list_tweets_4[ind][0:-1]
        count=0
        for ind2 in range(0,len(tweet)):
            upper=0
            for ind3 in range(0,26):
                num=string.count(tweet[ind2],string.ascii_uppercase[ind3])
                upper=upper+num
            if upper == len(tweet[ind2]):
                count=count+1
        if count >= 2:
            feature_neu=feature_neu+1
                

    for ind in index_pos:
        tweet=list_tweets_4[ind][0:-1]
        count=0
        for ind2 in range(0,len(tweet)):
            upper=0
            for ind3 in range(0,26):
                num=string.count(tweet[ind2],string.ascii_uppercase[ind3])
                upper=upper+num
            if upper == len(tweet[ind2]):
                count=count+1
        if count >= 2:
            feature_pos=feature_pos+1


    for ind in index_neg:
        tweet=list_tweets_4[ind][0:-1]
        count=0
        for ind2 in range(0,len(tweet)):
            upper=0
            for ind3 in range(0,26):
                num=string.count(tweet[ind2],string.ascii_uppercase[ind3])
                upper=upper+num
            if upper == len(tweet[ind2]):
                count=count+1
        if count >= 2:
            feature_neg=feature_neg+1

        

    
    

    percent_pos=float(feature_pos)/len(index_pos) ## percentage of tweets in which the feature occurs
    percent_neg=float(feature_neg)/len(index_neg)
    percent_neu=float(feature_neu)/len(index_neu)
    percent_ambig=float(feature_ambig)/len(index_ambig)


    result = [feature_pos,occur_pos,percent_pos,feature_neg,occur_neg,percent_neg,feature_neu,occur_neu,percent_neu,feature_ambig,occur_ambig,percent_ambig]

    return result
