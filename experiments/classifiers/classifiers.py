def tweet_formatting(directory):
##formats tweet for classification
##directory to the txt file containing a list of tweet texts (just a raw list with RT's removed)

    import nltk

    directory="D:\SEECS\Research & Projects\FYP\Codes\classifiers/jordansample_raw.txt"
    
    fp=open(directory,"r")
    tweetlist=fp.read()
    
    tweetlist=tweetlist.lower() ## converting from capital letters to lowercase
    
    tweetlist=eval(tweetlist) ## makes the data in list form

    for index in range(0,len(tweetlist)):
        tweetlist[index]=nltk.word_tokenize(tweetlist[index]) ##tokenization, also takes care of apostrophes

        porter=nltk.PorterStemmer() ##for porter stemming
        ind=0
        while(ind<len(tweetlist[index])):
            tweetlist[index][ind]=porter.stem(tweetlist[index][ind])
            
##            itr=ind+1
##            while(itr<len(tweetlist[index])): ##for removing repetitive words and characters after tokenization (like "!")(might remove emphasis)
##                if(tweetlist[index][itr]==tweetlist[index][ind]):
##                    del tweetlist[index][itr]
##                    itr-=1
##                itr+=1
            ind+=1


    return tweetlist

    ## should also work for removing some unwanted items like non-sensical short word tokens, blanks, brackets, colons, comas, periods, hyphens, apostrophe s etc
    ## these could be catered by removing any token of less than two characters unless it is some specific word like "no","ya","bi","hi","ah"
    ## this would be done at the location of removal of repetitive words

        
            
def simple_classifier(tweetlist,dict_directory):

    ##Here tweetlist is the output of the above function
    ##dic_directory is the directory of the prior polarity dictionary

    tweetlist=tweet_formatting(1) ## formatted list from above code
    tweetlist_raw_direc="D:\SEECS\Research & Projects\FYP\Codes\classifiers/jordansample_raw.txt" ## original unformatted list (for emoticon comparison)

##    dict_directory="D:\SEECS\Research & Projects\FYP\Codes\classifiers/dict_mpqa_stem_1.txt"
##    dict_directory="D:\SEECS\Research & Projects\FYP\Codes\classifiers/dict_harvard_stem_1.txt" ## for this dictionary you can include strong/weak subjectivity or not.
    dict_directory="D:\SEECS\Research & Projects\FYP\Codes\classifiers/dict_combined_stem.txt" ## combined dictionary of mpqa and harvard

    dict_slang_direc="D:\SEECS\Research & Projects\FYP\Codes\classifiers/dict_slang.txt" ## dictionary contains polar (sentiment expressing) internet slangs.
    dict_emoticon_direc="D:\SEECS\Research & Projects\FYP\Codes\classifiers/dict_emoticon.txt" ## dictionary contains emoticons
    

    fp=open(dict_directory,"r") ## for word dictionary
    dictionary=fp.read()
    dictionary=eval(dictionary)

    fq=open(dict_slang_direc,"r") ## for slang dictionary
    dictionary_slang=fq.read()
    dictionary_slang=eval(dictionary_slang)

    fr=open(dict_emoticon_direc,"r") ## for emoticon dictionary
    dictionary_emoticon=fr.read()
    dictionary_emoticon=eval(dictionary_emoticon)

    fs=open(tweetlist_raw_direc,"r") ## for emoticon comparison
    tweetlist_raw=fs.read()
    tweetlist_raw=eval(tweetlist_raw)


    for index in range(0,len(tweetlist)):
        words=0
        polarity=0

        for key in dictionary.keys():
            for ind in range(0,len(tweetlist[index])):
                if tweetlist[index][ind]==key and key!=0: ## The key!=0 part removes the effect of non-polar words in harvard dictionary
                    polarity=polarity+dictionary[key]
                    words+=1

        for key_1 in dictionary_slang.keys():
            for ind_1 in range(0,len(tweetlist[index])):
                if tweetlist[index][ind_1]==key_1 and key_1!=0:
                    polarity=polarity+dictionary_slang[key_1]
                    words+=1

        for key_2 in dictionary_emoticon.keys():
            if tweetlist_raw[index].find(key_2) != -1:
                polarity=polarity+dictionary_emoticon[key_2]
                words+=1

        polarity=float(polarity)
        if (words != 0):
            polarity=polarity/words
        tweetlist[index].append(polarity)
    


    return tweetlist ##contains polarity in last element of each list, where each list is an individual tweet
    


    
