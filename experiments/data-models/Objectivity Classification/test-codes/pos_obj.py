
def pos_obj_obj():

    import nltk
    import string

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/tweets_9858.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Objectivity Classification/labels_obj_9858.txt","r")

    list_tweets=file_tweets.readlines()
    list_labels=file_labels.readlines()


    #len_subj=5154
    #len_obj=4633

    len_subj=6061
    len_obj=3797


    list_tweets_2=[] ## simple tokenized list
    list_tweets_3=[] ## tokenized list with removed @'s and url's (word_tokenize is used for accurate results with POS tagging)


    posdict_obj={} ## dictionary mapping POS counts (on average) in objective (and subjective)
    posdict_subj={}
    posdict_total={} ##dictionry mapping probability of occurence of the particular pos tag in any given tweet

    posdict_ratio={} ## dictionary containing ratio of occurence of each POS tag according to sentiment


    for index in range(0,len(list_tweets)):
        list_tweets_2.append(nltk.word_tokenize((list_tweets[index])))  ## for populating list_tweets_3

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

        ##list_tweets_3[index]=nltk.wordpunct_tokenize(list_tweets_3[index])



###############################
        

    for index in range(0,len(list_tweets)): ## couting the total POS tags

        if list_labels[index]=="s\n":
            tokens_subj=list_tweets_3[index]
            pos=nltk.pos_tag(tokens_subj)
            for token in pos:
                if token[1] in posdict_subj:
                    posdict_subj[token[1]]+=1
                else:
                    posdict_subj[token[1]]=1

        if list_labels[index]=="0\n":
            tokens_obj=list_tweets_3[index]
            pos=nltk.pos_tag(tokens_obj)
            for token in pos:
                if token[1] in posdict_obj:
                    posdict_obj[token[1]]+=1
                else:
                    posdict_obj[token[1]]=1



    for pos_tag in posdict_subj.keys():  ## normalizing the POS tags for average count per tweet
        if posdict_subj[pos_tag] < 50:
            posdict_subj[pos_tag]= 'x'
        else:
            posdict_subj[pos_tag]=float(posdict_subj[pos_tag])/len_subj

    for pos_tag in posdict_obj.keys():
        if posdict_obj[pos_tag] < 50:
            posdict_obj[pos_tag]= 'x'
        else:
            posdict_obj[pos_tag]=float(posdict_obj[pos_tag])/len_obj


    for pos_tag in posdict_subj.keys():
        if posdict_subj[pos_tag]!='x'and posdict_obj[pos_tag]!='x':
            posdict_total[pos_tag]=((float(posdict_subj[pos_tag])*len_subj)+(float(posdict_obj[pos_tag])*len_obj))/(len_subj+len_obj)




    for pos_tag in posdict_subj.keys():   ## for calculating the effective ratio to determine which pos tags do a good job in differentiating b/w sentiments
        if (pos_tag in posdict_obj.keys()) and (posdict_subj[pos_tag]!='x') and (posdict_obj[pos_tag]!='x'):

            if posdict_subj[pos_tag]>= posdict_obj[pos_tag]:
                posdict_ratio[pos_tag] = (posdict_subj[pos_tag]-posdict_obj[pos_tag])/float(posdict_obj[pos_tag])  ##subjective are positve and objective negative

            else:
                posdict_ratio[pos_tag] = (posdict_subj[pos_tag]-posdict_obj[pos_tag])/float(posdict_subj[pos_tag])
            
                                                                    


    return [posdict_total,posdict_ratio,list_tweets_4]
