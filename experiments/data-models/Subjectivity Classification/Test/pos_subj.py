
def pos_subj():

    import nltk
    import string

    file_tweets=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification/tweets_subj_5261.txt","r")
    file_labels=open("D:\SEECS\Research & Projects\FYP\Codes\Data Models\Subjectivity Classification/labels_subj_5261.txt","r")

    list_tweets=file_tweets.readlines()
    list_labels=file_labels.read().splitlines()


    len_pos=2543
    len_neg=1877
    len_amb=841


    list_tweets_2=[] ## simple tokenized list
    list_tweets_3=[] ## tokenized list with removed @'s and url's (word_tokenize is used for accurate results with POS tagging)


    posdict_pos={} ## dictionary mapping POS counts (on average) in objective (and subjective)
    posdict_neg={}
    posdict_amb={}
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

        if list_labels[index]=="1":
            tokens_pos=list_tweets_3[index]
            pos=nltk.pos_tag(tokens_pos)
            for token in pos:
                if token[1] in posdict_pos:
                    posdict_pos[token[1]]+=1
                else:
                    posdict_pos[token[1]]=1

        if list_labels[index]=="-1":
            tokens_neg=list_tweets_3[index]
            pos=nltk.pos_tag(tokens_neg)
            for token in pos:
                if token[1] in posdict_neg:
                    posdict_neg[token[1]]+=1
                else:
                    posdict_neg[token[1]]=1

        if list_labels[index]=="??":
            tokens_amb=list_tweets_3[index]
            pos=nltk.pos_tag(tokens_amb)
            for token in pos:
                if token[1] in posdict_amb:
                    posdict_amb[token[1]]+=1
                else:
                    posdict_amb[token[1]]=1



    for pos_tag in posdict_pos.keys():  ## normalizing the POS tags for average count per tweet
        posdict_pos[pos_tag]=float(posdict_pos[pos_tag])/len_pos

    for pos_tag in posdict_neg.keys():
        posdict_neg[pos_tag]=float(posdict_neg[pos_tag])/len_neg

    for pos_tag in posdict_amb.keys():
        posdict_amb[pos_tag]=float(posdict_amb[pos_tag])/len_amb




    for pos_tag in posdict_pos.keys():  ## for calculating the total probability count of occurence of a certain POS
        if (pos_tag in posdict_neg.keys()) and (pos_tag in posdict_amb.keys()):
            posdict_total[pos_tag]=((float(posdict_pos[pos_tag])*len_pos)+(float(posdict_neg[pos_tag])*len_neg)+(float(posdict_amb[pos_tag])*len_amb))/(len_pos+len_neg+len_amb)
        if (pos_tag in posdict_neg.keys()) and (pos_tag not in posdict_amb.keys()):
            posdict_total[pos_tag]=((float(posdict_pos[pos_tag])*len_pos)+(float(posdict_neg[pos_tag])*len_neg))/(len_pos+len_neg)
        if (pos_tag not in posdict_neg.keys()) and (pos_tag in posdict_amb.keys()):
            posdict_total[pos_tag]=((float(posdict_pos[pos_tag])*len_pos)+(float(posdict_amb[pos_tag])*len_amb))/(len_pos+len_amb)

    for pos_tag in posdict_amb.keys():
        if (pos_tag not in posdict_pos.keys()) and (pos_tag not in posdict_neg.keys()):
            posdict_total[pos_tag]=float(posdict_amb[pos_tag])
        if (pos_tag not in posdict_pos.keys()) and (pos_tag in posdict_neg.keys()):
            posdict_total[pos_tag]=((float(posdict_amb[pos_tag])*len_amb)+(float(posdict_neg[pos_tag])*len_neg))/(len_amb+len_neg)
    




    for pos_tag in posdict_pos.keys():   ## for calculating the effective ratio to determine which pos tags do a good job in differentiating b/w sentiments
        if posdict_pos[pos_tag]>(posdict_neg.get((pos_tag),0)+posdict_amb.get((pos_tag),0)):
            posdict_ratio[pos_tag] = (posdict_pos[pos_tag]-posdict_neg.get((pos_tag),0)-posdict_amb.get((pos_tag),0))/float(posdict_neg.get((pos_tag),0)+posdict_amb.get((pos_tag),0))  
            posdict_ratio[pos_tag]="pos,"+str(posdict_ratio[pos_tag])

    for pos_tag in posdict_neg.keys():  
        if posdict_neg[pos_tag]>(posdict_pos.get((pos_tag),0)+posdict_amb.get((pos_tag),0)):
            posdict_ratio[pos_tag] = (posdict_neg[pos_tag]-posdict_pos.get((pos_tag),0)-posdict_amb.get((pos_tag),0))/float(posdict_pos.get((pos_tag),0)+posdict_amb.get((pos_tag),0))  
            posdict_ratio[pos_tag]="neg,"+str(posdict_ratio[pos_tag])

    for pos_tag in posdict_amb.keys():  
        if posdict_amb[pos_tag]>(posdict_pos.get((pos_tag),0)+posdict_neg.get((pos_tag),0)):
            posdict_ratio[pos_tag] = (posdict_amb[pos_tag]-posdict_pos.get((pos_tag),0)-posdict_neg.get((pos_tag),0))/float(posdict_pos.get((pos_tag),0)+posdict_neg.get((pos_tag),0))  
            posdict_ratio[pos_tag]="amb,"+str(posdict_ratio[pos_tag])            
                                                                    


    return posdict_total,posdict_ratio,list_tweets_3,posdict_pos,posdict_neg,posdict_amb
