def harvard_prior_1(): 

## Look into difference between Negativ/Positiv and Neg/Pos!

    import string
    import nltk
    
    f=open("D:\SEECS\Research & Projects\FYP\Codes\Priorpolarity\harvard_prior_prob.txt","r")
##    f=open("D:\SEECS\Research & Projects\FYP\Codes\Priorpolarity/afroze.txt","r")

    text=f.read()
    text=text.lower() ##to make text into lowercase (from capital letters)
    textline=string.split(text,"\n") ##make a list of strings, with each string showing a new line from original text

    index=0
    textlinelist=textline ##to make the dimensions similar for next operation

    for str1 in textline: ##to make a list of lists where each sublist is a list of words of one particular line
        textlinelist[index]=string.split(str1)
        index=index+1

    dict_raw={}

##    for list1 in textlinelist: ##method 1 for making a dictionary
##        a=list1[0]
##        str1=string.join(list1)
##        if string.find(str1,"Positiv")!=-1:
##            b=1
##        elif string.find(str1,"Negativ")!=-1:
##            b=-1
##        else:
##            b=0
##        if string.find(str1,"Strong")!=-1:
##            b=2*b
##        elif string.find(str1,"Weak")!=-1:
##            b=0.5*b
##        word_dict[a]=b
##
##    return word_dict
    

    polarity=0
    for list1 in textlinelist: ##method 2 (may be more effecient)
        word=list1[0]

        i=string.find(word,"#") ##to remove redundant words
        if i!=-1:
            word=word[0:i]
        
        for index in range(1,len(list1)): ##start from 1, since the 0th element is alwasy the word name
            if list1[index]=="positiv":
                polarity=1
            elif list1[index]=="negativ":
                polarity=-1
            if list1[index]=="strong":
                polarity=2*polarity
            elif list1[index]=="weak":
                polarity=0.5*polarity
        dict_raw[word]=polarity
        polarity=0


    sorted_list=dict_raw.items() ##to return the dictionary in the form of a list in alphabetical order
    sorted_list.sort()


    dict_porterstem={} ##to generate a porter stemmed version of the dictionary
    key_list=dict_raw.keys()
    porter=nltk.PorterStemmer()

    for words in key_list:
        dict_porterstem[porter.stem(words)]=dict_raw[words]
    
     
    return [dict_raw,sorted_list,dict_porterstem]



##########################################################################################################################################################


def mpqa_prior_1():


    import string
    import nltk

    f=open("D:\SEECS\Research & Projects\FYP\Codes\Priorpolarity/mpqa_prior_prob.txt","r")

    text=f.read()
    text=text.lower()
    textline=string.split(text,"type=")

    
    del(textline[0])
##    del(textline[-1])
    
    index=0
    for str1 in textline:
        textline[index]=string.split(str1)
        index=index+1
        
    dict_raw={}
    for list1 in textline:
        word=""
        inter=string.join(list1)
        index_begin=6+(string.find(inter,"word1="))

        while inter[index_begin]!=" ":
            word=word+inter[index_begin]
            index_begin=index_begin+1

        if list1[5]=="priorpolarity=negative":
            polarity=-1
        else:
            polarity =1
        if list1[0]=="strongsubj":
            polarity=polarity*2

        dict_raw[word]=polarity


    sorted_list=dict_raw.items()
    sorted_list.sort()


    dict_porterstem={}
    key_list=dict_raw.keys()
    porter=nltk.PorterStemmer()

    for words in key_list:
        dict_porterstem[porter.stem(words)]=dict_raw[words]

    return [dict_raw,sorted_list,dict_porterstem]

        
        
        
        
        
                                   
                                   
        

    

