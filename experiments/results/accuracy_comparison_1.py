def result(direc_hand,direc_auto):
    # directories of files containing hand labelled and automatically classified tweets

    import string

    direc_hand="D:\SEECS\Research & Projects\FYP\Codes\Results & Comaprison/jordansample_human.txt"
    direc_auto="D:\SEECS\Research & Projects\FYP\Codes\Results & Comaprison/jordan_combined_slang.txt"

    fa=open(direc_hand,"r")
    hand=fa.readlines()

    fb=open(direc_auto,"r")
    auto=fb.read()
    auto=eval(auto)

    
    index_pos=[]
    index_neg=[]
    index_neu=[]
    for i in range(0,len(hand)):
        if (string.find(hand[i],"00",-10) != -1):
            index_neu.append(i)
        if (string.find(hand[i],"+1",-10) != -1):
            index_pos.append(i)
        if (string.find(hand[i],"+2",-10) != -1):
            index_pos.append(i)
        if (string.find(hand[i],"-1",-10) != -1):
            index_neg.append(i)
        if (string.find(hand[i],"-2",-10) != -1):
            index_neg.append(i)


    #########################


    ## For Postivie
    threshold_pos=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

    
    CLASSIFIED_POS=[] ## Total number of tweets classified as positive by the machine, for each threshold level
    for thres in threshold_pos:
        classified_pos=0 
        for index in range(0,len(auto)):
            if(auto[index][-1]>thres):
                classified_pos+=1
        CLASSIFIED_POS.append(classified_pos)
        

    POS=[] ## Total number of positive tweets classified as positive
    for thres in threshold_pos:
        pos=0
        for index in index_pos:
            if (auto[index][-1]>thres):
                pos+=1
        POS.append(pos)

    
    F1P=[] ## F1 Measure: Combination of Percision and Recall
    ReP=[] ## Recall: Total no. of pos tweets classified as pos, divided by total no. of pos tweets. (Same as "true Positive").
    PrP=[] ## Percision Positice: Total no. of pos tweets classified as pos, divided by total no. of pos classified tweets.
    for index in range(0,len(POS)):
        if CLASSIFIED_POS[index] != 0:
            rep=float(POS[index])/len(index_pos)  
            prp=float(POS[index])/CLASSIFIED_POS[index]
            if rep!=0 or prp!=0:
                f1p=(2*rep*prp)/(rep+prp)
        ReP.append(rep)
        PrP.append(prp)
        F1P.append(f1p)
    

    HP=ReP[:] ## True Positive(It is equal to True Positive).


    FP=[] ## False Positive: Total no. non-pos tweets classified as pos, divided by total no. of non-pos tweets.
    for thres in threshold_pos:
        pos=0
        for index in index_neg+index_neu:
            if (auto[index][-1]>thres):
                pos+=1
            fp=float(pos)/(len(index_neg)+len(index_neu))
        FP.append(fp)
    



    ## For Negative
    threshold_neg=[0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7,-1.8,-1.9,-2.0]
    

    CLASSIFIED_NEG=[] ## Total number of tweets classified as negative by the machine, for each threshold level
    for thres in threshold_neg:
        classified_neg=0 
        for index in range(0,len(auto)):
            if(auto[index][-1]<thres):
                classified_neg+=1
        CLASSIFIED_NEG.append(classified_neg)
        

    NEG=[] ## Total number of negative tweets classified as negative
    for thres in threshold_neg:
        neg=0
        for index in index_neg:
            if (auto[index][-1]<thres):
                neg+=1
        NEG.append(neg)

    
    F1N=[] ## F1 Measure: Combination of Percision and Recall
    ReN=[] ## Recall: Total no. of negative tweets classified as pos, divided by total no. of negative tweets. (Same as "True Negative").
    PrN=[] ## Percision Positice: Total no. of neg tweets classified as pos, divided by total no. of neg classified tweets.
    for index in range(0,len(NEG)):
        if CLASSIFIED_NEG[index] != 0:
            ren=float(NEG[index])/len(index_neg)  
            prn=float(NEG[index])/CLASSIFIED_NEG[index]
            if ren!=0 or prn!=0:
                f1n=(2*ren*prn)/(ren+prn)
        ReN.append(ren)
        PrN.append(prn)
        F1N.append(f1n)
    

    HN=ReN[:] ## True Positive(It is equal to True Positive).


    FN=[]
    for thres in threshold_neg:
        neg=0
        for index in index_pos+index_neu:
            if (auto[index][-1]<thres):
                neg+=1
            fn=float(neg)/(len(index_pos)+len(index_neu))
        FN.append(fn)



    ## For neutral
    threshold_neu=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

    
    CLASSIFIED_NEU=[] ## Total number of tweets classified as negative by the machine, for each threshold level
    for thres in threshold_neu:
        classified_neu=0 
        for index in range(0,len(auto)):
            if(auto[index][-1]<=thres and auto[index][-1]>=-thres):
                classified_neu+=1
        CLASSIFIED_NEU.append(classified_neu)
        

    NEU=[] ## Total number of negative tweets classified as negative
    for thres in threshold_neu:
        neu=0
        for index in index_neu:
            if (auto[index][-1]<=thres and auto[index][-1]>=-thres):
                neu+=1
        NEU.append(neu)

    
    F10=[] ## F1 Measure: Combination of Percision and Recall
    Re0=[] ## Recall: Total no. of negative tweets classified as pos, divided by total no. of negative tweets. (Same as "True Negative").
    Pr0=[] ## Percision Positice: Total no. of neg tweets classified as pos, divided by total no. of neg classified tweets.
    for index in range(0,len(NEU)):
        if CLASSIFIED_NEU[index] != 0:
            re0=float(NEU[index])/len(index_neu)  
            pr0=float(NEU[index])/CLASSIFIED_NEU[index]
            if re0!=0 or pr0!=0:
                f10=(2*re0*pr0)/(re0+pr0)
        Re0.append(re0)
        Pr0.append(pr0)
        F10.append(f10)
    

    H0=Re0[:] ## True Positive(It is equal to True Positive).

    
    F0=[]
    for thres in threshold_neu:
        neu=0
        for index in index_pos+index_neg:
            if (auto[index][-1]<=thres and auto[index][-1]>=-thres):
                neu+=1
            h0=float(neu)/(len(index_pos)+len(index_neg))
        F0.append(h0)
        

    #####################


    ## For frequency distribution of calculated sentiments

##    sen_range=range(-205,215,10)##range of sentiments from -2 to +2
##    freq_dist=[0]*(len(sen_range)-1)
##                   
##
##    for i in range(0,len(sen_range)-1):
##        for index in range(0,len(auto)):
##            a=float(sen_range[i])
##            b=float(sen_range[i+1]);
##            check=auto[index][-1]
##            if check>=a/100 and check<b/100:
##                freq_dist[i]+=1
    
                   
    #####################
    
    return [F1P,ReP,PrP,HP,FP,F1N,ReN,PrN,HN,FN,F10,Re0,Pr0,H0,F0]
                
                

                
        
        

    
