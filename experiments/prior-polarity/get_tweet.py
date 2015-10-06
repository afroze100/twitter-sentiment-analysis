def get_tweet(path):
    """give path of the text file as a string"""

    import string
    
    file1=open(path,"r")
    tweet_list=file1.readlines()

    my_dict={}

    for index1 in range(0:len(tweet_list)):
        tweet_list=tweet_list[0:-2]

    
    
