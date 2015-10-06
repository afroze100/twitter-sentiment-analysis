def search_tweets(query):

    import urllib
    import json

    query=str(query)
    num=1000
    

    search=urllib.urlopen("http://search.twitter.com/search.json?q="+query+"&rpp="+str(num))
    data=json.loads(search.read())

    x=1
    for result in data["results"]:
        if result["iso_language_code"]=="en":
            print x, result["created_at"], result["text"],"\n"
            x+=1
