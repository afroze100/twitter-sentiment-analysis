##Code for downloading tweet stream using Twitter API

def tweetstream(directory,threshold):

    directory="D:\SEECS\Research & Projects\FYP\Resources\Raw Resources\Tweets/sample_2_29dec.txt"

    f=open(directory,"w")

    import tweetstream

    with tweetstream.SampleStream("iAfroze","nobody") as stream:
        for tweet in stream:
	    if (len(str(tweet))>1000):
		print "tweet %d" % (stream.count)
		f.write(str(tweet))
		if (stream.count == 12000):
		    break
		else:
                    f.write("\n")

