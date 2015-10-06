import string
import cgi
import json
import re
import operator
import math
import dictionary
import requests

from google.appengine.api import urlfetch
from stem import porter
from requests_oauthlib import OAuth1



def getpasttweets(query):

	consumer_key = 'GfEjuaWkfEAUGiWKFKaPuw'
	consumer_secret = 'Oc4qvRO3Eum737h4EwE4BG3cwgxqE9C9ZRZurOcp8'
	token_key = '365093750-sO52uWHGyh8QPAbADofAUVSoE6njgdWULvZVThng'
	token_secret = 'tLH8qiaejBBgtDFrKhqO0pckrjVGEDYzWTuQpYJZgw'

	auth=OAuth1(consumer_key,consumer_secret,token_key,token_secret);

	query = query.replace(" ", "%20")
	query = cgi.escape(query)

	list_tweets=[]
	url = 'https://api.twitter.com/1.1/search/tweets.json?q='+query+'&count=100'

	try:
		search = requests.get(url, auth=auth)
	except requests.ConnectionError:
		return [-1, -1, -1]
	except requests.Timeout:
		return [-1, -1, -1]
	except requests.HTTPError:
		return [-1, -1, -1]

	data = json.loads(search.content)

	if "errors" in data.keys():
		if data['errors'][0]['message'] == "Rate limit exceeded":
			return [-2,-2,-2]
		else:
			return [-3,-3,-3]
	
	if data["statuses"] == []:
		return [-4, -4, -4]
	
	else:
		total_results = len(data["statuses"])
		time_start = data["statuses"][0]["created_at"]
		
		for index in range(0,len(data["statuses"])):
			result = data["statuses"][index]
			
			if result["lang"] == "en":
				list_tweets.append(result["text"])
	
		maximum = data["search_metadata"]["max_id"]
	
		for iter in range(4):
		
			if "next_results" in data["search_metadata"]:
				next = data["search_metadata"]["next_results"]
				url_next = 'https://api.twitter.com/1.1/search/tweets.json'+next

				try:
					search = requests.get(url_next, auth=auth)
				except requests.ConnectionError:
					
					try:
						search = requests.get(url_next, auth=auth)
					except requests.ConnectionError:
						
						time_end = data["statuses"][-1]["created_at"]
						time_search = timedifference(time_start, time_end)
						return [list_tweets, total_results, time_search]

				data = json.loads(search.content)
				total_results = total_results + len(data["statuses"])
			
				if data["statuses"] == [] or data["statuses"][0]["id"] >= maximum:
					time_end = data["statuses"][-1]["created_at"]
					time_search = timedifference(time_start, time_end)

					return [list_tweets, total_results, time_search]
			
				else:
					for ind in range(0,len(data["statuses"])):
						result = data["statuses"][ind]	

						if result["lang"] == "en":
							list_tweets.append(result["text"])
			else:
				break

		time_end = data["statuses"][-1]["created_at"]
		time_search = timedifference(time_start, time_end)

		return [list_tweets, total_results, time_search]



def tweetformat(list_tweets):

	list_tweets_5 = list_tweets[:]
	pattern_remove = '[!"#$%&\'()*+,-./:;<=>?@[\]\\\\^_`{|}~0123456789 ]'
	stemmer = porter.PorterStemmer()

	for index in range(0,len(list_tweets_5)):
		tweet = list_tweets_5[index]
		tweet = string.lower(tweet)			## for lowercase conversion of capital letters
		tweet = string.split(tweet)
		i = 0
		while i < len(tweet):
			if tweet[i][0] =="@" or re.search("http://",tweet[i]) or tweet[i]=="rt":	## for removing @'s url's and RT's
				del tweet[i]
			else:
				i+=1
		list_tweets_5[index] = string.join(tweet)
		list_tweets_5[index] = re.split(pattern_remove, list_tweets_5[index])		## for removing punctuations and digits
		for ind in range(0,len(list_tweets_5[index])):
			list_tweets_5[index][ind] = stemmer.stem(list_tweets_5[index][ind])

		list_tweets_5[index] = string.join(list_tweets_5[index])
		list_tweets_5[index] = string.split(list_tweets_5[index])

	return list_tweets_5



def getfeatures(list_tweets, list_tweets_5):

	dict_obj = dictionary.dict_obj
	dict_subj = dictionary.dict_subj

	features = []				## final list containing all feature info corresponding to each tweet

	presence_url = []
	presence_emoticon = []			## objectivity features
	num_exclamation = []
	words_obj =[]

	emoticon_pos = []			## subjectivity features
	emoticon_neg = []
	emoticon_score = []
	words_subj = []

	pos_emoticon_re="[:;=xX8]-?[)\]}DpP3B*]([^A-Za-z1-9]|$)| [Cc(\[{]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;3|&gt;-?[DPp3]([^A-Za-z1-9]|$)|[*^]([_-]+|\.,)[*^]|\^\^| n([_-]+|\.,)n"
 	neg_emoticon_re="[:;=xX8]['\"]?-?['\"]?[(\[{\\\\/\|@cCLsS]([^A-Za-z1-9/]|$)| [\])}/\\\\D]-?[:;=xX8]([^A-Za-z1-9]|$)|&lt;[/\\\\]3|([^A-Za-z1-9]|$)T(_+|\.)T|([^A-Za-z1-9]|$)[xX](_+|\.)[xX]|([^A-Za-z1-9]|$)@(_+|\.)@|([^A-Za-z1-9]|$)[uU](_+|\.)[uU]|([^A-Za-z1-9]|$)[oO](_+|\.)[oO]|([^A-Za-z1-9]|$)i(_+|\.)i|(&gt;|&lt;)[_\.,-]+(&gt;|&lt;)|[-=](_+|\.)[-=]|\._+\."

 	
 	for index in range(0,len(list_tweets)): 		
 		if string.find(list_tweets[index], "http://") == -1:			## for presence_url
 			presence_url.append("no")
 		else:
 			presence_url.append("yes")

 		num_exclamation.append(string.count(list_tweets[index], "!"))		## for num_exclamation

 		num_emot_pos = len(re.findall(pos_emoticon_re, list_tweets[index]))
 		num_emot_neg = len(re.findall(neg_emoticon_re, list_tweets[index]))
 		if num_emot_pos == 0 and num_emot_neg == 0:
 			presence_emoticon.append("no")							
 		else:
 			presence_emoticon.append("yes")							## for presence_emoticon
 		emoticon_pos.append(num_emot_pos)							## for emoticon_pos
 		emoticon_neg.append(num_emot_neg)							## for emoticon_neg
 		emoticon_score.append(num_emot_pos - num_emot_neg)			## for emoticon_score


 	for index in range(0,len(list_tweets_5)):
 		obj = 1.0
 		subj = 1.0
 		pos = 1.0
 		neg = 1.0
 		for word in list_tweets_5[index]:
 			if word in dict_obj:
 				obj = float(obj)*dict_obj[word][0]
 				subj = float(subj)*dict_obj[word][1]
 			if word in dict_subj:
 				pos = float(pos)*dict_subj[word][0]
 				neg = float(neg)*dict_subj[word][1]
 		words_obj.append(obj/float(obj+subj))
 		words_subj.append(pos/float(pos+neg))


 	for index in range(0,len(list_tweets)):
 		features.append([presence_url[index], presence_emoticon[index], num_exclamation[index], words_obj[index], emoticon_pos[index], emoticon_neg[index], emoticon_score[index], words_subj[index]])


	return features



def classification(features):
	features_final = []			## contains score_obj &score_subj
	list_labels = []			## contains label assigned to each tweet
	num_lables = []				## cnotains the total number of positive, negative and neutral labels

	num_pos = 0
	num_neg = 0
	num_neu = 0

	for index in range(0,len(features)):
		feat_obj = [0]*4
		feat_subj = [0]*4
		feat_pos = [0]*4
		feat_neg = [0]*4

		if features[index][0] == "yes":				## for presence_url
			feat_obj[0] = 0.2404
			feat_subj[0] = 0.0801 #0.0355
		else:
			feat_obj[0] = 0.7596
			feat_subj[0] = 0.9199 #0.9645

		if features[index][1] == "yes":				## for presence_emoticon
			feat_obj[1] = 0.0308
			feat_subj[1] = 0.2346
		else:
			feat_obj[1] = 0.9692
			feat_subj[1] = 0.7654

		if features[index][2] == 0:					## for num_exclamation
			feat_obj[2] = 0.8493
			feat_subj[2] = 0.7192
		elif features[index][2] == 1:
			feat_obj[2] = 0.1057
			feat_subj[2] = 0.1605
		elif features[index][2] == 2:
			feat_obj[2] = 0.0247
			feat_subj[2] = 0.0558
		elif features[index][2] == 3:
			feat_obj[2] = 0.0105
			feat_subj[2] = 0.0289
		elif features[index][2] == 4 or features[index][2] == 5:
			feat_obj[2] = 0.00806
			feat_subj[2] = 0.0216
		elif features[index][2] >= 6 and features[index][2] <= 9:
			feat_obj[2] = 0.00147
			feat_subj[2] = 0.0118
		else:
			feat_obj[2] = 0.000244
			feat_subj[2] = 0.002

		feat_obj[3] = features[index][3]			## for words_obj
		feat_subj[3] = 1-features[index][3]

		if features[index][4] == 0:					## for emoticon_pos
			feat_pos[0] = 0.7102
			feat_neg[0] = 0.9906					
		elif features[index][4] == 1:
			feat_pos[0] = 0.2574
			feat_neg[0] = 0.0088
		else:
			feat_pos[0] = 0.0347
			feat_neg[0] = 0.000589

		if features[index][5] == 0:					## for emoticon_neg
			feat_pos[1] = 0.984
			feat_neg[1] = 0.870					
		elif features[index][5] == 1:
			feat_pos[1] = 0.015
			feat_neg[1] = 0.1257
		else:
			feat_pos[1] = 0.00044
			feat_neg[1] = 0.00413

		if features[index][6] == 0:					## for emoticon_score
			feat_pos[2] = 0.7038
			feat_neg[2] = 0.8622				
		elif features[index][6] == 1:
			feat_pos[2] = 0.2550
			feat_neg[2] = 0.0076
		elif features[index][6] >= 2:
			feat_pos[2] = 0.0306
			feat_neg[2] = 0.000586
		elif features[index][6] == -1:
			feat_pos[2] = 0.0101
			feat_neg[2] = 0.1248
		else:
			feat_pos[2] = 0.00044
			feat_neg[2] = 0.0047

		feat_pos[3] = features[index][7]			## for words_subj
		feat_neg[3] = 1-features[index][7]

		score_obj = feat_obj[0]*feat_obj[1]*feat_obj[2]*feat_obj[3]
		score_subj = feat_subj[0]*feat_subj[1]*feat_subj[2]*feat_subj[3]
		final_score_obj =  float(score_obj)/(score_obj+score_subj)
		
		score_pos = feat_pos[0]*feat_pos[1]*feat_pos[2]*feat_pos[3]
		score_neg = feat_neg[0]*feat_neg[1]*feat_neg[2]*feat_neg[3]
		final_score_subj = float(score_pos)/(score_pos+score_neg)

		features_final.append([final_score_obj, final_score_subj])

		
		# if final_score_obj <= 0.25:			## kmeans clustering parameters
		# 	if final_score_subj > 0.5:
		# 		list_labels.append("1")
		# 	else:
		# 		list_labels.append("-1")
		# elif (final_score_subj >= final_score_obj + 0.25):
		# 	list_labels.append("1")
		# elif (final_score_subj <= -(final_score_obj) + 0.75):
		# 	list_labels.append("-1")
		# else:
		# 	list_labels.append("0")


		if final_score_obj <= 0.5:			## kmeans clustering parameters
			if final_score_subj > 0.5:
				list_labels.append("1")
			else:
				list_labels.append("-1")
		elif (final_score_subj >= (2*final_score_obj) - 0.5):
			list_labels.append("1")
		elif (final_score_subj <= (-2*final_score_obj) + 1.5):
			list_labels.append("-1")
		else:
			list_labels.append("0")


	for label in list_labels:
		if label == "0":
			num_neu+=1
		elif label == "1":
			num_pos+=1
		else:
			num_neg+=1


	num_labels = [num_pos, num_neg, num_neu]		

	return [features_final, list_labels, num_labels]



def selecttweets(list_tweets, list_labels, features_final, num_labels):
	dict_subj = {}
	dict_obj = {}

	list_tweets_positive = []
	list_tweets_negative = []
	list_tweets_neutral = []

	if num_labels[0] > 10 or num_labels[1] > 10: ## if more than 10 positive or negative labels

		for i in range(0,len(list_labels)):
			if list_labels[i] != "0":
				score_subj = (features_final[i][0]-0.5)*(features_final[i][1]-0.5)
				dict_subj[i] = score_subj
			else:
				score_obj = features_final[i][0]*(0.5+(0.5-abs(features_final[i][1]-0.5)))
				dict_obj[i] = score_obj
		sorted_dict_subj = sorted(dict_subj.iteritems(), key=operator.itemgetter(1))
		sorted_dict_obj = sorted(dict_obj.iteritems(), key=operator.itemgetter(1), reverse=True)
		
		previous_value = 1
		ind = 0
		index = 0
		while index < 10:
			if sorted_dict_subj[ind][1] != previous_value and sorted_dict_subj[ind][1] < 0:
				index_pos = sorted_dict_subj[ind][0]
				list_tweets_positive.append(list_tweets[index_pos])
				previous_value = sorted_dict_subj[ind][1]
				ind+=1
				index+=1
			else:
				ind+=1
			if ind == len(sorted_dict_subj):
				break

		previous_value = 0
		ind = 0
		index = 0
		while index < 10:
			if sorted_dict_subj[-ind-1][1] != previous_value and sorted_dict_subj[-ind-1][1] > 0:
				index_neg = sorted_dict_subj[-ind-1][0]
				list_tweets_negative.append(list_tweets[index_neg])
				previous_value = sorted_dict_subj[-ind-1][1]
				ind+=1
				index+=1
			else:
				ind+=1
			if ind == len(sorted_dict_subj):
				break
	
	else:
		for i in range(0,len(list_labels)):
			if list_labels[i] == "1":
				list_tweets_positive.append(list_tweets[i])
			elif list_labels[i] == "-1":
				list_tweets_negative.append(list_tweets[i])
			else:
				score_obj = features_final[i][0]*(0.5+(0.5-abs(features_final[i][1]-0.5)))
				dict_obj[i] = score_obj
		sorted_dict_obj = sorted(dict_obj.iteritems(), key=operator.itemgetter(1), reverse=True)

	if len(sorted_dict_obj) > 10:
		previous_value = 0
		ind = 0
		index = 0
		while index < 10:
			if sorted_dict_obj[ind][1] != previous_value:
				index_neu = sorted_dict_obj[ind][0]
				list_tweets_neutral.append(list_tweets[index_neu])
				previous_value = sorted_dict_obj[ind][1]
				ind+=1
				index+=1
			else:
				ind+=1
			if ind == len(sorted_dict_obj):
				break
	else:
		for ind in range(0,len(sorted_dict_obj)):
			index_neu = sorted_dict_obj[ind][0]
			list_tweets_neutral.append(list_tweets[index_neu])


	return [list_tweets_positive, list_tweets_negative, list_tweets_neutral]
		


def popularityscore(num_labels, total_results, time_search):
	min_tweets = 25
	overall_mean = 0.55
	
	num_pos = num_labels[0]
	num_neg = num_labels[1]

	if num_pos == 0 and num_neg == 0:
		score = 0

	else:
		if num_pos == num_neg:
			num_pos += 1
		score = float(num_pos)/(num_pos + num_neg)
		score = 2*(score - 0.5)

	# forward_max = float(1 - abs(score))/(3 - 4*((abs(score) - 0.5)**2))
	# backward_max = float(abs(score))/(3 - 4*((abs(score) - 0.5)**2))

	forward_max = float(1 - abs(score))/(2 + (8*(math.exp(-8*abs(score)))))
	backward_max = float(abs(score))/(2 + (8*(math.exp(-8*abs(score)))))

	factor = float(total_results)/time_search
	factor = math.log10(factor)
	if factor > 2:
		factor = 2
	elif factor < -2:
		factor = -2

	if score >= 0:
		if factor > 0:
			popularity_score = score + (float(factor)/2)*forward_max
		else:
			popularity_score = score + (float(factor)/2)*backward_max
	else:
		if factor > 0:
			popularity_score = score - (float(factor)/2)*forward_max
		else:
			popularity_score = score - (float(factor)/2)*backward_max

	popularity_score = 100*popularity_score
	popularity_score = "%.1f" %popularity_score

	return popularity_score



def timedifference(time_start, time_end):

	time_start = re.split("[ :]", time_start)
	time_end = re.split("[ :]", time_end)

	if int(time_start[2]) > int(time_end[2]):
		if time_start[1] == "Feb":
			days = 29 - (int(time_start[2]) - int(time_end[2]))
		elif time_start[1] == "Apr" or time_start[1] == "Jun" or time_start[1] == "Sep" or time_start[1] == "Nov":
			days = 30 - (int(time_start[2]) - int(time_end[2]))
		else:
			days = 31 - (int(time_start[2]) - int(time_end[2]))
	else:
		days = int(time_start[2]) - int(time_end[2])

	if int(time_start[4]) > int(time_end[4]):
		hours = 24 - (int(time_start[4]) - int(time_end[4]))
		days-=1
	else:
		hours = (int(time_start[4]) - int(time_end[4]))

	if int(time_start[5]) > int(time_end[5]):
		minutes = 60 - (int(time_start[5]) - int(time_end[5]))
		hours-=1
	else:
		minutes = (int(time_start[5]) - int(time_end[5]))

	time_lapsed = (24*days) + (60*hours) + minutes

	if time_lapsed == 0:
		time_lapsed += 1
	elif time_lapsed < 0:
		time_lapsed = abs(time_lapsed)

	return time_lapsed

