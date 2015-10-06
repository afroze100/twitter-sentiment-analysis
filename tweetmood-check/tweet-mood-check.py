import webapp2
import jinja2
import os
import string
import algorithm
import json

from google.appengine.ext import db


template_dir = os.path.dirname(__file__)
jinja_env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir), autoescape = True)



class Handler(webapp2.RequestHandler):

    def write(self, *a, **kw):
        self.response.out.write(*a, **kw)
    
    def render_str(self, template, **params):
        t = jinja_env.get_template(template)
        return t.render(**params)

    def render(self, template, **params):
        self.response.out.write(self.render_str(template, **params))



class MainPage(Handler):

    def get(self):
        self.render('MainPage.html')



class ContactPage(Handler):

    def get(self):
        self.render('ContactPage.html')



class AboutPage(Handler):

    def get(self):
        self.render('AboutPage.html')



class AdditionalsPage(Handler):

    def get(self):
        self.render('AdditionalsPage.html')



class TweetScoreMain(Handler):
    
    def get(self): 
        self.render('TweetScoreMain.html')



class TweetComparisonMain(Handler):

    def get(self):
        self.render('TweetComparisonMain.html')



class TweetStatsMain(Handler):

    def get(self):
        global query_cron_1

        self.render('TweetStatsMain.html', error = "ATTENTION: Due to severe rate limitations of new Twitter API, TweetStats is CURRENTLY OFFLINE. The data showed is past data (uptil March 2013) and is not being updated in real time.")



class TweetScoreResult(Handler):

    def post(self):
        query = self.request.get("query_1")

        if query == "":
            self.render('TweetScoreMain.html', error='Please enter a valid query', query=query)
        
        else:
            [list_tweets, total_results, time_search] = algorithm.getpasttweets(query)

            if list_tweets == -1:
                self.redirect('/error')
            
            elif list_tweets == -2  or list_tweets == []:
                self.render('TweetScoreMain.html', error='Too many requests. We apologize but this is a free service and rate limited by Twitter API (which has exceeded). Please try again in 5 minutes.')
            
            elif list_tweets == -3:
                self.render('TweetScoreMain.html', error='Unknown Error occured please try again, we apologize for inconvenience')

            elif list_tweets == -4:
                self.render('TweetScoreMain.html', error='Not enough results for your search query. Please try another query.')
            
            else:
                list_tweets_5 = algorithm.tweetformat(list_tweets)
                features = algorithm.getfeatures(list_tweets, list_tweets_5)
                [features_final, list_labels, num_labels] = algorithm.classification(features)
                [list_tweets_positive, list_tweets_negative, list_tweets_neutral] = algorithm.selecttweets(list_tweets, list_labels, features_final, num_labels)
                popularity_score = algorithm.popularityscore(num_labels, total_results, time_search)
                popularity_score_float = float(popularity_score)

                plotdata = [['Objectivity Score', 'Positive', 'Negative', 'Neutral']]
                for i in range(0,len(list_labels)):
                    obj_score = features_final[i][0]
                    subj_score = features_final[i][1]
                    if list_labels[i] == "1":
                        plotdata.append([obj_score, subj_score, None, None])
                    elif list_labels[i] == "-1":
                        plotdata.append([obj_score, None, subj_score, None])
                    elif list_labels[i] == "0":
                        plotdata.append([obj_score, None, None, subj_score])

                self.render('graph_tweetscore_1.html', pos=num_labels[0], neg=num_labels[1], neu=num_labels[2], plotdata=json.dumps(plotdata))
                self.render('TweetScoreResult.html', query=query, score=popularity_score, score_float=popularity_score_float, tweets_pos=list_tweets_positive, tweets_neg=list_tweets_negative, tweets_neu=list_tweets_neutral)



class TweetComparisonResult(Handler):

    def post(self):
        query_1 = self.request.get("query_2")
        query_2 = self.request.get("query_3")
        query_3 = self.request.get("query_4")

        cont = False    # for blocking some computations if some error while fetching tweets

        if (query_1 == "" and query_2 == "") or (query_2 == "" and query_3 == "") or (query_1 == "" and query_3 == ""):
            self.render('TweetComparisonMain.html', error="Enter Atleast 2 valid queries for comparison", query_1=query_1, query_2=query_2, query_3=query_3)

        else:
            popularity_score_overall = []
            popularity_score_overall_float = []
            num_labels_overall = []
            num_tweets_overall = []
            total_results_overall = []

            query_list = [query_1, query_2, query_3]
            
            if "" in query_list:
                query_list.remove("")

            for index_query in range(0,len(query_list)):
                query = query_list[index_query]
                
                [list_tweets, total_results, time_search] = algorithm.getpasttweets(query)

                if list_tweets == -1:
                    cont = false
                    self.redirect('/error')

                elif list_tweets == -2  or list_tweets == []:
                    self.render('TweetComparisonMain.html', error='Too many requests. We apologize but this is a free service and rate limited by Twitter API (which has exceeded). Please try again in 5 minutes.')
                
                elif list_tweets == -3:
                    self.render('TweetComparisonMain.html', error='Unknown Error occured please try again, we apologize for inconvenience')

                elif list_tweets == -4:
                    self.render('TweetComparisonMain.html', error='Not enough results for your search query: <b>%s</b>. Please try another query' %query, query_1=query_1, query_2=query_2, query_3=query_3)
                
                else:
                    list_tweets_5 = algorithm.tweetformat(list_tweets)
                    
                    features = algorithm.getfeatures(list_tweets, list_tweets_5)
                    [features_final, list_labels, num_labels] = algorithm.classification(features)
                    popularity_score = algorithm.popularityscore(num_labels, total_results, time_search)
                    
                    popularity_score_overall_float.append(float(popularity_score))
                    popularity_score_overall.append(popularity_score)
                    num_labels_overall.append(num_labels)
                    num_tweets_overall.append(len(list_tweets))
                    total_results_overall.append(total_results)

                    cont = True

                    # self.write("query: <b>%s</b>. Time: <b>%s</b> minutes<br>" %(query, time_search))

            if cont == True:
                max_index = popularity_score_overall_float.index(max(popularity_score_overall_float))
                max_query = query_list[max_index]

                self.render('graph_tweetcomparison_1.html', pos=num_labels_overall[0][0], neg=num_labels_overall[0][1], neu=num_labels_overall[0][2], query=query_list[0])
                self.render('graph_tweetcomparison_2.html', pos=num_labels_overall[1][0], neg=num_labels_overall[1][1], neu=num_labels_overall[1][2], query=query_list[1])
                if len(num_labels_overall) == 3:
                    self.render('graph_tweetcomparison_3.html', pos=num_labels_overall[2][0], neg=num_labels_overall[2][1], neu=num_labels_overall[2][2], query=query_list[2])
                    self.render('TweetComparisonResult.html', max_query=max_query, query_1=query_list[0], query_2=query_list[1], query_3=query_list[2], score_1=popularity_score_overall[0], score_2=popularity_score_overall[1], score_3=popularity_score_overall[2], score_1_float=popularity_score_overall_float[0], score_2_float=popularity_score_overall_float[1], score_3_float=popularity_score_overall_float[2])

                else:
                    self.render('TweetComparisonResult.html', max_query=max_query, query_1=query_list[0], query_2=query_list[1], query_3="", score_1=popularity_score_overall[0], score_2=popularity_score_overall[1], score_3="", score_1_float=popularity_score_overall_float[0], score_2_float=popularity_score_overall_float[1], score_3_float="")


class TweetStatsResult(Handler):

    def post(self):
        global query_cron_1
        list_score = []
        
        query = self.request.get('query_cron')
        limit = self.request.get('time_frame')

        cron_1_mapping = {"obama":1, "romney":2, "zardari":3, "imran khan":4, "pakistan":5, "star wars":6, "lord of the rings":7, "appl":8, "xbox":9, "ps3":10}
        query_number = cron_1_mapping[query]

        phrase = 'db.GqlQuery("SELECT * FROM PastDataH' + str(query_number) + ' ORDER BY date DESC")'
        results = eval(phrase)

        for result in results:
            list_score.append(result.score)

        limit = int(limit)
 
        if limit <= len(list_score):
            list_score_pruned = list_score[0:limit]
        else:
            list_score_pruned = list_score[:] + [0]*(limit - len(list_score))

        plotdata = [['Hours', query]]

        for i in range(limit-1,-1,-1):
            plotdata.append([-i, list_score_pruned[i]])

        self.render('graph_tweetstats_1.html', plotdata=json.dumps(plotdata))
        self.render('TweetStatsResult.html', query=query, error = 'ATTENTION: Due to severe rate limitations of new Twitter API, TweetStats is CURRENTLY OFFLINE. The data showed is past data (uptil March 2013) and is not being updated in real time.')


class PastDataH1(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH2(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH3(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH4(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH5(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH6(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH7(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH8(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH9(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)



class PastDataH10(db.Model):
    
    score = db.IntegerProperty(required = True)
    date = db.DateTimeProperty(auto_now_add=True)

                

class CronTask1(Handler):

    def get(self):
        global query_cron_1

        query_cron_1 = ["obama", "romney", "zardari", "imran khan", "pakistan", "star wars", "lord of the rings", "appl", "xbox", "ps3"]

        for index in range(1,len(query_cron_1)+1):
            query = query_cron_1[index-1]
            
############# NOTE: CURRENTLY TWEETSTATS IS TURNED OFF. TO TURN IT BACK ON UNCOMMMENT THE LINE BELOW AND COMMENT THE LINE BELOW THAT! ##############

            # [list_tweets, total_results, time_search] = algorithm.getpasttweets(query)
            list_tweets = [];
        
            if list_tweets == -1 or list_tweets == -2 or list_tweets == -3 or list_tweets == -4  or list_tweets == []:
                pass
        
            else:
                list_tweets_5 = algorithm.tweetformat(list_tweets)
                features = algorithm.getfeatures(list_tweets, list_tweets_5)
                [features_final, list_labels, num_labels] = algorithm.classification(features)
                popularity_score = algorithm.popularityscore(num_labels, total_results, time_search)

                popularity_score = int(float(popularity_score))
            
                phrase = "PastDataH" + str(index) + "(score = popularity_score)"
                data = eval(phrase)
                data.put()



class ErrorPage(Handler):

    def get(self):
        self.write('Well this is embarassing...<br>No donut for you!<br>Please try again later.')



app = webapp2.WSGIApplication([('/', MainPage), ('/crontask1', CronTask1),
                            ('/tweetscore', TweetScoreMain), ('/tweetscore_result', TweetScoreResult), 
                            ('/tweetcomparison', TweetComparisonMain), ('/tweetcomparison_result', TweetComparisonResult),
                            ('/tweetstats', TweetStatsMain), ('/tweetstats_result', TweetStatsResult), 
                            ('/contact', ContactPage), ('/about', AboutPage), ('/additionals', AdditionalsPage),
                            ('/error', ErrorPage)], debug=True)